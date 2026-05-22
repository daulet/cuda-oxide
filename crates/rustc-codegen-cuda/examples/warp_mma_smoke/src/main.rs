/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-time smoke test for warp-scoped `mma.sync`.
//!
//! Build with:
//!   cargo oxide build warp_mma_smoke --arch sm_80

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::mma::{load_a_m16n8k16, load_b_m16n8k16, mma_m16n8k16_f32_f16, zero_accumulator};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn warp_mma_smoke(mut output: DisjointSlice<u32>) {
        static mut A_TILE: SharedArray<u16, 256, 16> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, 128, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let mut a_index = tid;
        while a_index < 256 {
            unsafe {
                A_TILE[a_index] = 0x3c00;
            }
            a_index += 32;
        }
        let mut b_index = tid;
        while b_index < 128 {
            unsafe {
                B_TILE[b_index] = 0x3c00;
            }
            b_index += 32;
        }
        thread::sync_threads();

        let lane = (tid & 31) as usize;
        let a_row = lane & 15;
        let b_row = lane & 7;
        let a_ptr = unsafe { (&raw const A_TILE).cast::<u16>().add(a_row * 16) }.cast::<u8>();
        let b_ptr = unsafe { (&raw const B_TILE).cast::<u16>().add(b_row * 8) }.cast::<u8>();

        let acc = zero_accumulator();
        let a = unsafe { load_a_m16n8k16(a_ptr) };
        let b = unsafe { load_b_m16n8k16(b_ptr) };
        let acc = unsafe { mma_m16n8k16_f32_f16(acc, a, b) };

        let idx = thread::index_1d();
        if let Some(slot) = output.get_mut(idx) {
            *slot = acc.x().to_bits();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let mut output = DeviceBuffer::<u32>::zeroed(&stream, 32)?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        module.warp_mma_smoke(&stream, config, &mut output)?;
    }
    stream.synchronize()?;
    println!("warp MMA smoke completed for {} lanes", output.len());
    Ok(())
}
