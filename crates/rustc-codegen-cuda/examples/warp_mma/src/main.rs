/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reference-checked warp-scoped tensor-core GEMM tile.
//!
//! Run with:
//!   cargo oxide run warp_mma

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::mma::{load_a_m16n8k16, load_b_m16n8k16, mma_m16n8k16_f32_f16, zero_accumulator};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;
use half::f16;

const M: usize = 16;
const N: usize = 8;
const K: usize = 32;
const MMA_K: usize = 16;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn warp_mma_tile(a: &[u16], b: &[u16], mut out: DisjointSlice<f32>) {
        static mut A_TILE: SharedArray<u16, { M * MMA_K }, 16> = SharedArray::UNINIT;
        static mut B_TILE: SharedArray<u16, { MMA_K * N }, 16> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let lane = tid & 31;
        let mut acc = zero_accumulator();

        let mut k_tile = 0usize;
        while k_tile < K {
            let mut a_idx = tid;
            while a_idx < M * MMA_K {
                let row = a_idx / MMA_K;
                let col = a_idx % MMA_K;
                unsafe {
                    A_TILE[a_idx] = a[row * K + k_tile + col];
                }
                a_idx += 32;
            }

            let mut b_idx = tid;
            while b_idx < MMA_K * N {
                let row = b_idx / N;
                let col = b_idx % N;
                unsafe {
                    B_TILE[b_idx] = b[(k_tile + row) * N + col];
                }
                b_idx += 32;
            }

            thread::sync_threads();

            let a_row = (lane & 7) + ((lane & 8) as usize);
            let a_col = if lane & 16 == 0 { 0 } else { 8 };
            let b_row = (lane & 7) + ((lane & 8) as usize);
            let a_ptr = unsafe { (&raw const A_TILE).cast::<u16>().add(a_row * MMA_K + a_col) }
                .cast::<u8>();
            let b_ptr = unsafe { (&raw const B_TILE).cast::<u16>().add(b_row * N) }.cast::<u8>();

            let a_frag = unsafe { load_a_m16n8k16(a_ptr) };
            let b_frag = unsafe { load_b_m16n8k16(b_ptr) };
            acc = unsafe { mma_m16n8k16_f32_f16(acc, a_frag, b_frag) };

            thread::sync_threads();
            k_tile += MMA_K;
        }

        let group_id = lane >> 2;
        let thread_in_group = lane & 3;
        let col_base = thread_in_group * 2;

        unsafe {
            *out.get_unchecked_mut(group_id * N + col_base) = acc.x();
            *out.get_unchecked_mut(group_id * N + col_base + 1) = acc.y();
            *out.get_unchecked_mut((group_id + 8) * N + col_base) = acc.z();
            *out.get_unchecked_mut((group_id + 8) * N + col_base + 1) = acc.w();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let a = build_a();
    let b = build_b();
    let expected = reference(&a, &b);

    let a_dev = DeviceBuffer::<u16>::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::<u16>::from_host(&stream, &b)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, M * N)?;

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        module.warp_mma_tile(&stream, config, &a_dev, &b_dev, &mut out_dev)?;
    }
    stream.synchronize()?;

    let got = out_dev.to_host_vec(&stream)?;
    let mut max_error = 0.0f32;
    let mut max_index = 0usize;
    for (i, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        let error = (actual - want).abs();
        if error > max_error {
            max_error = error;
            max_index = i;
        }
    }

    if max_error <= 1e-3 {
        println!(
            "SUCCESS: warp MMA tile matched CPU reference for {M}x{N}x{K}; max error {max_error:.3e}"
        );
        Ok(())
    } else {
        let row = max_index / N;
        let col = max_index % N;
        eprintln!(
            "FAIL: warp MMA mismatch at ({row}, {col}): got {}, expected {}, max error {max_error:.3e}",
            got[max_index], expected[max_index]
        );
        std::process::exit(1);
    }
}

fn build_a() -> Vec<u16> {
    let mut data = vec![0u16; M * K];
    for row in 0..M {
        for col in 0..K {
            let value = ((row * 3 + col * 5) % 7 + 1) as f32;
            data[row * K + col] = f16::from_f32(value).to_bits();
        }
    }
    data
}

fn build_b() -> Vec<u16> {
    let mut data = vec![0u16; K * N];
    for row in 0..K {
        for col in 0..N {
            let value = ((row * 2 + col * 3) % 11 + 1) as f32;
            data[row * N + col] = f16::from_f32(value).to_bits();
        }
    }
    data
}

fn reference(a: &[u16], b: &[u16]) -> Vec<f32> {
    let mut out = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0f32;
            for kk in 0..K {
                let lhs = f16::from_bits(a[row * K + kk]).to_f32();
                let rhs = f16::from_bits(b[kk * N + col]).to_f32();
                sum += lhs * rhs;
            }
            out[row * N + col] = sum;
        }
    }
    out
}
