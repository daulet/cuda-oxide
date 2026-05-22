/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory residency controls on a real CUDA kernel path.
//!
//! Build and run with:
//!   cargo oxide run memory_residency

use cuda_core::{
    CudaContext, LaunchConfig, ManagedBuffer, MappedHostBuffer, MemoryAdvice, MemoryLocation,
    RegisteredHostMemory, StreamAttachment,
};
use cuda_device::{cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn combine_resident(
        input: *const f32,
        mapped_bias: *const f32,
        registered_extra: *const f32,
        output: *mut f32,
        len: u64,
    ) {
        let idx = thread::index_1d().get();
        if idx < len as usize {
            unsafe {
                let value = *input.add(idx) * 2.0
                    + *mapped_bias.add(idx)
                    + *registered_extra.add(idx);
                *output.add(idx) = value;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Memory Residency Controls ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let module = kernels::load(&ctx)?;
    let device = MemoryLocation::Device(ctx.cu_device());

    const N: usize = 1024;
    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let bias_host: Vec<f32> = (0..N).map(|i| (i % 7) as f32).collect();
    let mut extra_host: Vec<f32> = (0..N).map(|i| (i % 3) as f32).collect();

    let input = ManagedBuffer::from_slice(&ctx, &input_host)?;
    let output = ManagedBuffer::<f32>::zeroed(&ctx, N)?;
    let mapped_bias = MappedHostBuffer::from_slice(&ctx, &bias_host)?;
    let registered_extra = RegisteredHostMemory::new(&ctx, &mut extra_host)?;

    input.advise(MemoryAdvice::SetReadMostly)?;
    input.advise(MemoryAdvice::SetPreferredLocation(device))?;
    output.advise(MemoryAdvice::SetPreferredLocation(device))?;

    input.prefetch_to(&stream, device)?;
    output.prefetch_to(&stream, device)?;
    input.attach_to_stream(&stream, StreamAttachment::Single)?;
    output.attach_to_stream(&stream, StreamAttachment::Single)?;

    module.combine_resident(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        input.cu_deviceptr() as *const f32,
        mapped_bias.cu_deviceptr() as *const f32,
        registered_extra.cu_deviceptr() as *const f32,
        output.cu_deviceptr() as *mut f32,
        N as u64,
    )?;

    output.prefetch_to(&stream, MemoryLocation::Host)?;
    input.attach_to_stream(&stream, StreamAttachment::Global)?;
    output.attach_to_stream(&stream, StreamAttachment::Global)?;
    stream.synchronize()?;
    drop(registered_extra);

    let errors = output
        .as_slice()
        .iter()
        .enumerate()
        .filter(|(i, value)| {
            let expected = input_host[*i] * 2.0 + bias_host[*i] + extra_host[*i];
            (**value - expected).abs() > 1e-5
        })
        .count();

    assert_eq!(errors, 0, "memory residency kernel produced {errors} errors");
    println!("SUCCESS: memory residency kernel produced {N} correct elements");
    Ok(())
}
