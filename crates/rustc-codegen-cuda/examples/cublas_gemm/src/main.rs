/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! cuBLAS GEMM composed with a Rust-authored kernel on one stream.
//!
//! Build and run with:
//!   cargo oxide run cublas_gemm

use cuda_core::{
    Blas, CudaContext, DeviceBuffer, LaunchConfig, SgemmConfig, StridedBatchedSgemmConfig,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const EPSILON: f32 = 1.0e-4;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_bias(bias: f32, mut values: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(value) = values.get_mut(idx) {
            *value += bias;
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== cuBLAS Dense Linear Algebra Integration ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let blas = Blas::new(&ctx)?;
    let module = kernels::load(&ctx)?;

    run_regular_gemm(&stream, &blas, &module)?;
    run_batched_gemm(&stream, &blas, &module)?;

    println!("SUCCESS: cuBLAS GEMM paths matched CPU references");
    Ok(())
}

fn run_regular_gemm(
    stream: &cuda_core::CudaStream,
    blas: &Blas,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 4;
    let n = 5;
    let k = 3;
    let bias = 0.125;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 - 4.0) * 0.2).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 + 2.0) * 0.15).collect();
    let c_initial = vec![0.25; m * n];
    let mut expected = c_initial.clone();
    reference_sgemm(m, n, k, 1.1, &a, &b, 0.5, &mut expected);
    add_bias_host(&mut expected, bias);

    let a_dev = DeviceBuffer::from_host(stream, &a)?;
    let b_dev = DeviceBuffer::from_host(stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, &c_initial)?;

    let mut config = SgemmConfig::new(m, n, k);
    config.alpha = 1.1;
    config.beta = 0.5;
    blas.sgemm(stream, config, &a_dev, &b_dev, &mut c_dev)?;
    module.add_bias(
        stream,
        LaunchConfig::for_num_elems(c_dev.len() as u32),
        bias,
        &mut c_dev,
    )?;

    let actual = c_dev.to_host_vec(stream)?;
    assert_close("regular SGEMM", &actual, &expected);
    println!("regular SGEMM ok");
    Ok(())
}

fn run_batched_gemm(
    stream: &cuda_core::CudaStream,
    blas: &Blas,
    module: &kernels::LoadedModule,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 2;
    let n = 3;
    let k = 4;
    let batch_count = 3;
    let bias = -0.2;
    let mut config = StridedBatchedSgemmConfig::packed(m, n, k, batch_count)?;
    config.alpha = 0.75;
    config.beta = 0.25;

    let a: Vec<f32> = (0..config.stride_a * batch_count)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
        .collect();
    let b: Vec<f32> = (0..config.stride_b * batch_count)
        .map(|i| ((i % 9) as f32 - 3.0) * 0.2)
        .collect();
    let c_initial = vec![0.5; config.stride_c * batch_count];
    let mut expected = c_initial.clone();
    for batch in 0..batch_count {
        let a_offset = batch * config.stride_a;
        let b_offset = batch * config.stride_b;
        let c_offset = batch * config.stride_c;
        reference_sgemm(
            m,
            n,
            k,
            config.alpha,
            &a[a_offset..a_offset + config.stride_a],
            &b[b_offset..b_offset + config.stride_b],
            config.beta,
            &mut expected[c_offset..c_offset + config.stride_c],
        );
    }
    add_bias_host(&mut expected, bias);

    let a_dev = DeviceBuffer::from_host(stream, &a)?;
    let b_dev = DeviceBuffer::from_host(stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, &c_initial)?;

    blas.sgemm_strided_batched(stream, config, &a_dev, &b_dev, &mut c_dev)?;
    module.add_bias(
        stream,
        LaunchConfig::for_num_elems(c_dev.len() as u32),
        bias,
        &mut c_dev,
    )?;

    let actual = c_dev.to_host_vec(stream)?;
    assert_close("strided-batched SGEMM", &actual, &expected);
    println!("strided-batched SGEMM ok");
    Ok(())
}

fn reference_sgemm(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
) {
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            let index = row * n + col;
            c[index] = alpha * sum + beta * c[index];
        }
    }
}

fn add_bias_host(values: &mut [f32], bias: f32) {
    for value in values {
        *value += bias;
    }
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label} length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= EPSILON,
            "{label} mismatch at {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}
