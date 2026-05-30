/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]

use cuda_core::{
    Blas, BlasError, BlasMathMode, CudaContext, CudaStream, DeviceBuffer, ProjectionConfig,
    SgemmConfig, StridedBatchedSgemmConfig,
};

const EPSILON: f32 = 1.0e-4;

#[test]
fn blas_sgemm_paths_match_cpu_reference_and_validate_inputs()
-> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.new_stream()?;
    let blas = Blas::new(&ctx)?;

    blas.set_math_mode(BlasMathMode::Tf32TensorOp)?;
    check_sgemm_matches_cpu_reference(&stream, &blas)?;
    blas.set_math_mode(BlasMathMode::Default)?;
    check_gemm_ex_f16_f32_matches_cpu_reference(&stream, &blas)?;
    check_projection_layout_matches_cpu_reference(&stream, &blas)?;
    check_strided_batched_sgemm_matches_cpu_reference(&stream, &blas)?;
    check_sgemm_rejects_short_output_buffer(&stream, &blas)?;
    Ok(())
}

fn check_gemm_ex_f16_f32_matches_cpu_reference(
    stream: &CudaStream,
    blas: &Blas,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 3;
    let n = 2;
    let k = 4;
    let a = [
        f16::from_bits(0x3c00),
        f16::from_bits(0x3800),
        f16::from_bits(0xbc00),
        f16::from_bits(0x4000),
        f16::from_bits(0x3400),
        f16::from_bits(0xc000),
        f16::from_bits(0x3e00),
        f16::from_bits(0xb800),
        f16::from_bits(0x4200),
        f16::from_bits(0x3c00),
        f16::from_bits(0x0000),
        f16::from_bits(0xbc00),
    ];
    let b = [
        f16::from_bits(0x3800),
        f16::from_bits(0xbc00),
        f16::from_bits(0x4000),
        f16::from_bits(0x3400),
        f16::from_bits(0xb800),
        f16::from_bits(0x3e00),
        f16::from_bits(0x3c00),
        f16::from_bits(0xc000),
    ];
    let c_initial = vec![0.25; m * n];
    let mut expected = c_initial.clone();
    let a_f32 = a.map(|value| value as f32);
    let b_f32 = b.map(|value| value as f32);
    reference_sgemm(m, n, k, 1.0, &a_f32, &b_f32, 0.0, &mut expected);

    let a_dev = DeviceBuffer::from_host(stream, &a)?;
    let b_dev = DeviceBuffer::from_host(stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(stream, &c_initial)?;

    blas.gemm_ex_f16_f32(
        stream,
        SgemmConfig::new(m, n, k),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )?;
    let actual = c_dev.to_host_vec(stream)?;
    assert_close(&actual, &expected);
    Ok(())
}

fn check_projection_layout_matches_cpu_reference(
    stream: &CudaStream,
    blas: &Blas,
) -> Result<(), Box<dyn std::error::Error>> {
    let config = ProjectionConfig::new(4, 3, 2);
    let weights_f16 = [
        f16::from_bits(0x3c00),
        f16::from_bits(0x3800),
        f16::from_bits(0xbc00),
        f16::from_bits(0x4000),
        f16::from_bits(0x3400),
        f16::from_bits(0xc000),
        f16::from_bits(0x3e00),
        f16::from_bits(0xb800),
        f16::from_bits(0x4200),
        f16::from_bits(0x3c00),
        f16::from_bits(0x0000),
        f16::from_bits(0xbc00),
    ];
    let activations_f16 = [
        f16::from_bits(0x3c00),
        f16::from_bits(0xc000),
        f16::from_bits(0x3800),
        f16::from_bits(0x4000),
        f16::from_bits(0xb800),
        f16::from_bits(0x3e00),
        f16::from_bits(0xbc00),
        f16::from_bits(0x3400),
    ];
    let weights_f32 = weights_f16.map(|value| value as f32);
    let activations_f32 = activations_f16.map(|value| value as f32);
    let expected = reference_projection(config, &weights_f32, &activations_f32);

    let weights_f32_dev = DeviceBuffer::from_host(stream, &weights_f32)?;
    let activations_f32_dev = DeviceBuffer::from_host(stream, &activations_f32)?;
    let mut output_f32_dev = DeviceBuffer::<f32>::zeroed(stream, expected.len())?;
    blas.project_f32(
        stream,
        config,
        &weights_f32_dev,
        &activations_f32_dev,
        &mut output_f32_dev,
    )?;
    assert_close(&output_f32_dev.to_host_vec(stream)?, &expected);

    let weights_f16_dev = DeviceBuffer::from_host(stream, &weights_f16)?;
    let activations_f16_dev = DeviceBuffer::from_host(stream, &activations_f16)?;
    let mut output_f16_dev = DeviceBuffer::<f32>::zeroed(stream, expected.len())?;
    blas.project_f16_f32(
        stream,
        config,
        &weights_f16_dev,
        &activations_f16_dev,
        &mut output_f16_dev,
    )?;
    assert_close(&output_f16_dev.to_host_vec(stream)?, &expected);
    Ok(())
}

fn check_sgemm_matches_cpu_reference(
    stream: &CudaStream,
    blas: &Blas,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 3;
    let n = 4;
    let k = 2;
    let a: Vec<f32> = (0..m * k).map(|i| (i as f32 + 1.0) * 0.25).collect();
    let b: Vec<f32> = (0..k * n).map(|i| (i as f32 - 3.0) * 0.5).collect();
    let c_initial = vec![0.5; m * n];
    let mut expected = c_initial.clone();
    reference_sgemm(m, n, k, 1.25, &a, &b, 0.5, &mut expected);

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(&stream, &c_initial)?;

    let mut config = SgemmConfig::new(m, n, k);
    config.alpha = 1.25;
    config.beta = 0.5;
    blas.sgemm(&stream, config, &a_dev, &b_dev, &mut c_dev)?;

    let actual = c_dev.to_host_vec(&stream)?;
    assert_close(&actual, &expected);
    Ok(())
}

fn check_strided_batched_sgemm_matches_cpu_reference(
    stream: &CudaStream,
    blas: &Blas,
) -> Result<(), Box<dyn std::error::Error>> {
    let m = 2;
    let n = 3;
    let k = 4;
    let batch_count = 3;
    let mut config = StridedBatchedSgemmConfig::packed(m, n, k, batch_count)?;
    config.alpha = 0.75;
    config.beta = 0.25;

    let a: Vec<f32> = (0..config.stride_a * batch_count)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.2)
        .collect();
    let b: Vec<f32> = (0..config.stride_b * batch_count)
        .map(|i| ((i % 7) as f32 - 2.0) * 0.3)
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

    let a_dev = DeviceBuffer::from_host(&stream, &a)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b)?;
    let mut c_dev = DeviceBuffer::from_host(&stream, &c_initial)?;

    blas.sgemm_strided_batched(&stream, config, &a_dev, &b_dev, &mut c_dev)?;

    let actual = c_dev.to_host_vec(&stream)?;
    assert_close(&actual, &expected);
    Ok(())
}

fn check_sgemm_rejects_short_output_buffer(
    stream: &CudaStream,
    blas: &Blas,
) -> Result<(), Box<dyn std::error::Error>> {
    let a_dev = DeviceBuffer::from_host(&stream, &[1.0f32, 2.0, 3.0, 4.0])?;
    let b_dev = DeviceBuffer::from_host(&stream, &[1.0f32, 0.0, 0.0, 1.0])?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, 3)?;

    let err = blas
        .sgemm(
            &stream,
            SgemmConfig::new(2, 2, 2),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
        .expect_err("short output buffer should be rejected before cuBLAS");

    assert!(matches!(
        err,
        BlasError::BufferTooSmall {
            name: "c",
            required: 4,
            actual: 3
        }
    ));
    stream.synchronize()?;
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

fn reference_projection(
    config: ProjectionConfig,
    weights: &[f32],
    activations: &[f32],
) -> Vec<f32> {
    let mut output = Vec::with_capacity(config.out_dim * config.n_tokens);
    for token in activations.chunks_exact(config.in_dim) {
        for row in weights.chunks_exact(config.in_dim) {
            output.push(row.iter().zip(token).map(|(w, x)| w * x).sum());
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let error = (actual - expected).abs();
        assert!(
            error <= EPSILON,
            "index {index}: actual={actual}, expected={expected}, error={error}"
        );
    }
}
