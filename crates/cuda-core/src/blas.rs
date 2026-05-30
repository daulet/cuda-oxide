/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Stream-aware cuBLAS integration.
//!
//! The public API is row-major because cuda-oxide examples and host buffers use
//! ordinary Rust row-major `Vec<T>` layout. Internally, cuBLAS still receives
//! the equivalent column-major transposed problem.

use std::error;
use std::ffi::c_void;
use std::fmt::{self, Display, Formatter};
use std::sync::Arc;

use crate::context::CudaContext;
use crate::device_buffer::DeviceBuffer;
use crate::error::DriverError;
use crate::stream::CudaStream;

/// cuBLAS floating-point execution mode used by [`Blas`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlasMathMode {
    /// Use standard cuBLAS math behavior.
    Default,
    /// Permit TF32 tensor-op acceleration where cuBLAS supports it.
    Tf32TensorOp,
}

impl From<BlasMathMode> for cublas_sys::MathMode {
    fn from(mode: BlasMathMode) -> Self {
        match mode {
            BlasMathMode::Default => Self::Default,
            BlasMathMode::Tf32TensorOp => Self::Tf32TensorOp,
        }
    }
}

/// Row-major SGEMM configuration.
///
/// Computes `C = alpha * A * B + beta * C` where:
/// - `A` is `m x k`,
/// - `B` is `k x n`,
/// - `C` is `m x n`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SgemmConfig {
    /// Rows in `A` and `C`.
    pub m: usize,
    /// Columns in `B` and `C`.
    pub n: usize,
    /// Shared inner dimension.
    pub k: usize,
    /// Scale factor for `A * B`.
    pub alpha: f32,
    /// Scale factor for the existing `C` values.
    pub beta: f32,
}

impl SgemmConfig {
    /// Build a config with `alpha = 1.0` and `beta = 0.0`.
    pub fn new(m: usize, n: usize, k: usize) -> Self {
        Self {
            m,
            n,
            k,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

/// Row-major strided-batched SGEMM configuration.
///
/// Each batch computes `C_i = alpha * A_i * B_i + beta * C_i` with the same
/// per-matrix shapes as [`SgemmConfig`]. Strides are measured in `f32`
/// elements, not bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StridedBatchedSgemmConfig {
    /// Rows in each `A` and `C`.
    pub m: usize,
    /// Columns in each `B` and `C`.
    pub n: usize,
    /// Shared inner dimension.
    pub k: usize,
    /// Number of independent matrix products.
    pub batch_count: usize,
    /// Element stride between consecutive `A` matrices.
    pub stride_a: usize,
    /// Element stride between consecutive `B` matrices.
    pub stride_b: usize,
    /// Element stride between consecutive `C` matrices.
    pub stride_c: usize,
    /// Scale factor for `A * B`.
    pub alpha: f32,
    /// Scale factor for the existing `C` values.
    pub beta: f32,
}

impl StridedBatchedSgemmConfig {
    /// Build a packed row-major batched config with `alpha = 1.0` and
    /// `beta = 0.0`.
    pub fn packed(m: usize, n: usize, k: usize, batch_count: usize) -> Result<Self, BlasError> {
        Ok(Self {
            m,
            n,
            k,
            batch_count,
            stride_a: checked_mul("a matrix elements", m, k)?,
            stride_b: checked_mul("b matrix elements", k, n)?,
            stride_c: checked_mul("c matrix elements", m, n)?,
            alpha: 1.0,
            beta: 0.0,
        })
    }
}

/// Errors returned by [`Blas`].
#[derive(Debug)]
pub enum BlasError {
    /// CUDA driver operation failed.
    Driver(DriverError),
    /// cuBLAS operation failed.
    Cublas(cublas_sys::CublasError),
    /// A stream or buffer belongs to a different context than the `Blas`
    /// handle.
    ContextMismatch { resource: &'static str },
    /// Matrix dimensions and batch counts must be non-zero.
    InvalidZero { name: &'static str },
    /// A value does not fit in the integer type required by cuBLAS.
    TooLarge {
        name: &'static str,
        value: usize,
        max: usize,
    },
    /// A multiplication overflowed while computing required element counts.
    Overflow { name: &'static str },
    /// A device buffer is too small for the requested matrix view.
    BufferTooSmall {
        name: &'static str,
        required: usize,
        actual: usize,
    },
    /// A strided-batched matrix stride is smaller than one packed matrix.
    StrideTooSmall {
        name: &'static str,
        stride: usize,
        minimum: usize,
    },
}

impl Display for BlasError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(err) => write!(f, "CUDA driver error: {err}"),
            Self::Cublas(err) => write!(f, "cuBLAS error: {err}"),
            Self::ContextMismatch { resource } => {
                write!(f, "{resource} belongs to a different CUDA context")
            }
            Self::InvalidZero { name } => write!(f, "{name} must be non-zero"),
            Self::TooLarge { name, value, max } => {
                write!(f, "{name} value {value} exceeds cuBLAS maximum {max}")
            }
            Self::Overflow { name } => write!(f, "{name} overflowed"),
            Self::BufferTooSmall {
                name,
                required,
                actual,
            } => write!(
                f,
                "{name} buffer too small: need {required} elements, got {actual}"
            ),
            Self::StrideTooSmall {
                name,
                stride,
                minimum,
            } => write!(
                f,
                "{name} stride too small: need at least {minimum} elements, got {stride}"
            ),
        }
    }
}

impl error::Error for BlasError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Driver(err) => Some(err),
            Self::Cublas(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DriverError> for BlasError {
    fn from(value: DriverError) -> Self {
        Self::Driver(value)
    }
}

impl From<cublas_sys::CublasError> for BlasError {
    fn from(value: cublas_sys::CublasError) -> Self {
        Self::Cublas(value)
    }
}

/// cuBLAS handle tied to a cuda-oxide context.
pub struct Blas {
    handle: cublas_sys::Handle,
    ctx: Arc<CudaContext>,
}

impl Blas {
    /// Create a cuBLAS handle for `ctx`.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, BlasError> {
        ctx.bind_to_thread()?;
        Ok(Self {
            handle: cublas_sys::Handle::load()?,
            ctx: ctx.clone(),
        })
    }

    /// Query the cuBLAS version backing this handle.
    pub fn version(&self) -> Result<i32, BlasError> {
        Ok(self.handle.version()?)
    }

    /// Set the cuBLAS floating-point math policy for subsequent operations.
    pub fn set_math_mode(&self, mode: BlasMathMode) -> Result<(), BlasError> {
        self.ctx.bind_to_thread()?;
        Ok(self.handle.set_math_mode(mode.into())?)
    }

    /// Enqueue row-major SGEMM on `stream`.
    pub fn sgemm(
        &self,
        stream: &CudaStream,
        config: SgemmConfig,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<(), BlasError> {
        ensure_same_context(&self.ctx, a.context(), "a buffer")?;
        let dims = validate_sgemm(config, a, b, c)?;
        self.bind_stream(stream)?;

        // Row-major C(m,n)=A(m,k)*B(k,n) is column-major
        // C^T(n,m)=B^T(n,k)*A^T(k,m).
        unsafe {
            self.handle.sgemm(
                cublas_sys::Operation::None,
                cublas_sys::Operation::None,
                dims.n,
                dims.m,
                dims.k,
                &config.alpha,
                b.cu_deviceptr() as *const f32,
                dims.n,
                a.cu_deviceptr() as *const f32,
                dims.k,
                &config.beta,
                c.cu_deviceptr() as *mut f32,
                dims.n,
            )?;
        }
        Ok(())
    }

    /// Enqueue row-major mixed-precision GEMM with F16 inputs and F32 output.
    pub fn gemm_ex_f16_f32(
        &self,
        stream: &CudaStream,
        config: SgemmConfig,
        a: &DeviceBuffer<f16>,
        b: &DeviceBuffer<f16>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<(), BlasError> {
        ensure_same_context(&self.ctx, a.context(), "a buffer")?;
        let dims = validate_gemm_ex_f16_f32(config, a, b, c)?;
        self.bind_stream(stream)?;

        // Row-major C(m,n)=A(m,k)*B(k,n) is column-major
        // C^T(n,m)=B^T(n,k)*A^T(k,m).
        unsafe {
            self.handle.gemm_ex_f16_f32(
                cublas_sys::Operation::None,
                cublas_sys::Operation::None,
                dims.n,
                dims.m,
                dims.k,
                &config.alpha,
                b.cu_deviceptr() as *const c_void,
                dims.n,
                a.cu_deviceptr() as *const c_void,
                dims.k,
                &config.beta,
                c.cu_deviceptr() as *mut f32,
                dims.n,
            )?;
        }
        Ok(())
    }

    /// Enqueue row-major strided-batched SGEMM on `stream`.
    pub fn sgemm_strided_batched(
        &self,
        stream: &CudaStream,
        config: StridedBatchedSgemmConfig,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
    ) -> Result<(), BlasError> {
        ensure_same_context(&self.ctx, a.context(), "a buffer")?;
        let dims = validate_batched_sgemm(config, a, b, c)?;
        self.bind_stream(stream)?;

        unsafe {
            self.handle.sgemm_strided_batched(
                cublas_sys::Operation::None,
                cublas_sys::Operation::None,
                dims.n,
                dims.m,
                dims.k,
                &config.alpha,
                b.cu_deviceptr() as *const f32,
                dims.n,
                dims.stride_b,
                a.cu_deviceptr() as *const f32,
                dims.k,
                dims.stride_a,
                &config.beta,
                c.cu_deviceptr() as *mut f32,
                dims.n,
                dims.stride_c,
                dims.batch_count,
            )?;
        }
        Ok(())
    }

    fn bind_stream(&self, stream: &CudaStream) -> Result<(), BlasError> {
        ensure_same_context(&self.ctx, stream.context(), "stream")?;
        self.ctx.bind_to_thread()?;
        self.handle
            .set_stream(stream.cu_stream().cast::<c_void>())?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct SgemmDims {
    m: i32,
    n: i32,
    k: i32,
}

#[derive(Clone, Copy)]
struct BatchedSgemmDims {
    m: i32,
    n: i32,
    k: i32,
    batch_count: i32,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn validate_sgemm(
    config: SgemmConfig,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &DeviceBuffer<f32>,
) -> Result<SgemmDims, BlasError> {
    ensure_same_context(a.context(), b.context(), "b buffer")?;
    ensure_same_context(a.context(), c.context(), "c buffer")?;
    validate_gemm_lengths(config, a.len(), b.len(), c.len())
}

fn validate_gemm_ex_f16_f32(
    config: SgemmConfig,
    a: &DeviceBuffer<f16>,
    b: &DeviceBuffer<f16>,
    c: &DeviceBuffer<f32>,
) -> Result<SgemmDims, BlasError> {
    ensure_same_context(a.context(), b.context(), "b buffer")?;
    ensure_same_context(a.context(), c.context(), "c buffer")?;
    validate_gemm_lengths(config, a.len(), b.len(), c.len())
}

fn validate_gemm_lengths(
    config: SgemmConfig,
    a_len: usize,
    b_len: usize,
    c_len: usize,
) -> Result<SgemmDims, BlasError> {
    let m = to_nonzero_i32("m", config.m)?;
    let n = to_nonzero_i32("n", config.n)?;
    let k = to_nonzero_i32("k", config.k)?;

    let a_required = checked_mul("a matrix elements", config.m, config.k)?;
    let b_required = checked_mul("b matrix elements", config.k, config.n)?;
    let c_required = checked_mul("c matrix elements", config.m, config.n)?;
    ensure_len("a", a_len, a_required)?;
    ensure_len("b", b_len, b_required)?;
    ensure_len("c", c_len, c_required)?;

    Ok(SgemmDims { m, n, k })
}

fn validate_batched_sgemm(
    config: StridedBatchedSgemmConfig,
    a: &DeviceBuffer<f32>,
    b: &DeviceBuffer<f32>,
    c: &DeviceBuffer<f32>,
) -> Result<BatchedSgemmDims, BlasError> {
    ensure_same_context(a.context(), b.context(), "b buffer")?;
    ensure_same_context(a.context(), c.context(), "c buffer")?;

    let m = to_nonzero_i32("m", config.m)?;
    let n = to_nonzero_i32("n", config.n)?;
    let k = to_nonzero_i32("k", config.k)?;
    let batch_count = to_nonzero_i32("batch_count", config.batch_count)?;

    let a_matrix = checked_mul("a matrix elements", config.m, config.k)?;
    let b_matrix = checked_mul("b matrix elements", config.k, config.n)?;
    let c_matrix = checked_mul("c matrix elements", config.m, config.n)?;
    ensure_stride("a", config.stride_a, a_matrix)?;
    ensure_stride("b", config.stride_b, b_matrix)?;
    ensure_stride("c", config.stride_c, c_matrix)?;

    let a_required = strided_required(
        "a required elements",
        config.stride_a,
        a_matrix,
        config.batch_count,
    )?;
    let b_required = strided_required(
        "b required elements",
        config.stride_b,
        b_matrix,
        config.batch_count,
    )?;
    let c_required = strided_required(
        "c required elements",
        config.stride_c,
        c_matrix,
        config.batch_count,
    )?;
    ensure_len("a", a.len(), a_required)?;
    ensure_len("b", b.len(), b_required)?;
    ensure_len("c", c.len(), c_required)?;

    Ok(BatchedSgemmDims {
        m,
        n,
        k,
        batch_count,
        stride_a: to_i64("stride_a", config.stride_a)?,
        stride_b: to_i64("stride_b", config.stride_b)?,
        stride_c: to_i64("stride_c", config.stride_c)?,
    })
}

fn ensure_same_context(
    expected: &Arc<CudaContext>,
    actual: &Arc<CudaContext>,
    resource: &'static str,
) -> Result<(), BlasError> {
    if Arc::ptr_eq(expected, actual) {
        Ok(())
    } else {
        Err(BlasError::ContextMismatch { resource })
    }
}

fn to_nonzero_i32(name: &'static str, value: usize) -> Result<i32, BlasError> {
    if value == 0 {
        return Err(BlasError::InvalidZero { name });
    }
    if value > i32::MAX as usize {
        return Err(BlasError::TooLarge {
            name,
            value,
            max: i32::MAX as usize,
        });
    }
    Ok(value as i32)
}

fn to_i64(name: &'static str, value: usize) -> Result<i64, BlasError> {
    if value > i64::MAX as usize {
        return Err(BlasError::TooLarge {
            name,
            value,
            max: i64::MAX as usize,
        });
    }
    Ok(value as i64)
}

fn checked_mul(name: &'static str, lhs: usize, rhs: usize) -> Result<usize, BlasError> {
    lhs.checked_mul(rhs).ok_or(BlasError::Overflow { name })
}

fn checked_add(name: &'static str, lhs: usize, rhs: usize) -> Result<usize, BlasError> {
    lhs.checked_add(rhs).ok_or(BlasError::Overflow { name })
}

fn ensure_len(name: &'static str, actual: usize, required: usize) -> Result<(), BlasError> {
    if actual >= required {
        Ok(())
    } else {
        Err(BlasError::BufferTooSmall {
            name,
            required,
            actual,
        })
    }
}

fn ensure_stride(name: &'static str, stride: usize, minimum: usize) -> Result<(), BlasError> {
    if stride >= minimum {
        Ok(())
    } else {
        Err(BlasError::StrideTooSmall {
            name,
            stride,
            minimum,
        })
    }
}

fn strided_required(
    name: &'static str,
    stride: usize,
    matrix_elements: usize,
    batch_count: usize,
) -> Result<usize, BlasError> {
    let last_offset = checked_mul(name, stride, batch_count - 1)?;
    checked_add(name, last_offset, matrix_elements)
}
