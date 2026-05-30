/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA cuBLAS.
//!
//! This crate intentionally wraps only the cuBLAS surface cuda-oxide uses for
//! production dense linear algebra integration: handle lifecycle, version
//! probing, stream and math-mode binding, `Sgemm`, and
//! `SgemmStridedBatched`.
//!
//! # Library discovery
//!
//! [`LibCublas::load`] tries (in order):
//! 1. `LIBCUBLAS_PATH` env var, if set.
//! 2. The system loader (`libcublas.so.13`, `libcublas.so.12`, `libcublas.so`).
//! 3. `<root>/lib64/libcublas.so` for `<root>` in `CUDA_HOME`, `CUDA_PATH`,
//!    `/usr/local/cuda`, `/opt/cuda`.

use libloading::{Library, Symbol};
use std::ffi::{c_int, c_longlong, c_void};
use std::path::PathBuf;
use std::ptr;
use thiserror::Error;

// ============================================================================
// FFI types
// ============================================================================

/// Opaque cuBLAS handle (`cublasHandle_t`).
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct CublasHandle(*mut c_void);

/// cuBLAS operation selector (`cublasOperation_t`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Use the matrix as stored.
    None = 0,
    /// Transpose the matrix.
    Transpose = 1,
    /// Conjugate transpose. Equivalent to transpose for real-valued SGEMM.
    ConjugateTranspose = 2,
}

/// cuBLAS floating-point math behavior selector (`cublasMath_t`).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MathMode {
    /// Keep default cuBLAS math selection.
    Default = 0,
    /// Permit TF32 tensor-op execution where cuBLAS supports it.
    Tf32TensorOp = 3,
}

/// cuBLAS status values (`cublasStatus_t`).
#[allow(dead_code)]
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CublasStatus {
    Success = 0,
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16,
}

// ============================================================================
// Errors
// ============================================================================

/// All errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum CublasError {
    /// `libcublas.so` could not be located on this system. `tried` lists every
    /// path or SONAME that was probed, in order, joined by newlines.
    #[error(
        "libcublas.so could not be located. Set LIBCUBLAS_PATH or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibraryNotFound {
        /// Newline-joined list of paths and SONAMEs that were probed.
        tried: String,
    },

    /// `libcublas.so` was loaded, but `dlsym` failed to resolve a function
    /// this crate requires.
    #[error("libcublas.so was found but a required symbol is missing: {symbol}: {source}")]
    SymbolNotFound {
        /// Name of the missing cuBLAS function.
        symbol: &'static str,
        /// Underlying `libloading` error returned by `dlsym`.
        #[source]
        source: libloading::Error,
    },

    /// A cuBLAS call returned a non-success status.
    #[error("cuBLAS error in {operation}: {status:?} ({code})")]
    Call {
        /// Name of the cuBLAS function that failed.
        operation: &'static str,
        /// Raw `cublasStatus_t` integer.
        code: i32,
        /// Best-effort typed status value.
        status: CublasStatus,
    },
}

// ============================================================================
// Library handle
// ============================================================================

/// Loaded cuBLAS library plus resolved function pointers.
pub struct LibCublas {
    _lib: Library,
    create: unsafe extern "C" fn(*mut CublasHandle) -> CublasStatus,
    destroy: unsafe extern "C" fn(CublasHandle) -> CublasStatus,
    get_version: unsafe extern "C" fn(CublasHandle, *mut c_int) -> CublasStatus,
    set_stream: unsafe extern "C" fn(CublasHandle, *mut c_void) -> CublasStatus,
    set_math_mode: unsafe extern "C" fn(CublasHandle, MathMode) -> CublasStatus,
    sgemm: unsafe extern "C" fn(
        CublasHandle,
        Operation,
        Operation,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        c_int,
        *const f32,
        c_int,
        *const f32,
        *mut f32,
        c_int,
    ) -> CublasStatus,
    sgemm_strided_batched: unsafe extern "C" fn(
        CublasHandle,
        Operation,
        Operation,
        c_int,
        c_int,
        c_int,
        *const f32,
        *const f32,
        c_int,
        c_longlong,
        *const f32,
        c_int,
        c_longlong,
        *const f32,
        *mut f32,
        c_int,
        c_longlong,
        c_int,
    ) -> CublasStatus,
}

// SAFETY: The struct holds an owned `libloading::Library` plus immutable
// function pointers. Individual cuBLAS handles are represented by `Handle` and
// are not shared mutably through this type.
unsafe impl Send for LibCublas {}
// SAFETY: Same reasoning as `Send`; calls through function pointers are
// externally synchronized by cuBLAS per handle.
unsafe impl Sync for LibCublas {}

/// Resolve a required symbol to a function pointer of inferred type `T`.
///
/// # Safety
///
/// The returned function pointer is valid only while the borrowed `lib`
/// remains loaded. Callers store it in [`LibCublas`] alongside the owning
/// `Library`.
unsafe fn resolve<T: Copy>(lib: &Library, name: &'static str) -> Result<T, CublasError> {
    let sym: Symbol<T> =
        unsafe { lib.get(name.as_bytes()) }.map_err(|source| CublasError::SymbolNotFound {
            symbol: name,
            source,
        })?;
    Ok(unsafe { *sym.into_raw() })
}

impl LibCublas {
    /// Locate and load `libcublas.so` at runtime, then resolve every cuBLAS
    /// function this crate uses.
    pub fn load() -> Result<Self, CublasError> {
        let mut tried = Vec::new();
        let lib = open_library(&mut tried).ok_or_else(|| CublasError::LibraryNotFound {
            tried: tried.join("\n  "),
        })?;

        unsafe {
            Ok(Self {
                create: resolve(&lib, "cublasCreate_v2")?,
                destroy: resolve(&lib, "cublasDestroy_v2")?,
                get_version: resolve(&lib, "cublasGetVersion_v2")?,
                set_stream: resolve(&lib, "cublasSetStream_v2")?,
                set_math_mode: resolve(&lib, "cublasSetMathMode")?,
                sgemm: resolve(&lib, "cublasSgemm_v2")?,
                sgemm_strided_batched: resolve(&lib, "cublasSgemmStridedBatched")?,
                _lib: lib,
            })
        }
    }
}

// ============================================================================
// Handle (RAII)
// ============================================================================

/// RAII wrapper around a `cublasHandle_t`.
///
/// The handle owns the [`LibCublas`] that created it, so the library remains
/// loaded until after `cublasDestroy_v2` runs.
pub struct Handle {
    cublas: LibCublas,
    handle: CublasHandle,
}

impl Handle {
    /// Load cuBLAS and create a handle from it.
    pub fn load() -> Result<Self, CublasError> {
        Self::new(LibCublas::load()?)
    }

    /// Create a cuBLAS handle.
    pub fn new(cublas: LibCublas) -> Result<Self, CublasError> {
        let mut handle = CublasHandle(ptr::null_mut());
        let status = unsafe { (cublas.create)(&mut handle) };
        check(status, "cublasCreate_v2")?;
        Ok(Self { cublas, handle })
    }

    /// Query the cuBLAS version as the integer reported by
    /// `cublasGetVersion_v2`.
    pub fn version(&self) -> Result<i32, CublasError> {
        let mut version = 0;
        let status = unsafe { (self.cublas.get_version)(self.handle, &mut version) };
        check(status, "cublasGetVersion_v2")?;
        Ok(version)
    }

    /// Bind the handle to `stream`.
    ///
    /// Passing a null stream selects CUDA's default stream.
    pub fn set_stream(&self, stream: *mut c_void) -> Result<(), CublasError> {
        let status = unsafe { (self.cublas.set_stream)(self.handle, stream) };
        check(status, "cublasSetStream_v2")
    }

    /// Configure the cuBLAS floating-point math policy.
    pub fn set_math_mode(&self, mode: MathMode) -> Result<(), CublasError> {
        let status = unsafe { (self.cublas.set_math_mode)(self.handle, mode) };
        check(status, "cublasSetMathMode")
    }

    /// Enqueue `cublasSgemm_v2`.
    ///
    /// # Safety
    ///
    /// `a`, `b`, and `c` must be valid device pointers for the matrix shapes,
    /// strides, and transposition flags passed here. `alpha` and `beta` are
    /// host pointers because cuda-oxide keeps the default cuBLAS pointer mode.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm(
        &self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> Result<(), CublasError> {
        let status = unsafe {
            (self.cublas.sgemm)(
                self.handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                b,
                ldb,
                beta,
                c,
                ldc,
            )
        };
        check(status, "cublasSgemm_v2")
    }

    /// Enqueue `cublasSgemmStridedBatched`.
    ///
    /// # Safety
    ///
    /// Same pointer and shape requirements as [`Self::sgemm`], extended with
    /// valid element strides between consecutive matrices.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn sgemm_strided_batched(
        &self,
        transa: Operation,
        transb: Operation,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        stride_a: i64,
        b: *const f32,
        ldb: i32,
        stride_b: i64,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<(), CublasError> {
        let status = unsafe {
            (self.cublas.sgemm_strided_batched)(
                self.handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a,
                lda,
                stride_a,
                b,
                ldb,
                stride_b,
                beta,
                c,
                ldc,
                stride_c,
                batch_count,
            )
        };
        check(status, "cublasSgemmStridedBatched")
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        unsafe {
            (self.cublas.destroy)(self.handle);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn check(status: CublasStatus, operation: &'static str) -> Result<(), CublasError> {
    if status == CublasStatus::Success {
        return Ok(());
    }
    Err(CublasError::Call {
        operation,
        code: status as i32,
        status,
    })
}

fn open_library(tried: &mut Vec<String>) -> Option<Library> {
    if let Ok(path) = std::env::var("LIBCUBLAS_PATH") {
        if let Some(lib) = try_open(&path, tried) {
            return Some(lib);
        }
    }

    for soname in ["libcublas.so.13", "libcublas.so.12", "libcublas.so"] {
        if let Some(lib) = try_open(soname, tried) {
            return Some(lib);
        }
    }

    for root in cuda_roots() {
        let path = root.join("lib64/libcublas.so");
        if let Some(lib) = try_open(path, tried) {
            return Some(lib);
        }
    }

    None
}

fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(value) = std::env::var(var) {
            roots.push(PathBuf::from(value));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));
    roots
}

fn try_open<P: Into<PathBuf>>(path: P, tried: &mut Vec<String>) -> Option<Library> {
    let path = path.into();
    tried.push(path.display().to_string());
    unsafe { Library::new(&path).ok() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_library_and_queries_version() {
        let handle = Handle::load().expect("cuBLAS handle should be created");
        let version = handle
            .version()
            .expect("cuBLAS version should be available");
        assert!(version >= 12000, "unexpected cuBLAS version: {version}");
    }

    #[test]
    fn binds_default_stream() {
        let handle = Handle::load().expect("cuBLAS handle should be created");
        handle
            .set_stream(ptr::null_mut())
            .expect("cuBLAS should accept the default stream");
    }
}
