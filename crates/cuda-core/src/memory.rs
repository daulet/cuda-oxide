/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA device memory allocation and transfer operations.
//!
//! Provides both **stream-ordered** (`*_async`) and **synchronous** (`*_sync`)
//! variants. The async functions enqueue the operation on a stream and return
//! immediately; the transfer completes in stream order. The sync variants block
//! the calling thread until the operation finishes.
//!
//! All functions in this module are `unsafe` because they operate on raw device
//! pointers whose validity cannot be checked at compile time.

use crate::error::{DriverError, IntoResult};
use cuda_bindings::CUdeviceptr;
use std::ffi::c_void;
use std::mem::MaybeUninit;

/// Allocates `num_bytes` of device memory on `stream` using the stream-ordered
/// allocator (`cuMemAllocAsync`).
///
/// The returned pointer is usable by any kernel or memcpy enqueued on `stream`
/// after this call. Pair with [`free_async`] on the same (or a synchronized)
/// stream.
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `stream` must be a valid `CUstream` from the current context.
/// - `num_bytes` must not exceed the device memory pool limits.
pub unsafe fn malloc_async(
    stream: cuda_bindings::CUstream,
    num_bytes: usize,
) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAllocAsync(dev_ptr.as_mut_ptr(), num_bytes, stream).result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Frees device memory previously allocated with [`malloc_async`].
///
/// The free is enqueued on `stream` and completes in stream order. The pointer
/// must not be accessed by any work enqueued after this call on the same stream.
///
/// # Safety
///
/// - `dptr` must have been returned by [`malloc_async`] and not yet freed.
/// - `stream` must be a valid `CUstream` from the same context as the
///   allocation.
pub unsafe fn free_async(
    dptr: CUdeviceptr,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFreeAsync(dptr, stream) }.result()
}

/// Allocates `num_bytes` of device memory synchronously (`cuMemAlloc`).
///
/// Blocks the calling thread until the allocation completes. Pair with
/// [`free_sync`].
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `num_bytes` must not exceed available device memory.
pub unsafe fn malloc_sync(num_bytes: usize) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAlloc_v2(dev_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Frees device memory previously allocated with [`malloc_sync`].
///
/// Blocks the calling thread. All pending GPU work referencing `dptr` must have
/// completed before this call.
///
/// # Safety
///
/// - `dptr` must have been returned by [`malloc_sync`] and not yet freed.
/// - No in-flight GPU operations may reference `dptr`.
pub unsafe fn free_sync(dptr: CUdeviceptr) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFree_v2(dptr) }.result()
}

/// Allocates `num_bytes` of CUDA managed memory with global attachment.
///
/// Managed memory is accessible through the returned `CUdeviceptr` from GPU
/// work and through the corresponding host address after stream or context
/// synchronization. Pair with [`free_sync`].
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `num_bytes` must not exceed the managed-memory capacity available to the
///   driver. Passing zero bytes is not useful and the CUDA driver reports it as
///   an error.
pub unsafe fn malloc_managed(num_bytes: usize) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAllocManaged(
            dev_ptr.as_mut_ptr(),
            num_bytes,
            cuda_bindings::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
        )
        .result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Copies `num_bytes` from host memory at `src` to device memory at `dst`,
/// enqueued on `stream` (host-to-device, async).
///
/// The host buffer at `src` must remain valid and unmodified until the copy
/// completes (i.e., until a synchronization point on `stream`). For
/// guaranteed asynchronous behavior, `src` should point to page-locked
/// (pinned) host memory.
///
/// # Safety
///
/// - `dst` must be a valid device pointer with at least `num_bytes` allocated.
/// - `src` must point to at least `num_bytes` of readable host memory.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memcpy_htod_async<T>(
    dst: CUdeviceptr,
    src: *const T,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyHtoDAsync_v2(dst, src as *const _, num_bytes, stream) }.result()
}

/// Copies `num_bytes` from device memory at `src` to host memory at `dst`,
/// enqueued on `stream` (device-to-host, async).
///
/// The host buffer at `dst` must not be read until the copy completes.
/// For guaranteed asynchronous behavior, `dst` should point to page-locked
/// (pinned) host memory.
///
/// # Safety
///
/// - `src` must be a valid device pointer with at least `num_bytes` accessible.
/// - `dst` must point to at least `num_bytes` of writable host memory.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memcpy_dtoh_async<T>(
    dst: *mut T,
    src: CUdeviceptr,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyDtoHAsync_v2(dst as *mut _, src, num_bytes, stream) }.result()
}

/// Copies `num_bytes` from device memory at `src` to device memory at `dst`,
/// enqueued on `stream` (device-to-device, async).
///
/// `src` and `dst` may reside on different devices if peer access is enabled.
///
/// # Safety
///
/// - Both `dst` and `src` must be valid device pointers with at least
///   `num_bytes` accessible.
/// - `stream` must be a valid `CUstream` from the current context.
/// - `dst` and `src` must not overlap unless they are identical.
pub unsafe fn memcpy_dtod_async(
    dst: CUdeviceptr,
    src: CUdeviceptr,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemcpyDtoDAsync_v2(dst, src, num_bytes, stream) }.result()
}

/// Sets `num_bytes` of device memory at `dptr` to `value`, enqueued on
/// `stream`.
///
/// Each byte in the range `[dptr, dptr + num_bytes)` is set to `value`.
///
/// # Safety
///
/// - `dptr` must be a valid device pointer with at least `num_bytes` allocated.
/// - `stream` must be a valid `CUstream` from the current context.
pub unsafe fn memset_d8_async(
    dptr: CUdeviceptr,
    value: u8,
    num_bytes: usize,
    stream: cuda_bindings::CUstream,
) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemsetD8Async(dptr, value, num_bytes, stream) }.result()
}

/// Allocates `num_bytes` of page-locked host memory.
///
/// Pinned host memory can be used as a staging area for CUDA transfers that
/// need higher bandwidth, and is required for host-device copies that are
/// intended to overlap with GPU work. Pair with [`free_host`].
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - `num_bytes` must not exceed the host memory available for page-locked
///   allocations. Passing zero bytes is not useful and the CUDA driver reports
///   it as an error.
pub unsafe fn malloc_host(num_bytes: usize) -> Result<*mut c_void, DriverError> {
    let mut host_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemAllocHost_v2(host_ptr.as_mut_ptr(), num_bytes).result()?;
        Ok(host_ptr.assume_init())
    }
}

/// Frees page-locked host memory previously allocated with [`malloc_host`].
///
/// # Safety
///
/// - `ptr` must have been returned by [`malloc_host`] and not yet freed.
/// - No in-flight CUDA transfer or kernel may reference `ptr`.
pub unsafe fn free_host(ptr: *mut c_void) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemFreeHost(ptr) }.result()
}

/// Allocates `num_bytes` of page-locked host memory that has a device-visible
/// address.
///
/// Pair with [`free_host`]. Use [`host_get_device_pointer`] to retrieve the
/// address kernels can use.
///
/// # Safety
///
/// - A CUDA context must be bound to the calling thread.
/// - The active device must support mapped host memory.
/// - `num_bytes` must not exceed the host memory available for page-locked
///   allocations. Passing zero bytes is not useful and the CUDA driver reports
///   it as an error.
pub unsafe fn malloc_mapped_host(num_bytes: usize) -> Result<*mut c_void, DriverError> {
    let mut host_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemHostAlloc(
            host_ptr.as_mut_ptr(),
            num_bytes,
            cuda_bindings::CU_MEMHOSTALLOC_DEVICEMAP,
        )
        .result()?;
        Ok(host_ptr.assume_init())
    }
}

/// Returns the device-visible address for mapped or registered host memory.
///
/// # Safety
///
/// - `ptr` must point into memory allocated by [`malloc_mapped_host`] or
///   registered by [`host_register_mapped`].
/// - A CUDA context must be bound to the calling thread.
pub unsafe fn host_get_device_pointer(ptr: *mut c_void) -> Result<CUdeviceptr, DriverError> {
    let mut dev_ptr = MaybeUninit::uninit();
    unsafe {
        cuda_bindings::cuMemHostGetDevicePointer_v2(dev_ptr.as_mut_ptr(), ptr, 0).result()?;
        Ok(dev_ptr.assume_init())
    }
}

/// Registers an existing host allocation and exposes it to the GPU address
/// space.
///
/// Pair with [`host_unregister`].
///
/// # Safety
///
/// - `ptr` must point to at least `num_bytes` of live host memory.
/// - The range must not already be registered with CUDA.
/// - The range must satisfy the active CUDA driver's host-registration
///   constraints.
/// - The active device must support mapped host memory.
/// - No in-flight CUDA work may reference the range after it is unregistered.
pub unsafe fn host_register_mapped(ptr: *mut c_void, num_bytes: usize) -> Result<(), DriverError> {
    unsafe {
        cuda_bindings::cuMemHostRegister_v2(
            ptr,
            num_bytes,
            cuda_bindings::CU_MEMHOSTREGISTER_DEVICEMAP,
        )
    }
    .result()
}

/// Unregisters host memory previously registered by [`host_register_mapped`].
///
/// # Safety
///
/// - `ptr` must be the base pointer originally passed to
///   [`host_register_mapped`].
/// - No in-flight CUDA work may reference the registered range.
pub unsafe fn host_unregister(ptr: *mut c_void) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuMemHostUnregister(ptr) }.result()
}
