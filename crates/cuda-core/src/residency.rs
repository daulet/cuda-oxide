/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory residency handles for CUDA-facing Rust programs.
//!
//! These types cover the ownership boundaries that ordinary device buffers do
//! not: managed memory, host memory with a device-visible address, and
//! registration of caller-owned host memory.

use std::fmt;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;

use crate::context::CudaContext;
use crate::device_buffer::DeviceCopy;
use crate::error::DriverError;

/// Managed CUDA memory owned by Rust.
///
/// The allocation has one address that can be passed to kernels as a
/// `CUdeviceptr` and accessed on the host through slices. Host access after GPU
/// writes, or GPU access after host writes, must still be ordered with stream
/// or context synchronization.
///
/// Drop does not synchronize. All GPU work that may reference the allocation
/// must complete before the buffer is dropped.
///
/// Managed buffers require `T: DeviceCopy`, so host-owned values such as
/// [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{CudaContext, ManagedBuffer};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<CudaContext>) {
/// let _ = ManagedBuffer::<String>::zeroed(ctx, 1);
/// # }
/// ```
pub struct ManagedBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    device_ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is CUDA-managed memory. Moving ownership is safe when
// `T` can be sent.
unsafe impl<T: DeviceCopy + Send> Send for ManagedBuffer<T> {}
// SAFETY: shared host access exposes only `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for ManagedBuffer<T> {}

impl<T: DeviceCopy> ManagedBuffer<T> {
    /// Allocates managed memory and fills it with zero bytes.
    pub fn zeroed(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let buffer = Self::allocate(ctx, len)?;
        if buffer.num_bytes != 0 {
            unsafe {
                std::ptr::write_bytes(buffer.ptr.as_ptr().cast::<u8>(), 0, buffer.num_bytes);
            }
        }
        Ok(buffer)
    }

    /// Allocates managed memory and copies `data` into the host view.
    pub fn from_slice(ctx: &Arc<CudaContext>, data: &[T]) -> Result<Self, DriverError> {
        let buffer = Self::allocate(ctx, data.len())?;
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr(), data.len());
            }
        }
        Ok(buffer)
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns the CUDA context used to allocate this buffer.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the device-visible pointer.
    ///
    /// Empty and zero-sized buffers return `0`.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Returns the host pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the mutable host pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the host view.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the mutable host view.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn allocate(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(len)?;
        let (ptr, device_ptr) = if num_bytes == 0 {
            (NonNull::dangling(), 0)
        } else {
            ctx.bind_to_thread()?;
            let device_ptr = unsafe { crate::memory::malloc_managed(num_bytes)? };
            let ptr = NonNull::new(device_ptr as *mut T).ok_or_else(invalid_value)?;
            (ptr, device_ptr)
        };

        Ok(Self {
            ptr,
            device_ptr,
            len,
            num_bytes,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: DeviceCopy> Drop for ManagedBuffer<T> {
    fn drop(&mut self) {
        if self.device_ptr != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::free_sync(self.device_ptr) });
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for ManagedBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedBuffer")
            .field("ptr", &self.ptr)
            .field("device_ptr", &self.device_ptr)
            .field("len", &self.len)
            .field("num_bytes", &self.num_bytes)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: DeviceCopy> AsRef<[T]> for ManagedBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: DeviceCopy> AsMut<[T]> for ManagedBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: DeviceCopy> Deref for ManagedBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DeviceCopy> DerefMut for ManagedBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Owned page-locked host memory that also has a device-visible address.
///
/// Kernels can receive [`cu_deviceptr`](Self::cu_deviceptr), while host code
/// accesses the same allocation through ordinary Rust slices.
///
/// Drop does not synchronize. All GPU work that may reference the allocation
/// must complete before the buffer is dropped.
///
/// Mapped host buffers require `T: DeviceCopy`, so host-owned values such as
/// [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{CudaContext, MappedHostBuffer};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<CudaContext>) {
/// let _ = MappedHostBuffer::<String>::zeroed(ctx, 1);
/// # }
/// ```
pub struct MappedHostBuffer<T: DeviceCopy> {
    ptr: NonNull<T>,
    device_ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// SAFETY: the allocation is host memory. Moving ownership is safe when `T` can
// be sent.
unsafe impl<T: DeviceCopy + Send> Send for MappedHostBuffer<T> {}
// SAFETY: shared host access exposes only `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for MappedHostBuffer<T> {}

impl<T: DeviceCopy> MappedHostBuffer<T> {
    /// Allocates mapped host memory and fills it with zero bytes.
    pub fn zeroed(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let buffer = Self::allocate(ctx, len)?;
        if buffer.num_bytes != 0 {
            unsafe {
                std::ptr::write_bytes(buffer.ptr.as_ptr().cast::<u8>(), 0, buffer.num_bytes);
            }
        }
        Ok(buffer)
    }

    /// Allocates mapped host memory and copies `data` into it.
    pub fn from_slice(ctx: &Arc<CudaContext>, data: &[T]) -> Result<Self, DriverError> {
        let buffer = Self::allocate(ctx, data.len())?;
        if !data.is_empty() {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr.as_ptr(), data.len());
            }
        }
        Ok(buffer)
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns the CUDA context used to allocate this buffer.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the device-visible pointer.
    ///
    /// Empty and zero-sized buffers return `0`.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Returns the host pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the mutable host pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the buffer as a host slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the buffer as a mutable host slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    fn allocate(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(len)?;
        let (ptr, device_ptr) = if num_bytes == 0 {
            (NonNull::dangling(), 0)
        } else {
            ctx.bind_to_thread()?;
            let raw = unsafe { crate::memory::malloc_mapped_host(num_bytes)? };
            let ptr = NonNull::new(raw.cast::<T>()).ok_or_else(invalid_value)?;
            let device_ptr = unsafe { crate::memory::host_get_device_pointer(raw)? };
            (ptr, device_ptr)
        };

        Ok(Self {
            ptr,
            device_ptr,
            len,
            num_bytes,
            ctx: ctx.clone(),
            _marker: PhantomData,
        })
    }
}

impl<T: DeviceCopy> Drop for MappedHostBuffer<T> {
    fn drop(&mut self) {
        if self.num_bytes != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::free_host(self.ptr.as_ptr().cast()) });
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for MappedHostBuffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MappedHostBuffer")
            .field("ptr", &self.ptr)
            .field("device_ptr", &self.device_ptr)
            .field("len", &self.len)
            .field("num_bytes", &self.num_bytes)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: DeviceCopy> AsRef<[T]> for MappedHostBuffer<T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: DeviceCopy> AsMut<[T]> for MappedHostBuffer<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: DeviceCopy> Deref for MappedHostBuffer<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DeviceCopy> DerefMut for MappedHostBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

/// Registration guard for caller-owned host memory.
///
/// The guard does not own the backing slice; it unregisters the slice from CUDA
/// on drop. The lifetime keeps the original mutable borrow alive for as long as
/// kernels may receive the device-visible pointer.
///
/// Drop does not synchronize. All GPU work that may reference the registered
/// range must complete before the guard is dropped.
///
/// Registered host memory requires `T: DeviceCopy`, so host-owned values such
/// as [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{CudaContext, RegisteredHostMemory};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<CudaContext>, data: &mut [String]) {
/// let _ = RegisteredHostMemory::new(ctx, data);
/// # }
/// ```
pub struct RegisteredHostMemory<'a, T: DeviceCopy> {
    ptr: NonNull<T>,
    device_ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    _borrow: PhantomData<&'a mut [T]>,
}

// SAFETY: the guard owns no allocation, but it owns the registration lifetime.
// Moving it is safe when `T` can be sent.
unsafe impl<T: DeviceCopy + Send> Send for RegisteredHostMemory<'_, T> {}
// SAFETY: shared host access exposes only `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for RegisteredHostMemory<'_, T> {}

impl<'a, T: DeviceCopy> RegisteredHostMemory<'a, T> {
    /// Registers `data` with CUDA and returns its device-visible pointer.
    pub fn new(ctx: &Arc<CudaContext>, data: &'a mut [T]) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(data.len())?;
        let (ptr, device_ptr) = if num_bytes == 0 {
            (NonNull::dangling(), 0)
        } else {
            ctx.bind_to_thread()?;
            let ptr = NonNull::new(data.as_mut_ptr()).ok_or_else(invalid_value)?;
            unsafe {
                crate::memory::host_register_mapped(ptr.as_ptr().cast(), num_bytes)?;
            }
            let device_ptr =
                unsafe { crate::memory::host_get_device_pointer(ptr.as_ptr().cast())? };
            (ptr, device_ptr)
        };

        Ok(Self {
            ptr,
            device_ptr,
            len: data.len(),
            num_bytes,
            ctx: ctx.clone(),
            _borrow: PhantomData,
        })
    }

    /// Number of elements in the registered range.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the registered range contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total registered size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    /// Returns the CUDA context used to register this range.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Returns the device-visible pointer.
    ///
    /// Empty and zero-sized registered ranges return `0`.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.device_ptr
    }

    /// Returns the host pointer.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Returns the mutable host pointer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Returns the registered range as a host slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Returns the registered range as a mutable host slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: DeviceCopy> Drop for RegisteredHostMemory<'_, T> {
    fn drop(&mut self) {
        if self.num_bytes != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::host_unregister(self.ptr.as_ptr().cast()) });
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for RegisteredHostMemory<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RegisteredHostMemory")
            .field("ptr", &self.ptr)
            .field("device_ptr", &self.device_ptr)
            .field("len", &self.len)
            .field("num_bytes", &self.num_bytes)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: DeviceCopy> AsRef<[T]> for RegisteredHostMemory<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: DeviceCopy> AsMut<[T]> for RegisteredHostMemory<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: DeviceCopy> Deref for RegisteredHostMemory<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T: DeviceCopy> DerefMut for RegisteredHostMemory<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

fn allocation_size<T>(len: usize) -> Result<usize, DriverError> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(invalid_value)
}

fn invalid_value() -> DriverError {
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
}
