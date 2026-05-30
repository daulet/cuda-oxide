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
use crate::stream::CudaStream;

/// Destination or processor location for managed-memory controls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryLocation {
    /// Host memory.
    Host,
    /// Device memory for a CUDA device handle.
    Device(cuda_bindings::CUdevice),
}

impl MemoryLocation {
    fn cu_location(self) -> cuda_bindings::CUmemLocation {
        let mut location: cuda_bindings::CUmemLocation = unsafe { std::mem::zeroed() };
        let (location_type, id) = match self {
            MemoryLocation::Host => (
                cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_HOST,
                0,
            ),
            MemoryLocation::Device(device) => (
                cuda_bindings::CUmemLocationType_enum_CU_MEM_LOCATION_TYPE_DEVICE,
                device,
            ),
        };

        location.type_ = location_type;
        // CUDA 13.2 wraps `id` in an anonymous union while older headers expose
        // it directly. The field sits immediately after `type_` in both layouts.
        unsafe {
            let base = &mut location as *mut _ as *mut u8;
            (base.add(4) as *mut i32).write(id);
        }
        location
    }
}

/// Managed-memory advice accepted by [`ManagedBuffer::advise`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryAdvice {
    /// Mark the range as mostly read by processors.
    SetReadMostly,
    /// Undo [`SetReadMostly`](Self::SetReadMostly).
    UnsetReadMostly,
    /// Set the preferred residency location.
    SetPreferredLocation(MemoryLocation),
    /// Clear the preferred residency location.
    UnsetPreferredLocation,
    /// Pre-map the range for access by a processor.
    SetAccessedBy(MemoryLocation),
    /// Remove an accessed-by mapping.
    UnsetAccessedBy(MemoryLocation),
}

impl MemoryAdvice {
    fn raw(self) -> (cuda_bindings::CUmem_advise, MemoryLocation) {
        match self {
            MemoryAdvice::SetReadMostly => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_SET_READ_MOSTLY,
                MemoryLocation::Host,
            ),
            MemoryAdvice::UnsetReadMostly => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_UNSET_READ_MOSTLY,
                MemoryLocation::Host,
            ),
            MemoryAdvice::SetPreferredLocation(location) => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_SET_PREFERRED_LOCATION,
                location,
            ),
            MemoryAdvice::UnsetPreferredLocation => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION,
                MemoryLocation::Host,
            ),
            MemoryAdvice::SetAccessedBy(location) => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_SET_ACCESSED_BY,
                location,
            ),
            MemoryAdvice::UnsetAccessedBy(location) => (
                cuda_bindings::CUmem_advise_enum_CU_MEM_ADVISE_UNSET_ACCESSED_BY,
                location,
            ),
        }
    }
}

/// Stream association for managed memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamAttachment {
    /// Any stream on any device may access the memory.
    Global,
    /// Host-only attachment until the association is changed again.
    Host,
    /// Only the selected non-default stream may access the memory.
    Single,
}

impl StreamAttachment {
    fn flag(self) -> cuda_bindings::CUmemAttach_flags {
        match self {
            StreamAttachment::Global => cuda_bindings::CUmemAttach_flags_enum_CU_MEM_ATTACH_GLOBAL,
            StreamAttachment::Host => cuda_bindings::CUmemAttach_flags_enum_CU_MEM_ATTACH_HOST,
            StreamAttachment::Single => cuda_bindings::CUmemAttach_flags_enum_CU_MEM_ATTACH_SINGLE,
        }
    }
}

/// Owned residency strategy selected by an application policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyStrategy {
    /// Allocate CUDA managed memory.
    Managed,
    /// Allocate mapped page-locked host memory.
    MappedHost,
}

/// Allocation request passed to residency policy closures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyRequest {
    len: usize,
    num_bytes: usize,
}

impl ResidencyRequest {
    /// Number of requested elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the request has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Requested byte size after checked `len * size_of::<T>()` arithmetic.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.num_bytes
    }

    fn new<T>(len: usize) -> Result<Self, DriverError> {
        Ok(Self {
            len,
            num_bytes: allocation_size::<T>(len)?,
        })
    }
}

/// Owned buffer selected by a residency policy.
#[derive(Debug)]
pub enum ResidencyBuffer<T: DeviceCopy> {
    /// Managed-memory allocation.
    Managed(ManagedBuffer<T>),
    /// Mapped-host allocation.
    MappedHost(MappedHostBuffer<T>),
}

impl<T: DeviceCopy> ResidencyBuffer<T> {
    /// Allocates zeroed memory using `choose` to select the residency strategy.
    pub fn zeroed_with(
        ctx: &Arc<CudaContext>,
        len: usize,
        choose: impl FnOnce(ResidencyRequest) -> ResidencyStrategy,
    ) -> Result<Self, DriverError> {
        let request = ResidencyRequest::new::<T>(len)?;
        match choose(request) {
            ResidencyStrategy::Managed => ManagedBuffer::zeroed(ctx, len).map(Self::Managed),
            ResidencyStrategy::MappedHost => {
                MappedHostBuffer::zeroed(ctx, len).map(Self::MappedHost)
            }
        }
    }

    /// Allocates memory using `choose` and copies `data` into the selected
    /// residency strategy.
    pub fn from_slice_with(
        ctx: &Arc<CudaContext>,
        data: &[T],
        choose: impl FnOnce(ResidencyRequest) -> ResidencyStrategy,
    ) -> Result<Self, DriverError> {
        let request = ResidencyRequest::new::<T>(data.len())?;
        match choose(request) {
            ResidencyStrategy::Managed => ManagedBuffer::from_slice(ctx, data).map(Self::Managed),
            ResidencyStrategy::MappedHost => {
                MappedHostBuffer::from_slice(ctx, data).map(Self::MappedHost)
            }
        }
    }

    /// Number of elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Managed(buffer) => buffer.len(),
            Self::MappedHost(buffer) => buffer.len(),
        }
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total size in bytes.
    #[inline]
    pub fn num_bytes(&self) -> usize {
        match self {
            Self::Managed(buffer) => buffer.num_bytes(),
            Self::MappedHost(buffer) => buffer.num_bytes(),
        }
    }

    /// Returns the CUDA context used to allocate this buffer.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        match self {
            Self::Managed(buffer) => buffer.context(),
            Self::MappedHost(buffer) => buffer.context(),
        }
    }

    /// Returns the device-visible pointer.
    ///
    /// Empty and zero-sized buffers return `0`.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        match self {
            Self::Managed(buffer) => buffer.cu_deviceptr(),
            Self::MappedHost(buffer) => buffer.cu_deviceptr(),
        }
    }

    /// Returns the buffer as a host slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Managed(buffer) => buffer.as_slice(),
            Self::MappedHost(buffer) => buffer.as_slice(),
        }
    }

    /// Returns the buffer as a mutable host slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Managed(buffer) => buffer.as_mut_slice(),
            Self::MappedHost(buffer) => buffer.as_mut_slice(),
        }
    }
}

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

    /// Applies Unified Memory advice to the whole allocation.
    ///
    /// Empty and zero-sized buffers have no driver-side allocation, so this is
    /// a no-op for them.
    pub fn advise(&self, advice: MemoryAdvice) -> Result<(), DriverError> {
        if self.num_bytes == 0 {
            return Ok(());
        }

        self.ctx.bind_to_thread()?;
        let (advice, location) = advice.raw();
        unsafe {
            crate::memory::mem_advise(
                self.device_ptr,
                self.num_bytes,
                advice,
                location.cu_location(),
            )
        }
    }

    /// Enqueues a prefetch of the whole allocation to `location` on `stream`.
    ///
    /// Empty and zero-sized buffers have no driver-side allocation, so this is
    /// a no-op for them.
    pub fn prefetch_to(
        &self,
        stream: &CudaStream,
        location: MemoryLocation,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(&self.ctx, stream.context()),
            "managed buffer and stream must belong to the same CUDA context"
        );
        if self.num_bytes == 0 {
            return Ok(());
        }

        stream.context().bind_to_thread()?;
        unsafe {
            crate::memory::mem_prefetch_async(
                self.device_ptr,
                self.num_bytes,
                location.cu_location(),
                stream.cu_stream(),
            )
        }
    }

    /// Enqueues a stream association change for the whole allocation.
    ///
    /// Empty and zero-sized buffers have no driver-side allocation, so this is
    /// a no-op for them.
    pub fn attach_to_stream(
        &self,
        stream: &CudaStream,
        attachment: StreamAttachment,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(&self.ctx, stream.context()),
            "managed buffer and stream must belong to the same CUDA context"
        );
        assert!(
            !(stream.cu_stream().is_null() && attachment == StreamAttachment::Single),
            "single-stream managed-memory attachment requires a non-default stream"
        );
        if self.num_bytes == 0 {
            return Ok(());
        }

        stream.context().bind_to_thread()?;
        unsafe {
            crate::memory::stream_attach_mem_async(
                stream.cu_stream(),
                self.device_ptr,
                self.num_bytes,
                attachment.flag(),
            )
        }
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
                match unsafe { crate::memory::host_get_device_pointer(ptr.as_ptr().cast()) } {
                    Ok(device_ptr) => device_ptr,
                    Err(err) => {
                        let _ = unsafe { crate::memory::host_unregister(ptr.as_ptr().cast()) };
                        return Err(err);
                    }
                };
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

/// Registration guard for immutable host memory exposed to GPU reads.
///
/// The guard does not own the backing slice; it unregisters the slice from CUDA
/// on drop. The lifetime keeps the original shared borrow alive for as long as
/// kernels may receive the device-visible pointer. The CUDA read-only
/// registration flag is required because this type never grants mutable host
/// access to the mapped region.
///
/// Drop does not synchronize. All GPU work that may read the registered range
/// must complete before the guard is dropped.
///
/// Some devices or driver configurations do not support CUDA's read-only host
/// registration mode. In that case [`Self::new`] returns
/// `CUDA_ERROR_NOT_SUPPORTED`; callers choose any fallback policy explicitly.
///
/// Read-only registered host memory requires `T: DeviceCopy`, so host-owned
/// values such as [`String`] are rejected.
///
/// ```compile_fail
/// # use cuda_core::{CudaContext, ReadOnlyRegisteredHostMemory};
/// # fn rejects_non_device_copy(ctx: &std::sync::Arc<CudaContext>, data: &[String]) {
/// let _ = ReadOnlyRegisteredHostMemory::new(ctx, data);
/// # }
/// ```
pub struct ReadOnlyRegisteredHostMemory<'a, T: DeviceCopy> {
    ptr: NonNull<T>,
    device_ptr: CUdeviceptr,
    len: usize,
    num_bytes: usize,
    ctx: Arc<CudaContext>,
    _borrow: PhantomData<&'a [T]>,
}

// SAFETY: moving the guard transfers an immutable borrow across threads, which
// is safe only when the borrowed elements may be shared across threads.
unsafe impl<T: DeviceCopy + Sync> Send for ReadOnlyRegisteredHostMemory<'_, T> {}
// SAFETY: shared host access exposes only `&[T]`.
unsafe impl<T: DeviceCopy + Sync> Sync for ReadOnlyRegisteredHostMemory<'_, T> {}

impl<'a, T: DeviceCopy> ReadOnlyRegisteredHostMemory<'a, T> {
    /// Registers `data` as device-readable memory and returns its device pointer.
    ///
    /// Returns `CUDA_ERROR_NOT_SUPPORTED` when the device or driver does not
    /// implement read-only host registration.
    pub fn new(ctx: &Arc<CudaContext>, data: &'a [T]) -> Result<Self, DriverError> {
        let num_bytes = allocation_size::<T>(data.len())?;
        let (ptr, device_ptr) = if num_bytes == 0 {
            (NonNull::dangling(), 0)
        } else {
            ctx.bind_to_thread()?;
            let ptr = NonNull::new(data.as_ptr().cast_mut()).ok_or_else(invalid_value)?;
            unsafe {
                crate::memory::host_register_mapped_read_only(ptr.as_ptr().cast(), num_bytes)?;
            }
            let device_ptr =
                match unsafe { crate::memory::host_get_device_pointer(ptr.as_ptr().cast()) } {
                    Ok(device_ptr) => device_ptr,
                    Err(err) => {
                        let _ = unsafe { crate::memory::host_unregister(ptr.as_ptr().cast()) };
                        return Err(err);
                    }
                };
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

    /// Returns the registered range as a host slice.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: DeviceCopy> Drop for ReadOnlyRegisteredHostMemory<'_, T> {
    fn drop(&mut self) {
        if self.num_bytes != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::host_unregister(self.ptr.as_ptr().cast()) });
        }
    }
}

impl<T: DeviceCopy> fmt::Debug for ReadOnlyRegisteredHostMemory<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadOnlyRegisteredHostMemory")
            .field("ptr", &self.ptr)
            .field("device_ptr", &self.device_ptr)
            .field("len", &self.len)
            .field("num_bytes", &self.num_bytes)
            .field("ctx", &self.ctx)
            .finish()
    }
}

impl<T: DeviceCopy> AsRef<[T]> for ReadOnlyRegisteredHostMemory<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: DeviceCopy> Deref for ReadOnlyRegisteredHostMemory<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

fn allocation_size<T>(len: usize) -> Result<usize, DriverError> {
    len.checked_mul(std::mem::size_of::<T>())
        .ok_or_else(invalid_value)
}

fn invalid_value() -> DriverError {
    DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
}
