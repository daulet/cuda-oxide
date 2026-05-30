/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::sys;
use cuda_core::{
    CudaContext, ManagedBuffer, MappedHostBuffer, MemoryAdvice, MemoryLocation,
    ReadOnlyPageableHostMemory, ReadOnlyRegisteredHostMemory, RegisteredHostMemory,
    ResidencyBuffer, ResidencyStrategy, StreamAttachment,
};

#[test]
fn context_reports_device_memory_capacity() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let info = ctx.memory_info().expect("failed to query CUDA memory info");

    assert!(info.total_bytes > 0);
    assert!(info.free_bytes <= info.total_bytes);
}

#[test]
fn managed_buffer_exposes_host_slice_and_device_pointer() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut buffer = ManagedBuffer::<u32>::zeroed(&ctx, 4).expect("failed to allocate managed");

    assert_eq!(buffer.len(), 4);
    assert_eq!(buffer.num_bytes(), 16);
    assert!(!buffer.is_empty());
    assert_ne!(buffer.cu_deviceptr(), 0);
    assert_eq!(buffer.as_slice(), &[0, 0, 0, 0]);
    assert!(format!("{buffer:?}").contains("ManagedBuffer"));

    buffer.as_mut_slice().copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(&buffer[..], &[1, 2, 3, 4]);
}

#[test]
fn managed_buffer_from_slice_preserves_input() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let buffer =
        ManagedBuffer::<u32>::from_slice(&ctx, &[5, 6, 7, 8]).expect("failed to allocate managed");

    assert_eq!(buffer.as_slice(), &[5, 6, 7, 8]);
}

#[test]
fn managed_buffer_supports_advice_prefetch_and_stream_attachment() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let buffer =
        ManagedBuffer::<u32>::from_slice(&ctx, &[1, 2, 3, 4]).expect("failed to allocate managed");
    let device = MemoryLocation::Device(ctx.cu_device());

    buffer
        .advise(MemoryAdvice::SetPreferredLocation(device))
        .expect("failed to set preferred location");
    buffer
        .advise(MemoryAdvice::SetAccessedBy(device))
        .expect("failed to set accessed-by");
    buffer
        .advise(MemoryAdvice::SetReadMostly)
        .expect("failed to set read-mostly");
    buffer
        .prefetch_to(&stream, device)
        .expect("failed to prefetch to device");
    buffer
        .attach_to_stream(&stream, StreamAttachment::Single)
        .expect("failed to attach to stream");
    buffer
        .attach_to_stream(&stream, StreamAttachment::Global)
        .expect("failed to restore global attachment");
    stream.synchronize().expect("failed to synchronize stream");

    buffer
        .advise(MemoryAdvice::UnsetReadMostly)
        .expect("failed to unset read-mostly");
    buffer
        .advise(MemoryAdvice::UnsetAccessedBy(device))
        .expect("failed to unset accessed-by");
    buffer
        .advise(MemoryAdvice::UnsetPreferredLocation)
        .expect("failed to unset preferred location");
    buffer
        .prefetch_to(&stream, MemoryLocation::Host)
        .expect("failed to prefetch to host");
    buffer
        .attach_to_stream(&stream, StreamAttachment::Host)
        .expect("failed to attach to host");
    buffer
        .attach_to_stream(&stream, StreamAttachment::Global)
        .expect("failed to restore global attachment");
    stream.synchronize().expect("failed to synchronize stream");
}

#[test]
fn managed_buffer_controls_skip_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let buffer =
        ManagedBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty managed buffer");

    buffer
        .advise(MemoryAdvice::SetReadMostly)
        .expect("empty advice should be a no-op");
    buffer
        .prefetch_to(&stream, MemoryLocation::Device(ctx.cu_device()))
        .expect("empty prefetch should be a no-op");
    buffer
        .attach_to_stream(&stream, StreamAttachment::Single)
        .expect("empty stream attach should be a no-op");
}

#[test]
fn mapped_host_buffer_exposes_host_slice_and_device_pointer() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut buffer =
        MappedHostBuffer::<u32>::zeroed(&ctx, 4).expect("failed to allocate mapped host");

    assert_eq!(buffer.len(), 4);
    assert_eq!(buffer.num_bytes(), 16);
    assert_ne!(buffer.cu_deviceptr(), 0);
    assert_eq!(buffer.as_slice(), &[0, 0, 0, 0]);
    assert!(format!("{buffer:?}").contains("MappedHostBuffer"));

    buffer.as_mut_slice().copy_from_slice(&[10, 11, 12, 13]);
    assert_eq!(&buffer[..], &[10, 11, 12, 13]);
}

#[test]
fn mapped_host_buffer_from_slice_preserves_input() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let buffer = MappedHostBuffer::<u32>::from_slice(&ctx, &[14, 15, 16, 17])
        .expect("failed to allocate mapped host");

    assert_eq!(buffer.as_slice(), &[14, 15, 16, 17]);
}

#[test]
fn registered_host_memory_maps_existing_slice() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut data = [21_u32, 22, 23, 24];

    {
        let mut registered =
            RegisteredHostMemory::new(&ctx, &mut data).expect("failed to register host memory");

        assert_eq!(registered.len(), 4);
        assert_eq!(registered.num_bytes(), 16);
        assert_ne!(registered.cu_deviceptr(), 0);
        assert_eq!(registered.as_slice(), &[21, 22, 23, 24]);
        assert!(format!("{registered:?}").contains("RegisteredHostMemory"));

        registered.as_mut_slice()[1] = 99;
    }

    assert_eq!(data, [21, 99, 23, 24]);
}

#[test]
fn read_only_registered_host_memory_maps_or_reports_unsupported() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let data = [31_u32, 32, 33, 34];

    match ReadOnlyRegisteredHostMemory::new(&ctx, &data) {
        Ok(registered) => {
            eprintln!("read-only registered host memory: mapped");
            assert_eq!(registered.len(), 4);
            assert_eq!(registered.num_bytes(), 16);
            assert_ne!(registered.cu_deviceptr(), 0);
            assert_eq!(registered.as_slice(), &[31, 32, 33, 34]);
            assert!(format!("{registered:?}").contains("ReadOnlyRegisteredHostMemory"));
        }
        Err(err) => {
            eprintln!("read-only registered host memory: {err:?}");
            assert_eq!(err.0, sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED);
        }
    }

    assert_eq!(data, [31, 32, 33, 34]);
}

#[test]
fn read_only_pageable_host_memory_prefetches_when_supported() {
    const PAGE_BYTES: usize = 4096;

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");
    let data = vec![41_u8; PAGE_BYTES * 2];
    let page_delta = (PAGE_BYTES - (data.as_ptr() as usize % PAGE_BYTES)) % PAGE_BYTES;
    let page = &data[page_delta..page_delta + PAGE_BYTES];

    match ReadOnlyPageableHostMemory::new(&ctx, page) {
        Ok(pageable) => {
            let device = MemoryLocation::Device(ctx.cu_device());
            eprintln!(
                "read-only pageable host memory: supported host_page_tables={}",
                ctx.pageable_memory_access_uses_host_page_tables()
                    .expect("failed to query host page-table access")
            );
            pageable
                .advise(MemoryAdvice::SetReadMostly)
                .expect("failed to set read-mostly advice for pageable host memory");
            pageable
                .advise(MemoryAdvice::SetPreferredLocation(device))
                .expect("failed to set preferred location for pageable host memory");
            unsafe { pageable.prefetch_to(&stream, device) }
                .expect("failed to prefetch pageable host memory");
            stream.synchronize().expect("failed to synchronize stream");
            assert_eq!(pageable.len(), PAGE_BYTES);
            assert_eq!(pageable.num_bytes(), PAGE_BYTES);
            assert_ne!(pageable.cu_deviceptr(), 0);
            assert_eq!(pageable.as_slice(), page);
            assert!(format!("{pageable:?}").contains("ReadOnlyPageableHostMemory"));
        }
        Err(err) => {
            eprintln!("read-only pageable host memory: {err:?}");
            assert_eq!(err.0, sys::cudaError_enum_CUDA_ERROR_NOT_SUPPORTED);
        }
    }
}

#[test]
fn residency_handles_support_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut data: [u32; 0] = [];
    let read_only_data: [u32; 0] = [];

    let managed =
        ManagedBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty managed buffer");
    let mapped =
        MappedHostBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty mapped host");
    let registered =
        RegisteredHostMemory::new(&ctx, &mut data).expect("failed to register empty slice");
    let read_only_registered = ReadOnlyRegisteredHostMemory::new(&ctx, &read_only_data)
        .expect("failed to register empty read-only slice");
    let read_only_pageable = ReadOnlyPageableHostMemory::new(&ctx, &read_only_data)
        .expect("failed to borrow empty read-only pageable slice");

    assert!(managed.is_empty());
    assert_eq!(managed.num_bytes(), 0);
    assert_eq!(managed.cu_deviceptr(), 0);
    assert_eq!(managed.as_slice(), &[]);

    assert!(mapped.is_empty());
    assert_eq!(mapped.num_bytes(), 0);
    assert_eq!(mapped.cu_deviceptr(), 0);
    assert_eq!(mapped.as_slice(), &[]);

    assert!(registered.is_empty());
    assert_eq!(registered.num_bytes(), 0);
    assert_eq!(registered.cu_deviceptr(), 0);
    assert_eq!(registered.as_slice(), &[]);

    assert!(read_only_registered.is_empty());
    assert_eq!(read_only_registered.num_bytes(), 0);
    assert_eq!(read_only_registered.cu_deviceptr(), 0);
    assert_eq!(read_only_registered.as_slice(), &[]);

    assert!(read_only_pageable.is_empty());
    assert_eq!(read_only_pageable.num_bytes(), 0);
    assert_eq!(read_only_pageable.cu_deviceptr(), 0);
    assert_eq!(read_only_pageable.as_slice(), &[]);
}

#[test]
fn residency_handles_support_zero_sized_types() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut data = [(); 8];
    let read_only_data = [(); 8];

    let managed =
        ManagedBuffer::<()>::zeroed(&ctx, 8).expect("failed to create zst managed buffer");
    let mapped =
        MappedHostBuffer::<()>::zeroed(&ctx, 8).expect("failed to create zst mapped host buffer");
    let registered =
        RegisteredHostMemory::new(&ctx, &mut data).expect("failed to register zst slice");
    let read_only_registered = ReadOnlyRegisteredHostMemory::new(&ctx, &read_only_data)
        .expect("failed to register read-only zst slice");
    let read_only_pageable = ReadOnlyPageableHostMemory::new(&ctx, &read_only_data)
        .expect("failed to borrow read-only pageable zst slice");

    assert_eq!(managed.len(), 8);
    assert_eq!(managed.num_bytes(), 0);
    assert_eq!(managed.as_slice(), &[(); 8]);

    assert_eq!(mapped.len(), 8);
    assert_eq!(mapped.num_bytes(), 0);
    assert_eq!(mapped.as_slice(), &[(); 8]);

    assert_eq!(registered.len(), 8);
    assert_eq!(registered.num_bytes(), 0);
    assert_eq!(registered.as_slice(), &[(); 8]);

    assert_eq!(read_only_registered.len(), 8);
    assert_eq!(read_only_registered.num_bytes(), 0);
    assert_eq!(read_only_registered.as_slice(), &[(); 8]);

    assert_eq!(read_only_pageable.len(), 8);
    assert_eq!(read_only_pageable.num_bytes(), 0);
    assert_eq!(read_only_pageable.as_slice(), &[(); 8]);
}

#[test]
fn residency_policy_selects_managed_or_mapped_host() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");

    let managed = ResidencyBuffer::<u32>::zeroed_with(&ctx, 4, |request| {
        assert_eq!(request.len(), 4);
        assert_eq!(request.num_bytes(), 16);
        ResidencyStrategy::Managed
    })
    .expect("failed to allocate policy-selected managed buffer");
    assert!(matches!(managed, ResidencyBuffer::Managed(_)));
    assert_eq!(managed.as_slice(), &[0, 0, 0, 0]);

    let mapped = ResidencyBuffer::<u32>::from_slice_with(&ctx, &[7, 8, 9, 10], |request| {
        assert!(!request.is_empty());
        assert_eq!(request.num_bytes(), 16);
        ResidencyStrategy::MappedHost
    })
    .expect("failed to allocate policy-selected mapped host buffer");
    assert!(matches!(mapped, ResidencyBuffer::MappedHost(_)));
    assert_eq!(mapped.as_slice(), &[7, 8, 9, 10]);
    assert_eq!(mapped.len(), 4);
    assert_eq!(mapped.num_bytes(), 16);
    assert_ne!(mapped.cu_deviceptr(), 0);
    assert!(format!("{mapped:?}").contains("MappedHost"));
}
