/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, ManagedBuffer, MappedHostBuffer, RegisteredHostMemory};

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
fn residency_handles_support_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut data: [u32; 0] = [];

    let managed =
        ManagedBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty managed buffer");
    let mapped =
        MappedHostBuffer::<u32>::zeroed(&ctx, 0).expect("failed to create empty mapped host");
    let registered =
        RegisteredHostMemory::new(&ctx, &mut data).expect("failed to register empty slice");

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
}

#[test]
fn residency_handles_support_zero_sized_types() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let mut data = [(); 8];

    let managed =
        ManagedBuffer::<()>::zeroed(&ctx, 8).expect("failed to create zst managed buffer");
    let mapped =
        MappedHostBuffer::<()>::zeroed(&ctx, 8).expect("failed to create zst mapped host buffer");
    let registered =
        RegisteredHostMemory::new(&ctx, &mut data).expect("failed to register zst slice");

    assert_eq!(managed.len(), 8);
    assert_eq!(managed.num_bytes(), 0);
    assert_eq!(managed.as_slice(), &[(); 8]);

    assert_eq!(mapped.len(), 8);
    assert_eq!(mapped.num_bytes(), 0);
    assert_eq!(mapped.as_slice(), &[(); 8]);

    assert_eq!(registered.len(), 8);
    assert_eq!(registered.num_bytes(), 0);
    assert_eq!(registered.as_slice(), &[(); 8]);
}
