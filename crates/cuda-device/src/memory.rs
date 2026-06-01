/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit global-memory access intrinsics.

/// Load two aligned bytes from global memory and zero-extend the result to `u32`.
///
/// # Safety
///
/// `ptr` must point to readable, two-byte-aligned storage in CUDA global memory.
#[inline(never)]
pub unsafe fn load_global_u16(ptr: *const u16) -> u32 {
    let _ = ptr;
    unreachable!("load_global_u16 called outside CUDA kernel context")
}

/// Load four aligned bytes from global memory.
///
/// # Safety
///
/// `ptr` must point to readable, four-byte-aligned storage in CUDA global memory.
#[inline(never)]
pub unsafe fn load_global_u32(ptr: *const u32) -> u32 {
    let _ = ptr;
    unreachable!("load_global_u32 called outside CUDA kernel context")
}
