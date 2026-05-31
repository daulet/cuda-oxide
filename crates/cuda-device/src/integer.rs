/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed integer arithmetic device intrinsics.

/// Compute four signed i8 products packed in each i32 operand and add `acc`.
///
/// Lowers to PTX `dp4a.s32.s32`.
#[inline(never)]
pub fn dp4a_i8(a: i32, b: i32, acc: i32) -> i32 {
    let _ = (a, b, acc);
    unreachable!("dp4a_i8 called outside CUDA kernel context")
}

/// Apply PTX `prmt.b32` with the fixed `0xba98` byte-sign selector.
///
/// This selector broadcasts the high bit of each input byte after packed
/// nonzero-mask preparation.
#[inline(never)]
pub fn prmt_b32_ba98(value: u32) -> u32 {
    let _ = value;
    unreachable!("prmt_b32_ba98 called outside CUDA kernel context")
}
