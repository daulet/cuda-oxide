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
