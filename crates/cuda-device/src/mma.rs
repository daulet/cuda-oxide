/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-scoped matrix multiply-accumulate (`mma.sync`) intrinsics.
//!
//! This module covers the per-warp tensor-core path used by Ampere-class and
//! consumer Blackwell targets where WGMMA or tcgen05 are not the right model.
//! The initial surface is the common `m16n8k16` shape with f16 inputs and f32
//! accumulators:
//!
//! ```text
//! D(16x8) = A(16x16) * B(16x8) + C(16x8)
//! ```
//!
//! Operand fragments are loaded from shared memory with `ldmatrix`, then
//! consumed by `mma.sync`. All functions in this module are warp-synchronous:
//! every lane in the warp must reach the same call with the same layout.

use crate::cusimd::CuSimd;

/// A fragment of matrix A for `m16n8k16` f16 MMA.
pub type MmaOperandA = CuSimd<u32, 4>;

/// A fragment of matrix B for `m16n8k16` f16 MMA.
pub type MmaOperandB = CuSimd<u32, 2>;

/// Four f32 accumulator registers per lane for `m16n8k16`.
pub type MmaAccumulator = CuSimd<f32, 4>;

/// Return a zeroed accumulator fragment.
#[inline(always)]
pub const fn zero_accumulator() -> MmaAccumulator {
    CuSimd::new([0.0, 0.0, 0.0, 0.0])
}

/// Load an A operand fragment from shared memory.
///
/// Lowers to:
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x4.shared.b16
/// ```
///
/// # Safety
///
/// - `ptr` must point into shared memory.
/// - The shared tile must satisfy the `ldmatrix` alignment and layout rules.
/// - All lanes in the warp must call this with the same dynamic control flow.
#[inline(never)]
pub unsafe fn load_a_m16n8k16(ptr: *const u8) -> MmaOperandA {
    let _ = ptr;
    unreachable!("load_a_m16n8k16 called outside CUDA kernel context")
}

/// Load a B operand fragment from shared memory.
///
/// Lowers to:
///
/// ```ptx
/// ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
/// ```
///
/// # Safety
///
/// - `ptr` must point into shared memory.
/// - The shared tile must satisfy the `ldmatrix` alignment and layout rules.
/// - All lanes in the warp must call this with the same dynamic control flow.
#[inline(never)]
pub unsafe fn load_b_m16n8k16(ptr: *const u8) -> MmaOperandB {
    let _ = ptr;
    unreachable!("load_b_m16n8k16 called outside CUDA kernel context")
}

/// Execute one warp-scoped `m16n8k16` f16 MMA step.
///
/// Computes `acc + a * b` and returns the updated accumulator fragment. Repeat
/// this call across K tiles to build larger GEMM shapes.
///
/// # Safety
///
/// - `a` and `b` must come from matching `m16n8k16` shared-memory tiles.
/// - All lanes in the warp must call this with the same dynamic control flow.
/// - The target GPU must support `mma.sync.aligned.m16n8k16` f16 MMA.
#[inline(always)]
pub unsafe fn mma_m16n8k16_f32_f16(
    acc: MmaAccumulator,
    a: MmaOperandA,
    b: MmaOperandB,
) -> MmaAccumulator {
    unsafe {
        mma_m16n8k16_f32_f16_raw(
            a.x(),
            a.y(),
            a.z(),
            a.w(),
            b.x(),
            b.y(),
            acc.x(),
            acc.y(),
            acc.z(),
            acc.w(),
        )
    }
}

#[doc(hidden)]
#[inline(never)]
pub unsafe fn mma_m16n8k16_f32_f16_raw(
    a0: u32,
    a1: u32,
    a2: u32,
    a3: u32,
    b0: u32,
    b1: u32,
    c0: f32,
    c1: f32,
    c2: f32,
    c3: f32,
) -> MmaAccumulator {
    let _ = (a0, a1, a2, a3, b0, b1, c0, c1, c2, c3);
    unreachable!("mma_m16n8k16_f32_f16_raw called outside CUDA kernel context")
}
