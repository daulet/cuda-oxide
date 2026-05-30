/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]

pub use cuda_macros::{
    cluster_launch, convergent, cuda_module, device, gpu_printf, kernel, launch_bounds, pure,
    readonly,
};

// Re-export for convenience
pub mod atomic;
pub mod barrier;
pub mod clc;
pub mod cluster;
pub mod cooperative_groups;
pub mod cusimd;
pub mod debug;
pub mod disjoint;
pub mod fence;
pub mod grid;
pub mod integer;
pub mod lowp {
    pub use cuda_lowp::*;
}
pub mod mma;
pub mod selection;
pub mod shared;
pub mod tcgen05;
pub mod thread;
pub mod tma;
pub mod warp;
pub mod wgmma;

pub use barrier::{
    // Core type
    Barrier,
    BarrierToken,
    GeneralBarrier,
    Invalidated,
    // Typestate managed barrier
    ManagedBarrier,
    MmaBarrier,
    MmaBarrierHandle,
    Ready,
    // Kind markers
    TmaBarrier,
    TmaBarrier0,
    TmaBarrier1,
    // Type aliases
    TmaBarrierHandle,
    // State markers
    Uninit,
};
pub use cusimd::{CuSimd, Float2, Float4, TmemRegs4, TmemRegs32};
pub use disjoint::DisjointSlice;
pub use fence::*;
pub use lowp::{
    Fp4E2M1, Fp4x2E2M1, Fp4x4E2M1, Fp8E4M3, Fp8E5M2, Fp8x2E4M3, Fp8x2E5M2, Fp8x4E4M3, Fp8x4E5M2,
};
pub use mma::{MmaAccumulator, MmaOperandA, MmaOperandB};
pub use selection::{TopK, TopKEntry, block_topk_f32};
pub use shared::{DynamicSharedArray, SharedArray};
pub use tcgen05::{
    TensorMemoryHandle, TmemAddress, TmemDeallocated, TmemF32x4, TmemF32x32, TmemGuard, TmemReady,
    TmemUninit,
};
pub use thread::*;
pub use tma::TmaDescriptor;
