/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-scoped tensor-core MMA operations.
//!
//! These ops represent the per-warp `ldmatrix` / `mma.sync` path used for
//! `m16n8k16` f16-input, f32-accumulator matrix multiply.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::Context,
    context::Ptr,
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Load the A operand fragment for `m16n8k16` from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
#[pliron_op(
    name = "nvvm.mma_ldmatrix_m8n8_x4",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<4>],
)]
pub struct MmaLdMatrixM8N8X4Op;

impl MmaLdMatrixM8N8X4Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaLdMatrixM8N8X4Op { op }
    }
}

/// Load the B operand fragment for `m16n8k16` from shared memory.
///
/// PTX: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`
#[pliron_op(
    name = "nvvm.mma_ldmatrix_m8n8_x2_trans",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<2>],
)]
pub struct MmaLdMatrixM8N8X2TransOp;

impl MmaLdMatrixM8N8X2TransOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaLdMatrixM8N8X2TransOp { op }
    }
}

/// Execute one `m16n8k16` f16-input, f32-accumulator MMA step.
///
/// Operand order is `a0..a3`, `b0..b1`, `c0..c3`; results are `d0..d3`.
///
/// PTX: `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
#[pliron_op(
    name = "nvvm.mma_sync_m16n8k16_f32_f16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<10>, NResultsInterface<4>],
)]
pub struct MmaSyncM16N8K16F32F16Op;

impl MmaSyncM16N8K16F32F16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        MmaSyncM16N8K16F32F16Op { op }
    }
}

/// Register warp MMA operations with the context.
pub(super) fn register(ctx: &mut Context) {
    MmaLdMatrixM8N8X4Op::register(ctx);
    MmaLdMatrixM8N8X2TransOp::register(ctx);
    MmaSyncM16N8K16F32F16Op::register(ctx);
}
