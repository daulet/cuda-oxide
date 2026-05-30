/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed integer arithmetic operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Four-way signed i8 dot product with signed i32 accumulation.
///
/// Corresponds to PTX `dp4a.s32.s32`.
#[pliron_op(
    name = "nvvm.idp4a_s_s",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>],
)]
pub struct Dp4aSignedSignedOp;

impl Dp4aSignedSignedOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Dp4aSignedSignedOp { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    Dp4aSignedSignedOp::register(ctx);
}
