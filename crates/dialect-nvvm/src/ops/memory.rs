/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit global-memory operations.

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    context::{Context, Ptr},
    op::Op,
    operation::Operation,
};
use pliron_derive::pliron_op;

/// Two-byte global-memory load with a zero-extended `u32` result.
#[pliron_op(
    name = "nvvm.ld_global_u16",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LoadGlobalU16Op;

impl LoadGlobalU16Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

/// Four-byte global-memory load.
#[pliron_op(
    name = "nvvm.ld_global_u32",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct LoadGlobalU32Op;

impl LoadGlobalU32Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

pub(super) fn register(ctx: &mut Context) {
    LoadGlobalU16Op::register(ctx);
    LoadGlobalU32Op::register(ctx);
}
