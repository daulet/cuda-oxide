/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Explicit global-memory intrinsic conversion.

use crate::convert::intrinsics::common::inline_asm_convergent;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

fn convert_load_global(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    mnemonic: &str,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 1 {
        return pliron::input_err_noloc!("global load requires 1 operand [ptr]");
    }

    let u32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let asm_template = format!("{mnemonic} $0, [$1];");
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        u32_ty.into(),
        operands,
        &asm_template,
        "=r,l,~{memory}",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}

pub(crate) fn convert_load_global_u16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_load_global(ctx, rewriter, op, "ld.global.u16")
}

pub(crate) fn convert_load_global_u32(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    convert_load_global(ctx, rewriter, op, "ld.global.u32")
}
