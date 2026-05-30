/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed integer arithmetic intrinsic conversion.
//!
//! LLVM 18, used by the current CUDA build path, does not lower an NVVM DP4A
//! intrinsic. Emit PTX directly so the instruction remains usable there.

use crate::convert::intrinsics::common::inline_asm_convergent;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::rewriter::Rewriter;
use pliron::operation::Operation;
use pliron::result::Result;

pub(crate) fn convert_dp4a_i8(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 3 {
        return pliron::input_err_noloc!("dp4a requires 3 operands [a, b, acc]");
    }
    let asm_op = inline_asm_convergent(
        ctx,
        rewriter,
        i32_ty.into(),
        operands,
        "dp4a.s32.s32 $0, $1, $2, $3;",
        "=r,r,r,r",
    );
    rewriter.replace_operation(ctx, op, asm_op);
    Ok(())
}
