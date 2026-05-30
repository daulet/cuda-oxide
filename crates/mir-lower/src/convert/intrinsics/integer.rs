/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed integer arithmetic intrinsic conversion.

use crate::convert::intrinsics::common::call_intrinsic;
use dialect_llvm::types as llvm_types;
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
    let func_ty = llvm_types::FuncType::get(
        ctx,
        i32_ty.into(),
        vec![i32_ty.into(), i32_ty.into(), i32_ty.into()],
        false,
    );
    let call_op = call_intrinsic(ctx, rewriter, op, "llvm_nvvm_idp4a_s_s", func_ty, operands)?;
    rewriter.replace_operation(ctx, op, call_op);
    Ok(())
}
