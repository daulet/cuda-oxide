/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-scoped `mma.sync` intrinsic conversion.

use dialect_llvm::{ops as llvm, types as llvm_types};
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::irbuild::dialect_conversion::{DialectConversionRewriter, OperandsInfo};
use pliron::irbuild::inserter::Inserter;
use pliron::irbuild::rewriter::Rewriter;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::result::Result;
use pliron::r#type::TypeObj;

fn extract_struct_results(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    struct_result: pliron::value::Value,
    len: u32,
) -> Result<Vec<pliron::value::Value>> {
    let mut values = Vec::with_capacity(len as usize);
    for i in 0..len {
        let extract_op = llvm::ExtractValueOp::new(ctx, struct_result, vec![i])
            .map_err(|e| pliron::input_error_noloc!("{}", e))?;
        rewriter.insert_operation(ctx, extract_op.get_operation());
        values.push(extract_op.get_operation().deref(ctx).get_result(0));
    }
    Ok(values)
}

/// Convert an A-fragment shared-memory load to `ldmatrix.x4`.
pub(crate) fn convert_ldmatrix_x4(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| i32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        vec![ptr],
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $4; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [%ptr32]; ",
            "}"
        ),
        "=r,=r,=r,=r,l,~{memory}",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    let struct_result = asm_op.deref(ctx).get_result(0);
    let values = extract_struct_results(ctx, rewriter, struct_result, 4)?;
    rewriter.replace_operation_with_values(ctx, op, values);
    Ok(())
}

/// Convert a B-fragment shared-memory load to `ldmatrix.x2.trans`.
pub(crate) fn convert_ldmatrix_x2_trans(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let ptr = op.deref(ctx).get_operand(0);
    let i32_ty = IntegerType::get(ctx, 32, Signedness::Signless);
    let field_types: Vec<Ptr<TypeObj>> = (0..2).map(|_| i32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        vec![ptr],
        concat!(
            "{ ",
            ".reg .u64 %ptr64; ",
            ".reg .u32 %ptr32; ",
            "cvta.to.shared.u64 %ptr64, $2; ",
            "cvt.u32.u64 %ptr32, %ptr64; ",
            "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [%ptr32]; ",
            "}"
        ),
        "=r,=r,l,~{memory}",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    let struct_result = asm_op.deref(ctx).get_result(0);
    let values = extract_struct_results(ctx, rewriter, struct_result, 2)?;
    rewriter.replace_operation_with_values(ctx, op, values);
    Ok(())
}

/// Convert one `m16n8k16` f16/f32 MMA step to `mma.sync`.
pub(crate) fn convert_mma_sync_m16n8k16_f32_f16(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    op: Ptr<Operation>,
    _operands_info: &OperandsInfo,
) -> Result<()> {
    let operands: Vec<_> = op.deref(ctx).operands().collect();
    if operands.len() != 10 {
        return pliron::input_err_noloc!("mma.sync m16n8k16 requires 10 operands");
    }

    let f32_ty = FP32Type::get(ctx);
    let field_types: Vec<Ptr<TypeObj>> = (0..4).map(|_| f32_ty.into()).collect();
    let struct_ty = llvm_types::StructType::get_unnamed(ctx, field_types);

    let inline_asm = llvm::InlineAsmOp::new_convergent(
        ctx,
        struct_ty.into(),
        operands,
        concat!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ",
            "{$0, $1, $2, $3}, ",
            "{$4, $5, $6, $7}, ",
            "{$8, $9}, ",
            "{$10, $11, $12, $13};"
        ),
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
    );

    let asm_op = inline_asm.get_operation();
    rewriter.insert_operation(ctx, asm_op);
    let struct_result = asm_op.deref(ctx).get_result(0);
    let values = extract_struct_results(ctx, rewriter, struct_result, 4)?;
    rewriter.replace_operation_with_values(ctx, op, values);
    Ok(())
}
