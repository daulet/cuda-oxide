/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-scoped `mma.sync` intrinsics.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::types;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::{MmaLdMatrixM8N8X2TransOp, MmaLdMatrixM8N8X4Op, MmaSyncM16N8K16F32F16Op};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{FP32Type, IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeObj;
use pliron::value::Value;
use rustc_public::mir;

fn destination_struct_type(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    loc: Location,
) -> TranslationResult<Ptr<TypeObj>> {
    let dest_rust_ty = match destination.ty(body.locals()) {
        Ok(t) => t,
        Err(e) => {
            return input_err!(
                loc,
                TranslationErr::unsupported(format!(
                    "failed to resolve destination type for intrinsic result: {e:?}"
                ))
            );
        }
    };
    types::translate_type(ctx, &dest_rust_ty)
}

#[allow(clippy::too_many_arguments)]
fn emit_cusimd_result(
    ctx: &mut Context,
    body: &mir::Body,
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    producer_op: Ptr<Operation>,
    results: Vec<Value>,
    elem_ty: Ptr<TypeObj>,
    len: u64,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
    no_target_msg: &str,
) -> TranslationResult<Ptr<Operation>> {
    let array_ty = dialect_mir::types::MirArrayType::get(ctx, elem_ty, len);
    let array_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructArrayOp::get_concrete_op_info(),
        vec![array_ty.into()],
        results,
        vec![],
        0,
    );
    array_op.deref_mut(ctx).set_loc(loc.clone());
    array_op.insert_after(ctx, producer_op);

    let struct_ty = destination_struct_type(ctx, body, destination, loc.clone())?;
    let array_result = array_op.deref(ctx).get_result(0);
    let struct_op = Operation::new(
        ctx,
        dialect_mir::ops::MirConstructStructOp::get_concrete_op_info(),
        vec![struct_ty],
        vec![array_result],
        vec![],
        0,
    );
    struct_op.deref_mut(ctx).set_loc(loc.clone());
    struct_op.insert_after(ctx, array_op);

    let struct_result = struct_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        struct_result,
        target,
        block_ptr,
        struct_op,
        value_map,
        block_map,
        loc,
        no_target_msg,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_load_a_m16n8k16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "load_a_m16n8k16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let result_types = (0..4).map(|_| u32_ty.into()).collect();
    let op = Operation::new(
        ctx,
        MmaLdMatrixM8N8X4Op::get_concrete_op_info(),
        result_types,
        vec![ptr],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let results = (0..4).map(|i| op.deref(ctx).get_result(i)).collect();
    emit_cusimd_result(
        ctx,
        body,
        destination,
        target,
        block_ptr,
        op,
        results,
        u32_ty.into(),
        4,
        value_map,
        block_map,
        loc,
        "load_a_m16n8k16 call without target block",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_load_b_m16n8k16(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 1 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "load_b_m16n8k16 expects 1 argument, got {}",
                args.len()
            ))
        );
    }

    let (ptr, last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let u32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
    let result_types = (0..2).map(|_| u32_ty.into()).collect();
    let op = Operation::new(
        ctx,
        MmaLdMatrixM8N8X2TransOp::get_concrete_op_info(),
        result_types,
        vec![ptr],
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let results = (0..2).map(|i| op.deref(ctx).get_result(i)).collect();
    emit_cusimd_result(
        ctx,
        body,
        destination,
        target,
        block_ptr,
        op,
        results,
        u32_ty.into(),
        2,
        value_map,
        block_map,
        loc,
        "load_b_m16n8k16 call without target block",
    )
}

#[allow(clippy::too_many_arguments)]
pub fn emit_mma_m16n8k16_f32_f16_raw(
    ctx: &mut Context,
    body: &mir::Body,
    args: &[mir::Operand],
    destination: &mir::Place,
    target: &Option<usize>,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    value_map: &mut ValueMap,
    block_map: &[Ptr<BasicBlock>],
    loc: Location,
) -> TranslationResult<Ptr<Operation>> {
    if args.len() != 10 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "mma_m16n8k16_f32_f16_raw expects 10 arguments, got {}",
                args.len()
            ))
        );
    }

    let mut last_op = prev_op;
    let mut operands = Vec::with_capacity(10);
    for arg in args {
        let (value, next_op) =
            rvalue::translate_operand(ctx, body, arg, value_map, block_ptr, last_op, loc.clone())?;
        last_op = next_op;
        operands.push(value);
    }

    let f32_ty = FP32Type::get(ctx);
    let result_types = (0..4).map(|_| f32_ty.into()).collect();
    let op = Operation::new(
        ctx,
        MmaSyncM16N8K16F32F16Op::get_concrete_op_info(),
        result_types,
        operands,
        vec![],
        0,
    );
    op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        op.insert_after(ctx, prev);
    } else {
        op.insert_at_front(block_ptr, ctx);
    }

    let results = (0..4).map(|i| op.deref(ctx).get_result(i)).collect();
    emit_cusimd_result(
        ctx,
        body,
        destination,
        target,
        block_ptr,
        op,
        results,
        f32_ty.into(),
        4,
        value_map,
        block_map,
        loc,
        "mma_m16n8k16_f32_f16_raw call without target block",
    )
}
