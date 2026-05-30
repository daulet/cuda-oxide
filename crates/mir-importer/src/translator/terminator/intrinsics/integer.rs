/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Packed integer arithmetic primitives.

use super::super::helpers::emit_store_result_and_goto;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_nvvm::ops::Dp4aSignedSignedOp;
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use rustc_public::mir;

/// Emits `integer::dp4a_i8(a, b, acc)`.
pub fn emit_dp4a_i8(
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
    if args.len() != 3 {
        return input_err!(
            loc.clone(),
            TranslationErr::unsupported(format!(
                "integer::dp4a_i8 expects 3 arguments [a, b, acc], got {}",
                args.len()
            ))
        );
    }

    let i32_type = IntegerType::get(ctx, 32, Signedness::Signed);
    let (a, mut last_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[0],
        value_map,
        block_ptr,
        prev_op,
        loc.clone(),
    )?;
    let (b, next_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[1],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next_op;
    let (acc, next_op) = rvalue::translate_operand(
        ctx,
        body,
        &args[2],
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = next_op;

    let dp4a_op = Operation::new(
        ctx,
        Dp4aSignedSignedOp::get_concrete_op_info(),
        vec![i32_type.to_ptr()],
        vec![a, b, acc],
        vec![],
        0,
    );
    dp4a_op.deref_mut(ctx).set_loc(loc.clone());
    if let Some(prev) = last_op {
        dp4a_op.insert_after(ctx, prev);
    } else {
        dp4a_op.insert_at_front(block_ptr, ctx);
    }
    let result = dp4a_op.deref(ctx).get_result(0);
    emit_store_result_and_goto(
        ctx,
        destination,
        result,
        target,
        block_ptr,
        dp4a_op,
        value_map,
        block_map,
        loc,
        "integer::dp4a_i8 call without target block",
    )
}
