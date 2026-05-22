/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cooperative selection primitives.
//!
//! The initial surface is deterministic top-k selection for `f32` scores.
//! Callers provide shared-memory scratch explicitly so kernels control the
//! temporary-memory footprint.

use crate::SharedArray;
use crate::cooperative_groups::{ThreadBlock, ThreadGroup};
use crate::thread;

/// One selected score and its row-local index.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopKEntry {
    /// Selected score.
    pub score: f32,
    /// Row-local index for `score`, or `u32::MAX` for an empty slot.
    pub index: u32,
}

impl TopKEntry {
    /// Empty entry used to initialize fixed-capacity top-k buffers.
    pub const EMPTY: Self = Self {
        score: f32::NEG_INFINITY,
        index: u32::MAX,
    };

    /// Construct a valid selected entry.
    #[inline(always)]
    pub const fn new(score: f32, index: u32) -> Self {
        Self { score, index }
    }

    /// Returns true when this entry holds a real candidate.
    #[inline(always)]
    pub const fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }

    /// Deterministic descending order: higher score wins, lower index breaks
    /// ties, and NaN ranks behind every non-NaN value.
    #[inline(always)]
    pub fn is_better_than(&self, other: &Self) -> bool {
        if !self.is_valid() {
            return false;
        }
        if !other.is_valid() {
            return true;
        }

        let self_nan = self.score != self.score;
        let other_nan = other.score != other.score;
        if self_nan || other_nan {
            return !self_nan && other_nan;
        }

        self.score > other.score || (self.score == other.score && self.index < other.index)
    }
}

/// Fixed-capacity top-k buffer sorted by [`TopKEntry::is_better_than`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TopK<const K: usize> {
    entries: [TopKEntry; K],
}

impl<const K: usize> TopK<K> {
    /// Empty top-k buffer.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            entries: [TopKEntry::EMPTY; K],
        }
    }

    /// Return the sorted selected entries.
    #[inline(always)]
    pub const fn entries(&self) -> &[TopKEntry; K] {
        &self.entries
    }

    /// Return one selected entry by rank.
    #[inline(always)]
    pub const fn get(&self, rank: usize) -> TopKEntry {
        self.entries[rank]
    }

    /// Insert a candidate and preserve descending top-k order.
    #[inline(always)]
    pub fn insert(&mut self, score: f32, index: u32) {
        self.insert_entry(TopKEntry::new(score, index));
    }

    /// Merge another fixed-capacity top-k buffer into this one.
    #[inline(always)]
    pub fn merge(&mut self, other: Self) {
        let mut i = 0usize;
        while i < K {
            self.insert_entry(other.entries[i]);
            i += 1;
        }
    }

    #[inline(always)]
    fn insert_entry(&mut self, candidate: TopKEntry) {
        const {
            assert!(K > 0, "TopK requires K > 0");
        }

        let mut rank = 0usize;
        while rank < K {
            if candidate.is_better_than(&self.entries[rank]) {
                let mut shift = K - 1;
                while shift > rank {
                    self.entries[shift] = self.entries[shift - 1];
                    shift -= 1;
                }
                self.entries[rank] = candidate;
                return;
            }
            rank += 1;
        }
    }
}

impl<const K: usize> Default for TopK<K> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// Compute sorted top-k scores for one row using a whole 1D thread block.
///
/// Each thread scans a block-strided slice of the row into a local [`TopK`],
/// then all per-thread buffers are merged through `scratch`. The result is
/// returned to every thread in the block.
///
/// # Contract
///
/// - The block must be one-dimensional and have exactly `BLOCK_THREADS`
///   threads.
/// - `scratch` must point to block-scoped shared memory with
///   `BLOCK_THREADS` entries.
/// - `row_start + row_len` must be in bounds for `scores`.
/// - Every thread in the block must call this function together.
#[inline(always)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn block_topk_f32<const K: usize, const BLOCK_THREADS: usize>(
    block: &ThreadBlock,
    scores: &[f32],
    row_start: usize,
    row_len: usize,
    scratch: *mut SharedArray<TopK<K>, BLOCK_THREADS>,
) -> TopK<K> {
    const {
        let valid_block = BLOCK_THREADS > 0
            && BLOCK_THREADS <= 1024
            && (BLOCK_THREADS & (BLOCK_THREADS - 1)) == 0;
        assert!(K > 0, "block_topk_f32 requires K > 0");
        assert!(
            valid_block,
            "block_topk_f32 requires a power-of-two block size in 1..=1024",
        );
    }

    let tid = thread_in_block_linear() as usize;
    let mut local = TopK::<K>::new();

    let mut offset = tid;
    while offset < row_len {
        let score = scores[row_start + offset];
        local.insert(score, offset as u32);
        offset += BLOCK_THREADS;
    }

    let scratch: &mut SharedArray<TopK<K>, BLOCK_THREADS> = unsafe { &mut *scratch };
    scratch[tid] = local;
    block.sync();

    let mut stride = BLOCK_THREADS >> 1;
    while stride > 0 {
        if tid < stride {
            let other = scratch[tid + stride];
            let mut current = scratch[tid];
            current.merge(other);
            scratch[tid] = current;
        }
        block.sync();
        stride >>= 1;
    }

    scratch[0]
}

#[inline(always)]
fn thread_in_block_linear() -> u32 {
    let tx = thread::threadIdx_x();
    let ty = thread::threadIdx_y();
    let tz = thread::threadIdx_z();
    let dx = thread::blockDim_x();
    let dy = thread::blockDim_y();
    (tz * dy + ty) * dx + tx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topk_orders_scores_and_ties_by_index() {
        let mut top = TopK::<3>::new();
        top.insert(3.0, 7);
        top.insert(5.0, 9);
        top.insert(5.0, 2);
        top.insert(4.0, 4);

        assert_eq!(top.get(0), TopKEntry::new(5.0, 2));
        assert_eq!(top.get(1), TopKEntry::new(5.0, 9));
        assert_eq!(top.get(2), TopKEntry::new(4.0, 4));
    }

    #[test]
    fn topk_ranks_nan_after_numbers() {
        let mut top = TopK::<2>::new();
        top.insert(f32::NAN, 0);
        top.insert(-10.0, 1);
        top.insert(f32::NAN, 2);

        assert_eq!(top.get(0), TopKEntry::new(-10.0, 1));
        assert_eq!(top.get(1).index, 0);
        assert!(top.get(1).score.is_nan());
    }
}
