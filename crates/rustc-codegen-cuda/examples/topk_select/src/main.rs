/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-cooperative top-k selection over multiple rows.
//!
//! Run with:
//!   cargo oxide run topk_select

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::cooperative_groups::this_thread_block;
use cuda_device::{DisjointSlice, SharedArray, TopK, block_topk_f32, kernel, thread};
use cuda_host::cuda_module;

const ROWS: usize = 4;
const ROW_LEN: usize = 257;
const TOP_K: usize = 4;
const BLOCK_THREADS: usize = 128;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn topk_rows(
        scores: &[f32],
        mut out_scores: DisjointSlice<f32>,
        mut out_indices: DisjointSlice<u32>,
    ) {
        static mut SCRATCH: SharedArray<TopK<TOP_K>, BLOCK_THREADS> = SharedArray::UNINIT;

        let row = thread::blockIdx_x() as usize;
        let block = this_thread_block();
        let top = block_topk_f32::<TOP_K, BLOCK_THREADS>(
            &block,
            scores,
            row * ROW_LEN,
            ROW_LEN,
            &raw mut SCRATCH,
        );

        let rank = thread::threadIdx_x() as usize;
        if rank < TOP_K {
            let entry = top.get(rank);
            let out = row * TOP_K + rank;
            unsafe {
                *out_scores.get_unchecked_mut(out) = entry.score;
                *out_indices.get_unchecked_mut(out) = entry.index;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let scores = build_scores();
    let expected = reference_topk(&scores);

    let scores_dev = DeviceBuffer::<f32>::from_host(&stream, &scores)?;
    let mut out_scores = DeviceBuffer::<f32>::zeroed(&stream, ROWS * TOP_K)?;
    let mut out_indices = DeviceBuffer::<u32>::zeroed(&stream, ROWS * TOP_K)?;

    let config = LaunchConfig {
        grid_dim: (ROWS as u32, 1, 1),
        block_dim: (BLOCK_THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    module.topk_rows(
        &stream,
        config,
        &scores_dev,
        &mut out_scores,
        &mut out_indices,
    )?;
    stream.synchronize()?;

    let got_scores = out_scores.to_host_vec(&stream)?;
    let got_indices = out_indices.to_host_vec(&stream)?;

    for row in 0..ROWS {
        for rank in 0..TOP_K {
            let out = row * TOP_K + rank;
            let expected_entry = expected[out];
            let got_score = got_scores[out];
            let got_index = got_indices[out];
            if got_score != expected_entry.score || got_index != expected_entry.index {
                eprintln!(
                    "FAIL: row {row} rank {rank}: got ({got_score}, {got_index}), expected ({}, {})",
                    expected_entry.score, expected_entry.index
                );
                std::process::exit(1);
            }
        }
    }

    println!(
        "SUCCESS: top-k selection matched CPU reference for {ROWS} rows x {ROW_LEN} scores (K={TOP_K})"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct HostEntry {
    score: f32,
    index: u32,
}

fn build_scores() -> Vec<f32> {
    let mut scores = vec![0.0f32; ROWS * ROW_LEN];
    for row in 0..ROWS {
        for col in 0..ROW_LEN {
            let mut score = ((row * 17 + col * 13) % 97) as f32 - 48.0;
            if col % 53 == 0 {
                score = 1000.0 - row as f32;
            }
            if col == 5 + row {
                score = f32::NAN;
            }
            scores[row * ROW_LEN + col] = score;
        }
    }
    scores
}

fn reference_topk(scores: &[f32]) -> Vec<HostEntry> {
    let mut out = Vec::with_capacity(ROWS * TOP_K);
    for row in 0..ROWS {
        let mut entries = Vec::with_capacity(ROW_LEN);
        for col in 0..ROW_LEN {
            entries.push(HostEntry {
                score: scores[row * ROW_LEN + col],
                index: col as u32,
            });
        }
        entries.sort_by(|a, b| compare_entries(a, b));
        out.extend_from_slice(&entries[..TOP_K]);
    }
    out
}

fn compare_entries(a: &HostEntry, b: &HostEntry) -> std::cmp::Ordering {
    match (a.score.is_nan(), b.score.is_nan()) {
        (true, true) => a.index.cmp(&b.index),
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => b
            .score
            .total_cmp(&a.score)
            .then_with(|| a.index.cmp(&b.index)),
    }
}
