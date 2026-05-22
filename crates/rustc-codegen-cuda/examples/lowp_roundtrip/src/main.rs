/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Low-precision host/device storage and conversion round trip.
//!
//! Run with:
//!   cargo oxide run lowp_roundtrip

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use cuda_lowp::{Fp4E2M1, Fp4x2E2M1, Fp8E4M3, Fp8E5M2};

const VALUES_U32: u32 = 12;
const VALUES: usize = VALUES_U32 as usize;
const PACKED_FP4: usize = VALUES / 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn convert_lowp(
        e4: &[Fp8E4M3],
        e5: &[Fp8E5M2],
        e2_pairs: &[Fp4x2E2M1],
        source: &[f32],
        mut e4_wide: DisjointSlice<f32>,
        mut e5_wide: DisjointSlice<f32>,
        mut e2_wide: DisjointSlice<f32>,
        mut narrowed_bits: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d().get() as usize;

        if idx < VALUES {
            let e4_value = e4[idx];
            let e5_value = e5[idx];
            let e2_value = Fp4E2M1::from_f32_sat(source[idx]);

            unsafe {
                *e4_wide.get_unchecked_mut(idx) = e4_value.to_f32();
                *e5_wide.get_unchecked_mut(idx) = e5_value.to_f32();
                *narrowed_bits.get_unchecked_mut(idx) =
                    u32::from(Fp8E4M3::from_f32_sat(source[idx]).to_bits())
                        | (u32::from(Fp8E5M2::from_f32_sat(source[idx]).to_bits()) << 8)
                        | (u32::from(e2_value.to_bits()) << 16);
            }
        }

        if idx < PACKED_FP4 {
            let pair = e2_pairs[idx];
            let out = idx * 2;
            unsafe {
                *e2_wide.get_unchecked_mut(out) = pair.lo().to_f32();
                *e2_wide.get_unchecked_mut(out + 1) = pair.hi().to_f32();
                *narrowed_bits.get_unchecked_mut(VALUES + idx) = u32::from(pair.to_bits());
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    let source = source_values();
    let e4 = source
        .iter()
        .copied()
        .map(Fp8E4M3::from_f32_sat)
        .collect::<Vec<_>>();
    let e5 = source
        .iter()
        .copied()
        .map(Fp8E5M2::from_f32_sat)
        .collect::<Vec<_>>();
    let e2_pairs = pack_fp4(&source);

    let e4_dev = DeviceBuffer::<Fp8E4M3>::from_host(&stream, &e4)?;
    let e5_dev = DeviceBuffer::<Fp8E5M2>::from_host(&stream, &e5)?;
    let e2_dev = DeviceBuffer::<Fp4x2E2M1>::from_host(&stream, &e2_pairs)?;
    let source_dev = DeviceBuffer::<f32>::from_host(&stream, &source)?;

    check_bits(
        "typed e4 DeviceBuffer round trip",
        &e4_dev.to_host_vec(&stream)?,
        &e4,
    );
    check_bits(
        "typed e5 DeviceBuffer round trip",
        &e5_dev.to_host_vec(&stream)?,
        &e5,
    );
    check_bits(
        "typed e2 pair DeviceBuffer round trip",
        &e2_dev.to_host_vec(&stream)?,
        &e2_pairs,
    );

    let mut e4_wide = DeviceBuffer::<f32>::zeroed(&stream, VALUES)?;
    let mut e5_wide = DeviceBuffer::<f32>::zeroed(&stream, VALUES)?;
    let mut e2_wide = DeviceBuffer::<f32>::zeroed(&stream, VALUES)?;
    let mut narrowed_bits = DeviceBuffer::<u32>::zeroed(&stream, VALUES + PACKED_FP4)?;

    module.convert_lowp(
        &stream,
        LaunchConfig::for_num_elems(VALUES_U32),
        &e4_dev,
        &e5_dev,
        &e2_dev,
        &source_dev,
        &mut e4_wide,
        &mut e5_wide,
        &mut e2_wide,
        &mut narrowed_bits,
    )?;
    stream.synchronize()?;

    check_f32(
        "e4 widen",
        &e4_wide.to_host_vec(&stream)?,
        &e4_wide_reference(&e4),
    );
    check_f32(
        "e5 widen",
        &e5_wide.to_host_vec(&stream)?,
        &e5_wide_reference(&e5),
    );
    check_f32(
        "e2 widen",
        &e2_wide.to_host_vec(&stream)?,
        &fp4_wide_reference(&e2_pairs),
    );
    check_u32(
        "device narrowing bits",
        &narrowed_bits.to_host_vec(&stream)?,
        &narrowed_reference(&source, &e2_pairs),
    );

    println!("SUCCESS: low-precision typed buffers and device conversions matched host references");
    Ok(())
}

fn source_values() -> Vec<f32> {
    vec![
        -448.0,
        -7.0,
        -1.25,
        -0.0,
        0.0,
        0.001953125,
        0.5,
        1.25,
        6.0,
        449.0,
        f32::INFINITY,
        f32::NAN,
    ]
}

fn pack_fp4(source: &[f32]) -> Vec<Fp4x2E2M1> {
    let mut out = Vec::with_capacity(PACKED_FP4);
    for pair in source.chunks_exact(2) {
        out.push(Fp4x2E2M1::new(
            Fp4E2M1::from_f32_sat(pair[0]),
            Fp4E2M1::from_f32_sat(pair[1]),
        ));
    }
    out
}

fn e4_wide_reference(values: &[Fp8E4M3]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn e5_wide_reference(values: &[Fp8E5M2]) -> Vec<f32> {
    values.iter().map(|value| value.to_f32()).collect()
}

fn fp4_wide_reference(values: &[Fp4x2E2M1]) -> Vec<f32> {
    let mut out = Vec::with_capacity(VALUES);
    for value in values {
        out.push(value.lo().to_f32());
        out.push(value.hi().to_f32());
    }
    out
}

fn narrowed_reference(source: &[f32], pairs: &[Fp4x2E2M1]) -> Vec<u32> {
    let mut out = Vec::with_capacity(VALUES + PACKED_FP4);
    for value in source {
        out.push(
            u32::from(Fp8E4M3::from_f32_sat(*value).to_bits())
                | (u32::from(Fp8E5M2::from_f32_sat(*value).to_bits()) << 8)
                | (u32::from(Fp4E2M1::from_f32_sat(*value).to_bits()) << 16),
        );
    }
    for pair in pairs {
        out.push(u32::from(pair.to_bits()));
    }
    out
}

trait Bits {
    fn bits(self) -> u32;
}

impl Bits for Fp8E4M3 {
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
}

impl Bits for Fp8E5M2 {
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
}

impl Bits for Fp4x2E2M1 {
    fn bits(self) -> u32 {
        u32::from(self.to_bits())
    }
}

fn check_bits<T>(name: &str, got: &[T], expected: &[T])
where
    T: Copy + Bits,
{
    assert_eq!(got.len(), expected.len(), "{name} length mismatch");
    for (index, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        if actual.bits() != want.bits() {
            eprintln!(
                "FAIL {name} at {index}: got 0x{:x}, expected 0x{:x}",
                actual.bits(),
                want.bits()
            );
            std::process::exit(1);
        }
    }
}

fn check_f32(name: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{name} length mismatch");
    for (index, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        let matches = if actual.is_nan() || want.is_nan() {
            actual.is_nan() && want.is_nan()
        } else {
            actual.to_bits() == want.to_bits()
        };
        if !matches {
            eprintln!("FAIL {name} at {index}: got {actual:?}, expected {want:?}");
            std::process::exit(1);
        }
    }
}

fn check_u32(name: &str, got: &[u32], expected: &[u32]) {
    assert_eq!(got.len(), expected.len(), "{name} length mismatch");
    for (index, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        if actual != want {
            eprintln!("FAIL {name} at {index}: got 0x{actual:x}, expected 0x{want:x}");
            std::process::exit(1);
        }
    }
}
