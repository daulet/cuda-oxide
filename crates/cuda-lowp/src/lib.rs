/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![no_std]

//! Low-precision inference storage types shared by cuda-oxide host and device code.
//!
//! The fp8 and fp4 formats match CUDA 13.2's `__NV_E4M3`, `__NV_E5M2`, and
//! `__NV_E2M1` storage interpretations. Narrowing conversions use
//! round-to-nearest-even and saturate finite overflows to the largest finite
//! value of the target format.

use core::cmp::Ordering;

/// CUDA `__NV_E4M3` fp8 storage.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8E4M3(u8);

impl Fp8E4M3 {
    /// Construct from raw storage bits.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Return raw storage bits.
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from `f32` using round-to-nearest-even and finite saturation.
    #[inline]
    pub fn from_f32_sat(value: f32) -> Self {
        Self(encode_sat(value, 0x80, 0x7f, 3, 7, 0x7e, 0x7f, false, 0x7f))
    }

    /// Widen to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, 0x80, 0x7f, 3, 7, 0x7f, false)
    }

    /// Returns true for the CUDA E4M3 NaN encodings `0x7f` and `0xff`.
    #[inline]
    pub const fn is_nan(self) -> bool {
        (self.0 & 0x7f) == 0x7f
    }

    /// Deterministic numeric ordering with NaN values ordered after numbers.
    #[inline]
    pub fn total_cmp(self, other: Self) -> Ordering {
        total_cmp(self.to_f32(), other.to_f32(), self.0, other.0)
    }
}

/// CUDA `__NV_E5M2` fp8 storage.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8E5M2(u8);

impl Fp8E5M2 {
    /// Construct from raw storage bits.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Return raw storage bits.
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    /// Convert from `f32` using round-to-nearest-even and finite saturation.
    #[inline]
    pub fn from_f32_sat(value: f32) -> Self {
        Self(encode_sat(value, 0x80, 0x7f, 2, 15, 0x7b, 0xff, true, 0x7f))
    }

    /// Widen to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.0, 0x80, 0x7f, 2, 15, 0xff, true)
    }

    /// Returns true for E5M2 NaN encodings.
    #[inline]
    pub const fn is_nan(self) -> bool {
        ((self.0 & 0x7c) == 0x7c) && (self.0 & 0x03) != 0
    }

    /// Deterministic numeric ordering with NaN values ordered after numbers.
    #[inline]
    pub fn total_cmp(self, other: Self) -> Ordering {
        total_cmp(self.to_f32(), other.to_f32(), self.0, other.0)
    }
}

/// CUDA `__NV_E2M1` fp4 storage in the low nibble.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Fp4E2M1(u8);

impl Fp4E2M1 {
    /// Construct from raw storage bits. Only the low nibble is retained.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits & 0x0f)
    }

    /// Return the low-nibble storage bits.
    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0 & 0x0f
    }

    /// Convert from `f32` using round-to-nearest-even and finite saturation.
    ///
    /// CUDA's E2M1 conversion maps NaN inputs to positive maxnorm.
    #[inline]
    pub fn from_f32_sat(value: f32) -> Self {
        Self(encode_sat(value, 0x08, 0x07, 1, 1, 0x07, 0xff, false, 0x07))
    }

    /// Widen to `f32`.
    #[inline]
    pub fn to_f32(self) -> f32 {
        decode(self.to_bits(), 0x08, 0x07, 1, 1, 0xff, false)
    }

    /// E2M1 has no NaN storage encodings.
    #[inline]
    pub const fn is_nan(self) -> bool {
        false
    }

    /// Deterministic numeric ordering.
    #[inline]
    pub fn total_cmp(self, other: Self) -> Ordering {
        total_cmp(
            self.to_f32(),
            other.to_f32(),
            self.to_bits(),
            other.to_bits(),
        )
    }
}

impl Eq for Fp4E2M1 {}

impl PartialEq for Fp4E2M1 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }
}

/// Two E4M3 values packed as CUDA does: low element in bits 0..8.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8x2E4M3(u16);

impl Fp8x2E4M3 {
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn new(lo: Fp8E4M3, hi: Fp8E4M3) -> Self {
        Self((lo.to_bits() as u16) | ((hi.to_bits() as u16) << 8))
    }

    #[inline]
    pub const fn lo(self) -> Fp8E4M3 {
        Fp8E4M3::from_bits(self.0 as u8)
    }

    #[inline]
    pub const fn hi(self) -> Fp8E4M3 {
        Fp8E4M3::from_bits((self.0 >> 8) as u8)
    }

    #[inline]
    pub const fn get(self, index: usize) -> Fp8E4M3 {
        assert!(index < 2, "Fp8x2E4M3 index out of range");
        Fp8E4M3::from_bits((self.0 >> (index * 8)) as u8)
    }
}

/// Four E4M3 values packed as two CUDA fp8x2 lanes.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8x4E4M3(u32);

impl Fp8x4E4M3 {
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn new(x0: Fp8E4M3, x1: Fp8E4M3, x2: Fp8E4M3, x3: Fp8E4M3) -> Self {
        Self(
            (x0.to_bits() as u32)
                | ((x1.to_bits() as u32) << 8)
                | ((x2.to_bits() as u32) << 16)
                | ((x3.to_bits() as u32) << 24),
        )
    }

    #[inline]
    pub const fn get(self, index: usize) -> Fp8E4M3 {
        assert!(index < 4, "Fp8x4E4M3 index out of range");
        Fp8E4M3::from_bits((self.0 >> (index * 8)) as u8)
    }
}

/// Two E5M2 values packed as CUDA does: low element in bits 0..8.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8x2E5M2(u16);

impl Fp8x2E5M2 {
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn new(lo: Fp8E5M2, hi: Fp8E5M2) -> Self {
        Self((lo.to_bits() as u16) | ((hi.to_bits() as u16) << 8))
    }

    #[inline]
    pub const fn lo(self) -> Fp8E5M2 {
        Fp8E5M2::from_bits(self.0 as u8)
    }

    #[inline]
    pub const fn hi(self) -> Fp8E5M2 {
        Fp8E5M2::from_bits((self.0 >> 8) as u8)
    }

    #[inline]
    pub const fn get(self, index: usize) -> Fp8E5M2 {
        assert!(index < 2, "Fp8x2E5M2 index out of range");
        Fp8E5M2::from_bits((self.0 >> (index * 8)) as u8)
    }
}

/// Four E5M2 values packed as two CUDA fp8x2 lanes.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp8x4E5M2(u32);

impl Fp8x4E5M2 {
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.0
    }

    #[inline]
    pub const fn new(x0: Fp8E5M2, x1: Fp8E5M2, x2: Fp8E5M2, x3: Fp8E5M2) -> Self {
        Self(
            (x0.to_bits() as u32)
                | ((x1.to_bits() as u32) << 8)
                | ((x2.to_bits() as u32) << 16)
                | ((x3.to_bits() as u32) << 24),
        )
    }

    #[inline]
    pub const fn get(self, index: usize) -> Fp8E5M2 {
        assert!(index < 4, "Fp8x4E5M2 index out of range");
        Fp8E5M2::from_bits((self.0 >> (index * 8)) as u8)
    }
}

/// Two E2M1 values packed as CUDA does: low element in bits 0..4.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp4x2E2M1(u8);

impl Fp4x2E2M1 {
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u8 {
        self.0
    }

    #[inline]
    pub const fn new(lo: Fp4E2M1, hi: Fp4E2M1) -> Self {
        Self(lo.to_bits() | (hi.to_bits() << 4))
    }

    #[inline]
    pub const fn lo(self) -> Fp4E2M1 {
        Fp4E2M1::from_bits(self.0)
    }

    #[inline]
    pub const fn hi(self) -> Fp4E2M1 {
        Fp4E2M1::from_bits(self.0 >> 4)
    }
}

/// Four E2M1 values packed as two CUDA fp4x2 lanes.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fp4x4E2M1(u16);

impl Fp4x4E2M1 {
    #[inline]
    pub const fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    #[inline]
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    #[inline]
    pub const fn new(x0: Fp4E2M1, x1: Fp4E2M1, x2: Fp4E2M1, x3: Fp4E2M1) -> Self {
        Self(
            (x0.to_bits() as u16)
                | ((x1.to_bits() as u16) << 4)
                | ((x2.to_bits() as u16) << 8)
                | ((x3.to_bits() as u16) << 12),
        )
    }

    #[inline]
    pub const fn get(self, index: usize) -> Fp4E2M1 {
        assert!(index < 4, "Fp4x4E2M1 index out of range");
        Fp4E2M1::from_bits((self.0 >> (index * 4)) as u8)
    }
}

#[inline]
fn decode(
    bits: u8,
    sign_bit: u8,
    abs_mask: u8,
    mantissa_bits: u8,
    exponent_bias: i32,
    nan_abs_bits: u8,
    has_e5_specials: bool,
) -> f32 {
    let sign = bits & sign_bit;
    let abs_bits = bits & abs_mask;

    if nan_abs_bits != 0xff && abs_bits == nan_abs_bits {
        return canonical_nan(bits);
    }

    let exp_mask = abs_mask >> mantissa_bits;
    let mant_mask = (1u8 << mantissa_bits) - 1;
    let exponent = (abs_bits >> mantissa_bits) & exp_mask;
    let mantissa = abs_bits & mant_mask;

    if has_e5_specials && exponent == exp_mask {
        if mantissa == 0 {
            return if sign == 0 {
                f32::INFINITY
            } else {
                f32::NEG_INFINITY
            };
        }
        return canonical_nan(bits);
    }

    let value = if exponent == 0 {
        if mantissa == 0 {
            0.0
        } else {
            (mantissa as f32) * pow2(1 - exponent_bias - i32::from(mantissa_bits))
        }
    } else {
        ((1u32 << mantissa_bits) as f32 + mantissa as f32)
            * pow2(i32::from(exponent) - exponent_bias - i32::from(mantissa_bits))
    };

    if sign == 0 { value } else { -value }
}

#[inline]
fn encode_sat(
    value: f32,
    sign_bit: u8,
    abs_mask: u8,
    mantissa_bits: u8,
    exponent_bias: i32,
    max_finite_abs_bits: u8,
    nan_abs_bits: u8,
    has_e5_specials: bool,
    nan_input_bits: u8,
) -> u8 {
    let value_bits = value.to_bits();
    let abs_bits = value_bits & 0x7fff_ffff;

    if abs_bits > 0x7f80_0000 {
        return nan_input_bits;
    }

    let sign = if (value_bits & 0x8000_0000) == 0 {
        0
    } else {
        sign_bit
    };
    let abs = f32::from_bits(abs_bits);

    if abs == 0.0 {
        return sign;
    }

    if abs_bits == 0x7f80_0000 {
        return sign | max_finite_abs_bits;
    }

    let mut best_bits = 0u8;
    let mut best_diff = f32::INFINITY;
    let mut candidate = 0u8;
    while candidate <= max_finite_abs_bits {
        let candidate_value = decode(
            candidate,
            sign_bit,
            abs_mask,
            mantissa_bits,
            exponent_bias,
            nan_abs_bits,
            has_e5_specials,
        );
        let diff = if abs >= candidate_value {
            abs - candidate_value
        } else {
            candidate_value - abs
        };

        if diff < best_diff || (diff == best_diff && (candidate & 1) == 0 && (best_bits & 1) != 0) {
            best_bits = candidate;
            best_diff = diff;
        }

        if candidate == max_finite_abs_bits {
            break;
        }
        candidate += 1;
    }

    sign | best_bits
}

#[inline]
fn pow2(exponent: i32) -> f32 {
    debug_assert!((-126..=127).contains(&exponent));
    f32::from_bits(((exponent + 127) as u32) << 23)
}

#[inline]
fn canonical_nan(bits: u8) -> f32 {
    f32::from_bits(0x7fc0_0000 | u32::from(bits & 1))
}

#[inline]
fn total_cmp(a: f32, b: f32, a_bits: u8, b_bits: u8) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => a_bits.cmp(&b_bits),
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => {
            if a < b {
                Ordering::Less
            } else if a > b {
                Ordering::Greater
            } else {
                Ordering::Equal
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp8_e4m3_decodes_cuda_samples() {
        assert_eq!(Fp8E4M3::from_bits(0x00).to_f32(), 0.0);
        assert_eq!(
            Fp8E4M3::from_bits(0x80).to_f32().to_bits(),
            (-0.0f32).to_bits()
        );
        assert_eq!(Fp8E4M3::from_bits(0x01).to_f32(), 0.001953125);
        assert_eq!(Fp8E4M3::from_bits(0x38).to_f32(), 1.0);
        assert_eq!(Fp8E4M3::from_bits(0x7e).to_f32(), 448.0);
        assert_eq!(Fp8E4M3::from_bits(0xfe).to_f32(), -448.0);
        assert!(Fp8E4M3::from_bits(0x7f).to_f32().is_nan());
        assert!(Fp8E4M3::from_bits(0xff).is_nan());
    }

    #[test]
    fn fp8_e5m2_decodes_cuda_samples() {
        assert_eq!(Fp8E5M2::from_bits(0x01).to_f32(), 0.0000152587890625);
        assert_eq!(Fp8E5M2::from_bits(0x3c).to_f32(), 1.0);
        assert_eq!(Fp8E5M2::from_bits(0x7b).to_f32(), 57344.0);
        assert_eq!(Fp8E5M2::from_bits(0xfb).to_f32(), -57344.0);
        assert_eq!(Fp8E5M2::from_bits(0x7c).to_f32(), f32::INFINITY);
        assert_eq!(Fp8E5M2::from_bits(0xfc).to_f32(), f32::NEG_INFINITY);
        assert!(Fp8E5M2::from_bits(0x7d).is_nan());
        assert!(Fp8E5M2::from_bits(0xff).to_f32().is_nan());
    }

    #[test]
    fn fp4_e2m1_decodes_cuda_samples() {
        let expected: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        let mut bits = 0u8;
        while bits < 16 {
            assert_eq!(
                Fp4E2M1::from_bits(bits).to_f32().to_bits(),
                expected[bits as usize].to_bits()
            );
            bits += 1;
        }
    }

    #[test]
    fn narrowing_matches_cuda_probe_samples() {
        assert_eq!(Fp8E4M3::from_f32_sat(448.0).to_bits(), 0x7e);
        assert_eq!(Fp8E4M3::from_f32_sat(449.0).to_bits(), 0x7e);
        assert_eq!(Fp8E4M3::from_f32_sat(f32::INFINITY).to_bits(), 0x7e);
        assert_eq!(Fp8E4M3::from_f32_sat(f32::NAN).to_bits(), 0x7f);

        assert_eq!(Fp8E5M2::from_f32_sat(448.0).to_bits(), 0x5f);
        assert_eq!(Fp8E5M2::from_f32_sat(1000.0).to_bits(), 0x64);
        assert_eq!(Fp8E5M2::from_f32_sat(f32::INFINITY).to_bits(), 0x7b);
        assert_eq!(Fp8E5M2::from_f32_sat(f32::NAN).to_bits(), 0x7f);

        assert_eq!(Fp4E2M1::from_f32_sat(6.0).to_bits(), 0x7);
        assert_eq!(Fp4E2M1::from_f32_sat(7.0).to_bits(), 0x7);
        assert_eq!(Fp4E2M1::from_f32_sat(-3.0).to_bits(), 0xd);
        assert_eq!(Fp4E2M1::from_f32_sat(f32::NAN).to_bits(), 0x7);
    }

    #[test]
    fn narrowing_uses_round_to_nearest_even() {
        assert_eq!(Fp8E4M3::from_f32_sat(1.0625).to_bits(), 0x38);
        assert_eq!(Fp8E4M3::from_f32_sat(1.1875).to_bits(), 0x3a);
        assert_eq!(Fp8E5M2::from_f32_sat(1.125).to_bits(), 0x3c);
        assert_eq!(Fp8E5M2::from_f32_sat(1.375).to_bits(), 0x3e);
        assert_eq!(Fp4E2M1::from_f32_sat(1.25).to_bits(), 0x2);
        assert_eq!(Fp4E2M1::from_f32_sat(1.75).to_bits(), 0x4);
    }

    #[test]
    fn total_cmp_orders_nan_after_numbers() {
        assert_eq!(
            Fp8E4M3::from_bits(0x7e).total_cmp(Fp8E4M3::from_bits(0x7f)),
            Ordering::Less
        );
        assert_eq!(
            Fp8E5M2::from_bits(0xfc).total_cmp(Fp8E5M2::from_bits(0xfb)),
            Ordering::Less
        );
        assert_eq!(
            Fp4E2M1::from_bits(0x8).total_cmp(Fp4E2M1::from_bits(0x0)),
            Ordering::Equal
        );
    }

    #[test]
    fn pack_helpers_use_cuda_lane_order() {
        let e4 = Fp8x2E4M3::new(Fp8E4M3::from_bits(0x12), Fp8E4M3::from_bits(0x34));
        assert_eq!(e4.to_bits(), 0x3412);
        assert_eq!(e4.lo().to_bits(), 0x12);
        assert_eq!(e4.hi().to_bits(), 0x34);
        assert_eq!(e4.get(1).to_bits(), 0x34);

        let e4x4 = Fp8x4E4M3::new(
            Fp8E4M3::from_bits(0x12),
            Fp8E4M3::from_bits(0x34),
            Fp8E4M3::from_bits(0x56),
            Fp8E4M3::from_bits(0x78),
        );
        assert_eq!(e4x4.to_bits(), 0x78563412);
        assert_eq!(e4x4.get(2).to_bits(), 0x56);

        let e5 = Fp8x2E5M2::new(Fp8E5M2::from_bits(0x56), Fp8E5M2::from_bits(0x78));
        assert_eq!(e5.to_bits(), 0x7856);
        assert_eq!(e5.lo().to_bits(), 0x56);
        assert_eq!(e5.hi().to_bits(), 0x78);
        assert_eq!(e5.get(0).to_bits(), 0x56);

        let e5x4 = Fp8x4E5M2::new(
            Fp8E5M2::from_bits(0x9a),
            Fp8E5M2::from_bits(0xbc),
            Fp8E5M2::from_bits(0xde),
            Fp8E5M2::from_bits(0xf0),
        );
        assert_eq!(e5x4.to_bits(), 0xf0debc9a);
        assert_eq!(e5x4.get(3).to_bits(), 0xf0);

        let e2 = Fp4x2E2M1::new(Fp4E2M1::from_bits(0x3), Fp4E2M1::from_bits(0xc));
        assert_eq!(e2.to_bits(), 0xc3);
        assert_eq!(e2.lo().to_bits(), 0x3);
        assert_eq!(e2.hi().to_bits(), 0xc);

        let e2x4 = Fp4x4E2M1::new(
            Fp4E2M1::from_bits(0x1),
            Fp4E2M1::from_bits(0x2),
            Fp4E2M1::from_bits(0x3),
            Fp4E2M1::from_bits(0x4),
        );
        assert_eq!(e2x4.to_bits(), 0x4321);
        assert_eq!(e2x4.get(2).to_bits(), 0x3);
    }

    #[test]
    fn bit_constructors_are_exhaustive() {
        let mut bits = 0u16;
        while bits <= 255 {
            let byte = bits as u8;
            assert_eq!(Fp8E4M3::from_bits(byte).to_bits(), byte);
            assert_eq!(Fp8E5M2::from_bits(byte).to_bits(), byte);
            bits += 1;
        }

        let mut nibble = 0u8;
        while nibble < 16 {
            assert_eq!(Fp4E2M1::from_bits(nibble).to_bits(), nibble);
            nibble += 1;
        }
    }
}
