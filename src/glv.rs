//! GLV (Gallant-Lambert-Vanstone) optimization for scalar multiplication.
//!
//! For curves with j-invariant 0 (y² = x³ + b), there exists an efficient
//! endomorphism φ(x,y) = (ζx, y) where ζ is a cube root of unity. The GLV
//! method decomposes a scalar k into two half-size scalars k₁, k₂ such that
//! [k]P = [k₁]P + [k₂]φ(P), which is faster via Shamir's trick.
//!
//! The scalar decomposition uses only division by powers of two (bit shifts)
//! for constant-time operation.

use crate::arithmetic::{adc, mac};
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

/// GLV constants for a specific curve.
///
/// Following the secp256k1 approach: we store only b₁, b₂ and λ.
/// Decomposition uses modular scalar arithmetic:
///   c₁ = (g₁ · k) >> 384
///   c₂ = (g₂ · k) >> 384
///   k₂ = c₁·(−b₁) + c₂·(−b₂) mod n
///   k₁ = k − k₂·λ mod n
pub(crate) struct GlvParams {
    /// −b₁ mod n (4 limbs, raw non-Montgomery scalar)
    pub minus_b1: [u64; 4],
    /// −b₂ mod n (4 limbs, raw non-Montgomery scalar)
    pub minus_b2: [u64; 4],
    /// Precomputed g₁ = round(2^384 · |b₂| / n), stored as 4 limbs
    pub g1: [u64; 4],
    /// Precomputed g₂ = round(2^384 · |b₁| / n), stored as 5 limbs
    pub g2: [u64; 5],
}

// --- Pallas GLV constants ---
// Scalar field order: q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// LAMBDA = Fq::ZETA
pub(crate) const PALLAS_GLV: GlvParams = GlvParams {
    minus_b1: [0x8cb1279300000000, 0x49e69d1640a89953, 0x0, 0x0],
    minus_b2: [0x992d30ed00000000, 0x224698fc094cf91b, 0x0, 0x4000000000000000],
    g1: [0x7bf422ae2d81872b, 0xffffffffff666e35, 0xcc66e8d000000003, 0x11ebf07],
    g2: [0x4a95a2d972171db4, 0x61afdea68480fa55, 0x32c49e4bffffffff, 0x279a745902a2654e, 0x1],
};

// --- Vesta GLV constants ---
// Scalar field order: p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// LAMBDA = Fp::ZETA
pub(crate) const VESTA_GLV: GlvParams = GlvParams {
    minus_b1: [0x8cb1279300000001, 0x49e69d1640a89953, 0x0, 0x0],
    minus_b2: [0xa61376b900000002, 0x224698fc09054959, 0x0, 0x4000000000000000],
    g1: [0x7bf563dd917ae05e, 0xffffffffff666e35, 0xcc66e8cffffffffb, 0x11ebf07],
    g2: [0x841414c24bf99a83, 0x61afdea685cc1578, 0x32c49e4c00000003, 0x279a745902a2654e, 0x1],
};

/// Compute the high limbs of a product, `(a * b) >> (skip * 64)`.
///
/// Only computes from column (skip - 1) onward; the dropped carry from
/// lower columns may cause ±1 error, which is tolerable for GLV.
#[inline]
fn mul_high(a: &[u64], b: &[u64; 4], skip: usize, out: &mut [u64]) {
    let n = a.len();
    let total = n + 4;
    for o in out.iter_mut() {
        *o = 0;
    }

    let start = if skip > 0 { skip - 1 } else { 0 };
    let mut carry = 0u64;

    for col in start..total {
        let mut acc_lo = carry;
        let mut acc_hi = 0u64;
        carry = 0;

        let j_min = if col >= n { col - n + 1 } else { 0 };
        let j_max = if col < 4 { col + 1 } else { 4 };

        for j in j_min..j_max {
            let i = col - j;
            let (lo, hi) = mac(acc_lo, a[i], b[j], 0);
            acc_lo = lo;
            let (hi_sum, overflow) = adc(acc_hi, hi, 0);
            acc_hi = hi_sum;
            carry += overflow;
        }

        if col >= skip {
            out[col - skip] = acc_lo;
        }
        carry += acc_hi;
    }
}

/// GLV scalar multiplication: compute [k]P using the endomorphism.
///
/// Decomposes k = k₁ + k₂·λ via the lattice method, then computes
/// [k₁]P + [k₂]φ(P) using Shamir's trick.
///
/// - `scalar_mul_fn(point, scalar_raw)` performs standard scalar multiplication
///   on a raw (non-Montgomery) 4-limb scalar. Used for the half-size sub-multiplications
///   after decomposition.
/// - `endo_fn` applies the curve endomorphism φ(x,y) = (ζx, y).
/// - `scalar_field_ops` provides modular arithmetic in the scalar field:
///   `(mul, add, neg, to_raw, from_raw)`.
pub(crate) fn glv_mul<P, S>(
    point: &P,
    scalar: &S,
    params: &GlvParams,
    endo_fn: fn(&P) -> P,
    // Scalar field operations: we need mul, add, neg, and conversion to/from raw limbs
    scalar_to_raw: fn(&S) -> [u64; 4],
    scalar_from_raw: fn([u64; 4]) -> S,
    scalar_mul: fn(&S, &S) -> S,
    scalar_add: fn(&S, &S) -> S,
    scalar_neg: fn(&S) -> S,
    scalar_sub: fn(&S, &S) -> S,
    lambda: &S,
) -> P
where
    P: ConditionallySelectable
        + core::ops::Add<Output = P>
        + for<'a> core::ops::Add<&'a P, Output = P>
        + core::ops::Neg<Output = P>
        + Copy,
{
    let k_raw = scalar_to_raw(scalar);

    // c₁ = (g₁ · k) >> 384, c₂ = (g₂ · k) >> 384
    let mut c1_raw = [0u64; 4]; // only low 2 limbs will be nonzero
    mul_high(&params.g1, &k_raw, 6, &mut c1_raw[..2]);

    let mut c2_limbs = [0u64; 3];
    mul_high(&params.g2[..], &k_raw, 6, &mut c2_limbs);
    let c2_raw = [c2_limbs[0], c2_limbs[1], c2_limbs[2], 0u64];

    let c1 = scalar_from_raw(c1_raw);
    let c2 = scalar_from_raw(c2_raw);

    // k₂ = c₁·(−b₁) + c₂·(−b₂) mod n
    let minus_b1 = scalar_from_raw(params.minus_b1);
    let minus_b2 = scalar_from_raw(params.minus_b2);
    let k2 = scalar_add(
        &scalar_mul(&c1, &minus_b1),
        &scalar_mul(&c2, &minus_b2),
    );

    // k₁ = k − k₂·λ mod n
    let k1 = scalar_sub(scalar, &scalar_mul(&k2, lambda));

    // Now k1, k2 are ~128-bit scalars in the field.
    // Extract their raw representations to determine sign and magnitude.
    let k1_raw = scalar_to_raw(&k1);
    let k2_raw = scalar_to_raw(&k2);

    // Determine sign: positive decomposition values are < 2^130 (limb[3] == 0),
    // while negative values are represented as n - |val| ≈ 2^254 (limb[3] != 0).
    // Use constant-time comparison.
    let k1_neg = !k1_raw[3].ct_eq(&0);
    let k2_neg = !k2_raw[3].ct_eq(&0);

    let k1_final = {
        let neg_k1 = scalar_neg(&k1);
        let neg_raw = scalar_to_raw(&neg_k1);
        let pos_raw = k1_raw;
        [
            u64::conditional_select(&pos_raw[0], &neg_raw[0], k1_neg),
            u64::conditional_select(&pos_raw[1], &neg_raw[1], k1_neg),
            u64::conditional_select(&pos_raw[2], &neg_raw[2], k1_neg),
            u64::conditional_select(&pos_raw[3], &neg_raw[3], k1_neg),
        ]
    };
    let k2_final = {
        let neg_k2 = scalar_neg(&k2);
        let neg_raw = scalar_to_raw(&neg_k2);
        let pos_raw = k2_raw;
        [
            u64::conditional_select(&pos_raw[0], &neg_raw[0], k2_neg),
            u64::conditional_select(&pos_raw[1], &neg_raw[1], k2_neg),
            u64::conditional_select(&pos_raw[2], &neg_raw[2], k2_neg),
            u64::conditional_select(&pos_raw[3], &neg_raw[3], k2_neg),
        ]
    };

    // Conditionally negate points
    let p1 = P::conditional_select(point, &(-*point), k1_neg);
    let p2_base = endo_fn(point);
    let p2 = P::conditional_select(&p2_base, &(-p2_base), k2_neg);
    let p12 = p1 + &p2;

    // Identity element
    let identity = p1 + &(-p1);

    // Constant-time Shamir's trick: always iterate GLV_SCALAR_BITS bits
    // to avoid leaking scalar magnitude through timing.
    let k1_bit = |i: usize| -> Choice {
        Choice::from(((k1_final[i / 64] >> (i % 64)) & 1) as u8)
    };
    let k2_bit = |i: usize| -> Choice {
        Choice::from(((k2_final[i / 64] >> (i % 64)) & 1) as u8)
    };

    // Initialize with top bit
    let b1 = k1_bit(GLV_SCALAR_BITS - 1);
    let b2 = k2_bit(GLV_SCALAR_BITS - 1);
    let s01 = P::conditional_select(&identity, &p1, b1);
    let s23 = P::conditional_select(&p2, &p12, b1);
    let mut acc = P::conditional_select(&s01, &s23, b2);

    // Process remaining bits (fixed count, no data-dependent branching)
    for i in (0..GLV_SCALAR_BITS - 1).rev() {
        acc = acc + &acc; // double

        let b1 = k1_bit(i);
        let b2 = k2_bit(i);
        let s01 = P::conditional_select(&identity, &p1, b1);
        let s23 = P::conditional_select(&p2, &p12, b1);
        let to_add = P::conditional_select(&s01, &s23, b2);
        acc = acc + &to_add;
    }

    acc
}

/// Maximum number of bits in the decomposed half-size scalars.
/// The GLV decomposition produces |k₁|, |k₂| < ~2^129, and after
/// negation to ensure positivity, the values are < n/2 < 2^254.
/// However, we only need to iterate over 129 bits since the
/// decomposed scalars are bounded by the lattice vector norms (~2^128).
/// We use 130 to be safe (matching secp256k1's constant-time approach).
const GLV_SCALAR_BITS: usize = 130;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul_high() {
        let a = [1u64, 0, 0, 0];
        let b = [5u64, 0, 0, 0];
        let mut out = [0u64; 2];
        mul_high(&a, &b, 6, &mut out);
        assert_eq!(out, [0, 0]);

        let a = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let b = [1u64, 0, 0, 0];
        let mut out = [0u64; 2];
        mul_high(&a, &b, 6, &mut out);
        assert_eq!(out, [0, 0]);
    }
}
