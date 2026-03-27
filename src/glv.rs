//! GLV (Gallant-Lambert-Vanstone) optimization for scalar multiplication.
//!
//! For curves with j-invariant 0 (y² = x³ + b), there exists an efficient
//! endomorphism φ(x,y) = (ζx, y) where ζ is a cube root of unity. The GLV
//! method decomposes a scalar k into two half-size scalars k₁, k₂ such that
//! [k]P = [k₁]P + [k₂]φ(P), which is faster via Shamir's trick.
//!
//! The scalar decomposition uses only division by powers of two (bit shifts)
//! for constant-time operation.

use crate::arithmetic::{adc, mac, sbb};
use subtle::{Choice, ConditionallySelectable};

/// GLV parameters for a specific curve.
///
/// The decomposition computes k₁, k₂ from scalar k using:
///   c₁ = (g₁ · k) >> S
///   c₂ = (g₂ · k) >> S
///   k₁ = k − c₁·a₁ − c₂·a₂
///   k₂ = c₁·|b₁| − c₂·b₂
///
/// where S = 384 and all divisions are bit shifts.
pub(crate) struct GlvParams {
    /// Lattice vector component a₁ (positive, 2 limbs)
    pub a1: [u64; 2],
    /// Lattice vector component |b₁| (always stored positive, 2 limbs)
    pub b1_abs: [u64; 2],
    /// Lattice vector component a₂ (positive, 2 limbs)
    pub a2: [u64; 2],
    /// Lattice vector component b₂ (positive, 2 limbs)
    pub b2: [u64; 2],
    /// Precomputed g₁ = round(2^384 · b₂ / n), 4 limbs
    pub g1: [u64; 4],
    /// Precomputed g₂ = round(2^384 · |b₁| / n), 5 limbs
    pub g2: [u64; 5],
}

// --- Pallas GLV constants ---
// Scalar field: Fq, order q = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001
// LAMBDA (eigenvalue of endomorphism) = Fq::ZETA
pub(crate) const PALLAS_GLV: GlvParams = GlvParams {
    a1: [0x7fcae1c700000001, 0x49e69d1640f04915],
    b1_abs: [0x8cb1279300000000, 0x49e69d1640a89953],
    a2: [0x8c46eb2100000002, 0xddb3d742c2892b7e],
    b2: [0xf319ba3400000001, 0x47afc1],
    g1: [0x7bf422ae2d81872b, 0xffffffffff666e35, 0xcc66e8d000000003, 0x11ebf07],
    g2: [0x4a95a2d972171db4, 0x61afdea68480fa55, 0x32c49e4bffffffff, 0x279a745902a2654e, 0x1],
};

// --- Vesta GLV constants ---
// Scalar field: Fp, order p = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
// LAMBDA (eigenvalue of endomorphism) = Fp::ZETA
pub(crate) const VESTA_GLV: GlvParams = GlvParams {
    a1: [0x7fcae1c700000000, 0x49e69d1640f04915],
    b1_abs: [0x8cb1279300000001, 0x49e69d1640a89953],
    a2: [0x8c46eb2100000001, 0xddb3d742c2892b7e],
    b2: [0xf319ba33ffffffff, 0x47afc1],
    g1: [0x7bf563dd917ae05e, 0xffffffffff666e35, 0xcc66e8cffffffffb, 0x11ebf07],
    g2: [0x841414c24bf99a83, 0x61afdea685cc1578, 0x32c49e4c00000003, 0x279a745902a2654e, 0x1],
};

/// Result of GLV scalar decomposition.
/// k₁ and k₂ are represented as (sign, magnitude) where magnitude fits in ~129 bits (3 limbs).
/// sign = true means negative.
pub(crate) struct GlvDecomposition {
    pub k1_neg: Choice,
    pub k1: [u64; 3],
    pub k2_neg: Choice,
    pub k2: [u64; 3],
}

/// Compute the high limbs of an n-limb × 4-limb product, starting at position `skip`.
///
/// Returns `out.len()` limbs representing `(a * b) >> (skip * 64)`.
/// Low columns (< skip) are computed only for their carry contribution,
/// their values are discarded.
#[inline]
fn mul_high(a: &[u64], b: &[u64; 4], skip: usize, out: &mut [u64]) {
    let n = a.len();
    let total = n + 4;
    for o in out.iter_mut() {
        *o = 0;
    }

    // Column-by-column schoolbook multiplication. For each column `col`,
    // sum all partial products a[i]*b[j] where i+j = col.
    // Only store columns >= skip; earlier columns just propagate carries.
    let mut carry = 0u64;
    for col in 0..total {
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

/// Multiply two small numbers and subtract from a wider number, returning
/// the low 3 limbs and a sign bit. This handles the pattern:
///   result = wide - mul_a - mul_b  (for k1)
///   result = mul_a - mul_b         (for k2)
///
/// All values involved are at most ~256 bits, result is ~128 bits.

/// Widening multiply: a (2 limbs) * b (2 limbs) -> 4 limbs
#[inline]
fn wmul_2x2(a: &[u64; 2], b: &[u64; 2]) -> [u64; 4] {
    let (r0, carry) = mac(0, a[0], b[0], 0);
    let (r1, carry) = mac(0, a[0], b[1], carry);
    let r2 = carry;

    let (r1, carry) = mac(r1, a[1], b[0], 0);
    let (r2, carry) = mac(r2, a[1], b[1], carry);
    let r3 = carry;

    [r0, r1, r2, r3]
}

/// Widening multiply: a (3 limbs) * b (2 limbs) -> 5 limbs
#[inline]
fn wmul_3x2(a: &[u64; 3], b: &[u64; 2]) -> [u64; 5] {
    let mut out = [0u64; 5];
    for i in 0..3 {
        let mut carry = 0u64;
        for j in 0..2 {
            let (lo, hi) = mac(out[i + j], a[i], b[j], carry);
            out[i + j] = lo;
            carry = hi;
        }
        out[i + 2] = carry;
    }
    out
}

/// Subtract b from a (5 limbs each), returning (result, borrow_flag).
/// borrow_flag has bit 63 set if a < b (result is negative in two's complement).
#[inline]
fn sub5(a: &[u64; 5], b: &[u64; 5]) -> ([u64; 5], u64) {
    let (r0, borrow) = sbb(a[0], b[0], 0);
    let (r1, borrow) = sbb(a[1], b[1], borrow);
    let (r2, borrow) = sbb(a[2], b[2], borrow);
    let (r3, borrow) = sbb(a[3], b[3], borrow);
    let (r4, borrow) = sbb(a[4], b[4], borrow);
    ([r0, r1, r2, r3, r4], borrow)
}

/// Negate a 5-limb number (two's complement: 0 - a).
#[inline]
fn neg5(a: &[u64; 5]) -> [u64; 5] {
    let (r0, borrow) = sbb(0, a[0], 0);
    let (r1, borrow) = sbb(0, a[1], borrow);
    let (r2, borrow) = sbb(0, a[2], borrow);
    let (r3, borrow) = sbb(0, a[3], borrow);
    let (r4, _) = sbb(0, a[4], borrow);
    [r0, r1, r2, r3, r4]
}

/// Add two 5-limb numbers.
#[inline]
fn add5(a: &[u64; 5], b: &[u64; 5]) -> [u64; 5] {
    let (r0, carry) = adc(a[0], b[0], 0);
    let (r1, carry) = adc(a[1], b[1], carry);
    let (r2, carry) = adc(a[2], b[2], carry);
    let (r3, carry) = adc(a[3], b[3], carry);
    let (r4, _) = adc(a[4], b[4], carry);
    [r0, r1, r2, r3, r4]
}

/// Extract (sign, |value|) from a 5-limb two's complement result.
/// Returns (is_negative, [low 3 limbs of absolute value]).
#[inline]
fn extract_signed(val: &[u64; 5], borrow: u64) -> (Choice, [u64; 3]) {
    let is_neg = Choice::from((borrow >> 63) as u8);
    let neg_val = neg5(val);
    let abs0 = u64::conditional_select(&val[0], &neg_val[0], is_neg);
    let abs1 = u64::conditional_select(&val[1], &neg_val[1], is_neg);
    let abs2 = u64::conditional_select(&val[2], &neg_val[2], is_neg);
    (is_neg, [abs0, abs1, abs2])
}

/// Decompose a 256-bit scalar k into k₁, k₂ such that k ≡ k₁ + k₂·λ (mod n).
/// Both |k₁| and |k₂| are at most ~129 bits.
/// All divisions are by powers of two (bit shifts only).
pub(crate) fn decompose_scalar(k: &[u64; 4], params: &GlvParams) -> GlvDecomposition {
    // c₁ = (g₁ · k) >> 384, c₂ = (g₂ · k) >> 384
    // Only compute the high 2 (or 3) limbs — positions [6..] of the full product.
    let mut c1 = [0u64; 2];
    mul_high(&params.g1, k, 6, &mut c1);

    let mut c2 = [0u64; 3];
    mul_high(&params.g2[..], k, 6, &mut c2);

    // k₁ = k − c₁·a₁ − c₂·a₂
    let c1a1 = wmul_2x2(&c1, &params.a1);
    let c2a2 = wmul_3x2(&c2, &params.a2);
    let sum_a = add5(
        &[c1a1[0], c1a1[1], c1a1[2], c1a1[3], 0],
        &c2a2,
    );
    let (k1_raw, k1_borrow) = sub5(
        &[k[0], k[1], k[2], k[3], 0],
        &sum_a,
    );
    let (k1_neg, k1_abs) = extract_signed(&k1_raw, k1_borrow);

    // k₂ = c₁·|b₁| − c₂·b₂
    let c1b1 = wmul_2x2(&c1, &params.b1_abs);
    let c2b2 = wmul_3x2(&c2, &params.b2);
    let (k2_raw, k2_borrow) = sub5(
        &[c1b1[0], c1b1[1], c1b1[2], c1b1[3], 0],
        &c2b2,
    );
    let (k2_neg, k2_abs) = extract_signed(&k2_raw, k2_borrow);

    GlvDecomposition {
        k1_neg: k1_neg,
        k1: k1_abs,
        k2_neg: k2_neg,
        k2: k2_abs,
    }
}

/// Get bit i from a 3-limb (192-bit) value.
#[inline]
fn get_bit(val: &[u64; 3], i: usize) -> Choice {
    let limb = val[i / 64];
    Choice::from(((limb >> (i % 64)) & 1) as u8)
}

/// Find the highest set bit position across two 192-bit values.
/// Returns the bit count (number of bits needed) or 0 if both are zero.
fn highest_bit(a: &[u64; 3], b: &[u64; 3]) -> usize {
    fn bit_length(v: &[u64; 3]) -> usize {
        if v[2] != 0 {
            192 - v[2].leading_zeros() as usize
        } else if v[1] != 0 {
            128 - v[1].leading_zeros() as usize
        } else if v[0] != 0 {
            64 - v[0].leading_zeros() as usize
        } else {
            0
        }
    }
    let a_bits = bit_length(a);
    let b_bits = bit_length(b);
    if a_bits > b_bits { a_bits } else { b_bits }
}

/// GLV scalar multiplication: compute [k]P using the endomorphism.
///
/// Decomposes k = k₁ + k₂·λ, then computes [k₁]P + [k₂]φ(P) using
/// Shamir's trick with a 4-entry lookup table.
///
/// The `endo_fn` applies the curve endomorphism φ(x,y) = (ζx, y).
pub(crate) fn glv_mul<P>(
    point: &P,
    scalar_repr: &[u8; 32],
    params: &GlvParams,
    endo_fn: fn(&P) -> P,
) -> P
where
    P: ConditionallySelectable
        + core::ops::Add<Output = P>
        + for<'a> core::ops::Add<&'a P, Output = P>
        + core::ops::Neg<Output = P>
        + Copy,
{
    // Convert scalar repr (little-endian bytes) to 4 u64 limbs
    let mut k = [0u64; 4];
    for i in 0..4 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&scalar_repr[i * 8..(i + 1) * 8]);
        k[i] = u64::from_le_bytes(bytes);
    }

    let decomp = decompose_scalar(&k, params);

    // Conditionally negate points based on scalar signs
    let p1_pos = *point;
    let p1_neg = -*point;
    let p1 = P::conditional_select(&p1_pos, &p1_neg, decomp.k1_neg);

    let p2_pos = endo_fn(point);
    let p2_neg = -p2_pos;
    let p2 = P::conditional_select(&p2_pos, &p2_neg, decomp.k2_neg);

    let p12 = p1 + &p2;

    // Find the number of bits to iterate (constant for a given scalar size,
    // but we cap at the actual highest bit for efficiency)
    let num_bits = highest_bit(&decomp.k1, &decomp.k2);

    // Identity element (P - P)
    let identity = p1 + &(-p1);

    if num_bits == 0 {
        return identity;
    }

    // Process the highest bit first to initialize the accumulator
    // At the top bit, at least one of k1/k2 has this bit set
    let k1_top = get_bit(&decomp.k1, num_bits - 1);
    let k2_top = get_bit(&decomp.k2, num_bits - 1);

    // 4-way constant-time table lookup
    let s01 = P::conditional_select(&identity, &p1, k1_top);
    let s23 = P::conditional_select(&p2, &p12, k1_top);
    let mut acc = P::conditional_select(&s01, &s23, k2_top);

    // Process remaining bits from high to low
    if num_bits >= 2 {
        for i in (0..num_bits - 1).rev() {
            acc = acc + &acc; // double

            let k1_bit = get_bit(&decomp.k1, i);
            let k2_bit = get_bit(&decomp.k2, i);

            // 4-way constant-time table lookup
            let s01 = P::conditional_select(&identity, &p1, k1_bit);
            let s23 = P::conditional_select(&p2, &p12, k1_bit);
            let to_add = P::conditional_select(&s01, &s23, k2_bit);

            // Always add (identity addition is handled by the add impl)
            acc = acc + &to_add;
        }
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompose_pallas() {
        // Test with a known scalar
        // k = 1: should decompose to k1=1, k2=0
        let k = [1u64, 0, 0, 0];
        let decomp = decompose_scalar(&k, &PALLAS_GLV);
        assert_eq!(decomp.k1, [1, 0, 0]);
        assert_eq!(decomp.k2, [0, 0, 0]);
        assert_eq!(decomp.k1_neg.unwrap_u8(), 0);
    }

    #[test]
    fn test_decompose_vesta() {
        let k = [1u64, 0, 0, 0];
        let decomp = decompose_scalar(&k, &VESTA_GLV);
        assert_eq!(decomp.k1, [1, 0, 0]);
        assert_eq!(decomp.k2, [0, 0, 0]);
        assert_eq!(decomp.k1_neg.unwrap_u8(), 0);
    }

    #[test]
    fn test_decompose_lambda_pallas() {
        // k = LAMBDA should decompose to small k1, k2
        let k = [0x2aa9d2e050aa0e4f, 0x0fed467d47c033af, 0x511db4d81cf70f5a, 0x06819a58283e528e];
        let decomp = decompose_scalar(&k, &PALLAS_GLV);
        // Verify the values are small (≤ ~130 bits, so top limb should be very small)
        assert!(decomp.k1[2] <= 3, "k1 too large: top limb = {}", decomp.k1[2]);
        assert!(decomp.k2[2] <= 3, "k2 too large: top limb = {}", decomp.k2[2]);
    }

    #[test]
    fn test_wmul_2x2() {
        let a = [3u64, 0];
        let b = [7u64, 0];
        let result = wmul_2x2(&a, &b);
        assert_eq!(result, [21, 0, 0, 0]);

        let a = [u64::MAX, 0];
        let b = [2u64, 0];
        let result = wmul_2x2(&a, &b);
        assert_eq!(result, [u64::MAX - 1, 1, 0, 0]);
    }

    #[test]
    fn test_mul_high() {
        // 1 * 5 = 5; high part (skip=6) should be all zeros
        let a = [1u64, 0, 0, 0];
        let b = [5u64, 0, 0, 0];
        let mut out = [0u64; 2];
        mul_high(&a, &b, 6, &mut out);
        assert_eq!(out, [0, 0]);

        // Test that mul_high gives the correct high limbs for a larger product
        let a = [u64::MAX, u64::MAX, u64::MAX, u64::MAX];
        let b = [1u64, 0, 0, 0];
        let mut out = [0u64; 2];
        mul_high(&a, &b, 6, &mut out);
        // Full product of [MAX,MAX,MAX,MAX] * [1,0,0,0] = [MAX,MAX,MAX,MAX,0,0,0,0]
        // Limbs [6..8] = [0, 0]
        assert_eq!(out, [0, 0]);
    }
}
