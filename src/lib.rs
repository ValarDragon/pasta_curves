//! Implementation of the Pallas / Vesta curve cycle.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(unknown_lints)]
#![allow(clippy::op_ref, clippy::same_item_push, clippy::upper_case_acronyms)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

#[macro_use]
mod macros;
mod curves;
mod fields;
mod glv;

pub mod arithmetic;
pub mod pallas;
pub mod vesta;

#[cfg(feature = "alloc")]
mod hashtocurve;

#[cfg(feature = "serde")]
mod serde_impl;

pub use curves::*;
pub use fields::*;

pub extern crate group;

#[cfg(feature = "alloc")]
#[test]
fn test_endo_consistency() {
    use crate::arithmetic::CurveExt;
    use group::{ff::WithSmallOrderMulGroup, Group};

    let a = pallas::Point::generator();
    assert_eq!(a * pallas::Scalar::ZETA, a.endo());
    let a = vesta::Point::generator();
    assert_eq!(a * vesta::Scalar::ZETA, a.endo());
}

#[cfg(feature = "alloc")]
#[test]
fn test_glv_scalar_mul_pallas() {
    use group::Group;

    let g = pallas::Point::generator();

    // Test with small scalars
    assert_eq!(g * pallas::Scalar::one(), g);
    assert_eq!(g * pallas::Scalar::zero(), pallas::Point::identity());
    assert_eq!(g * (-pallas::Scalar::one()), -g);

    // Test with scalar 2
    let two = pallas::Scalar::one() + pallas::Scalar::one();
    assert_eq!(g * two, g + g);

    // Test with scalar 3
    let three = two + pallas::Scalar::one();
    assert_eq!(g * three, g + g + g);

    // Test with random scalars: verify GLV gives the same result as double-and-add
    // by comparing with the known generator identity: [n]G = O
    let neg_one = -pallas::Scalar::one();
    let p = g * neg_one;
    assert_eq!(p + g, pallas::Point::identity());

    // Test with a random-ish point
    let h = g + g + g + g + g; // 5G
    let scalar = pallas::Scalar::from(12345u64);
    let result = h * scalar;
    // Verify by computing 5 * 12345 * G = 61725 * G
    let expected = g * pallas::Scalar::from(61725u64);
    assert_eq!(result, expected);
}

#[cfg(feature = "alloc")]
#[test]
fn test_glv_scalar_mul_vesta() {
    use group::Group;

    let g = vesta::Point::generator();

    // Test with small scalars
    assert_eq!(g * vesta::Scalar::one(), g);
    assert_eq!(g * vesta::Scalar::zero(), vesta::Point::identity());
    assert_eq!(g * (-vesta::Scalar::one()), -g);

    // Test with scalar 2
    let two = vesta::Scalar::one() + vesta::Scalar::one();
    assert_eq!(g * two, g + g);

    // Test with a random-ish point
    let h = g + g + g + g + g; // 5G
    let scalar = vesta::Scalar::from(12345u64);
    let result = h * scalar;
    let expected = g * vesta::Scalar::from(61725u64);
    assert_eq!(result, expected);
}

#[cfg(feature = "alloc")]
#[test]
fn test_glv_large_scalars() {
    use group::Group;

    let g = pallas::Point::generator();

    // Test with ZETA (cube root of unity in scalar field)
    // [zeta]G should equal endo(G) = (zeta_base * x, y)
    use crate::arithmetic::CurveExt;
    use ff::WithSmallOrderMulGroup;
    let zeta_result = g * pallas::Scalar::ZETA;
    assert_eq!(zeta_result, g.endo());

    // [zeta^2]G should equal endo(endo(G))
    let zeta_sq = pallas::Scalar::ZETA * pallas::Scalar::ZETA;
    let zeta_sq_result = g * zeta_sq;
    assert_eq!(zeta_sq_result, g.endo().endo());

    // [zeta^3]G = G (since zeta^3 = 1)
    let zeta_cu = zeta_sq * pallas::Scalar::ZETA;
    assert_eq!(zeta_cu, pallas::Scalar::one());
    assert_eq!(g * zeta_cu, g);
}

#[cfg(feature = "alloc")]
#[test]
fn test_glv_identity_point() {
    use group::Group;

    let id = pallas::Point::identity();
    let scalar = pallas::Scalar::from(42u64);
    assert_eq!(id * scalar, pallas::Point::identity());
}

#[cfg(feature = "alloc")]
#[test]
fn test_glv_affine_mul() {
    use group::{prime::PrimeCurveAffine, Group};

    // Test that affine multiplication also uses GLV correctly
    let g = pallas::Affine::generator();
    let scalar = pallas::Scalar::from(42u64);
    let result = g * scalar;

    let g_proj = pallas::Point::generator();
    let expected = g_proj * scalar;
    assert_eq!(result, expected);
}
