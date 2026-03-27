/// Dispatch scalar multiplication: use GLV if params are provided, otherwise double-and-add.
macro_rules! impl_scalar_mul_dispatch {
    // GLV variant: use the endomorphism-based fast multiplication
    ($name:ident, $base:ident, $scalar:ident, glv($glv_params:expr)) => {
        fn scalar_mul(point: &Self, scalar: &$scalar) -> Self {
            fn endo_fn(p: &$name) -> $name {
                $name {
                    x: p.x * <$base as ff::WithSmallOrderMulGroup<3>>::ZETA,
                    y: p.y,
                    z: p.z,
                }
            }
            fn s_to_raw(s: &$scalar) -> [u64; 4] {
                let repr = s.to_repr();
                let b: &[u8] = repr.as_ref();
                let mut limbs = [0u64; 4];
                for i in 0..4 {
                    let mut bytes = [0u8; 8];
                    bytes.copy_from_slice(&b[i * 8..(i + 1) * 8]);
                    limbs[i] = u64::from_le_bytes(bytes);
                }
                limbs
            }
            fn s_from_raw(limbs: [u64; 4]) -> $scalar {
                $scalar::from_raw(limbs)
            }
            fn s_mul(a: &$scalar, b: &$scalar) -> $scalar { *a * *b }
            fn s_add(a: &$scalar, b: &$scalar) -> $scalar { *a + *b }
            fn s_neg(a: &$scalar) -> $scalar { -*a }
            fn s_sub(a: &$scalar, b: &$scalar) -> $scalar { *a - *b }

            let lambda = <$scalar as ff::WithSmallOrderMulGroup<3>>::ZETA;
            crate::glv::glv_mul(
                point, scalar, &$glv_params, endo_fn,
                s_to_raw, s_from_raw, s_mul, s_add, s_neg, s_sub, &lambda,
            )
        }
    };
    // Default variant: simple double-and-add
    ($name:ident, $base:ident, $scalar:ident) => {
        fn scalar_mul(point: &Self, scalar: &$scalar) -> Self {
            Self::double_and_add(point, scalar)
        }
    };
}

macro_rules! impl_add_binop_specify_output {
    ($lhs:ident, $rhs:ident, $output:ident) => {
        impl<'b> ::core::ops::Add<&'b $rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn add(self, rhs: &'b $rhs) -> $output {
                &self + rhs
            }
        }

        impl<'a> ::core::ops::Add<$rhs> for &'a $lhs {
            type Output = $output;

            #[inline]
            fn add(self, rhs: $rhs) -> $output {
                self + &rhs
            }
        }

        impl ::core::ops::Add<$rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn add(self, rhs: $rhs) -> $output {
                &self + &rhs
            }
        }
    };
}

macro_rules! impl_sub_binop_specify_output {
    ($lhs:ident, $rhs:ident, $output:ident) => {
        impl<'b> ::core::ops::Sub<&'b $rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn sub(self, rhs: &'b $rhs) -> $output {
                &self - rhs
            }
        }

        impl<'a> ::core::ops::Sub<$rhs> for &'a $lhs {
            type Output = $output;

            #[inline]
            fn sub(self, rhs: $rhs) -> $output {
                self - &rhs
            }
        }

        impl ::core::ops::Sub<$rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn sub(self, rhs: $rhs) -> $output {
                &self - &rhs
            }
        }
    };
}

macro_rules! impl_binops_additive_specify_output {
    ($lhs:ident, $rhs:ident, $output:ident) => {
        impl_add_binop_specify_output!($lhs, $rhs, $output);
        impl_sub_binop_specify_output!($lhs, $rhs, $output);
    };
}

macro_rules! impl_binops_multiplicative_mixed {
    ($lhs:ident, $rhs:ident, $output:ident) => {
        impl<'b> ::core::ops::Mul<&'b $rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn mul(self, rhs: &'b $rhs) -> $output {
                &self * rhs
            }
        }

        impl<'a> ::core::ops::Mul<$rhs> for &'a $lhs {
            type Output = $output;

            #[inline]
            fn mul(self, rhs: $rhs) -> $output {
                self * &rhs
            }
        }

        impl ::core::ops::Mul<$rhs> for $lhs {
            type Output = $output;

            #[inline]
            fn mul(self, rhs: $rhs) -> $output {
                &self * &rhs
            }
        }
    };
}

macro_rules! impl_binops_additive {
    ($lhs:ident, $rhs:ident) => {
        impl_binops_additive_specify_output!($lhs, $rhs, $lhs);

        impl ::core::ops::SubAssign<$rhs> for $lhs {
            #[inline]
            fn sub_assign(&mut self, rhs: $rhs) {
                *self = &*self - &rhs;
            }
        }

        impl ::core::ops::AddAssign<$rhs> for $lhs {
            #[inline]
            fn add_assign(&mut self, rhs: $rhs) {
                *self = &*self + &rhs;
            }
        }

        impl<'b> ::core::ops::SubAssign<&'b $rhs> for $lhs {
            #[inline]
            fn sub_assign(&mut self, rhs: &'b $rhs) {
                *self = &*self - rhs;
            }
        }

        impl<'b> ::core::ops::AddAssign<&'b $rhs> for $lhs {
            #[inline]
            fn add_assign(&mut self, rhs: &'b $rhs) {
                *self = &*self + rhs;
            }
        }
    };
}

macro_rules! impl_binops_multiplicative {
    ($lhs:ident, $rhs:ident) => {
        impl_binops_multiplicative_mixed!($lhs, $rhs, $lhs);

        impl ::core::ops::MulAssign<$rhs> for $lhs {
            #[inline]
            fn mul_assign(&mut self, rhs: $rhs) {
                *self = &*self * &rhs;
            }
        }

        impl<'b> ::core::ops::MulAssign<&'b $rhs> for $lhs {
            #[inline]
            fn mul_assign(&mut self, rhs: &'b $rhs) {
                *self = &*self * rhs;
            }
        }
    };
}
