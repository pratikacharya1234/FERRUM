//! Data type definitions for tensor elements.
//!
//! FERRUM supports a variety of numeric types optimized for different use cases:
//!
//! | DType | Size | Use Case |
//! |-------|------|----------|
//! | F32   | 4B   | Default training dtype |
//! | F64   | 8B   | High precision computations |
//! | F16   | 2B   | Mixed precision training |
//! | BF16  | 2B   | Training (better range than F16) |
//! | I32   | 4B   | Indices, labels |
//! | I64   | 8B   | Large indices |
//! | U8    | 1B   | Image data, quantization |
//! | Bool  | 1B   | Masks, conditions |
//!
//! ## Type Safety
//!
//! Operations between incompatible dtypes will return [`FerrumError::DTypeMismatch`].
//! Use [`Tensor::to_dtype`] for explicit conversions.

use std::fmt;

#[allow(unused_imports)]
use bytemuck::{Pod, Zeroable};
#[allow(unused_imports)]
use half::{bf16, f16};
#[allow(unused_imports)]
use num_traits::{Float, Num, NumCast, One, Zero};

/// Supported tensor element data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    /// 32-bit floating point (default for training).
    F32 = 0,
    /// 64-bit floating point (high precision).
    F64 = 1,
    /// 16-bit floating point (IEEE 754).
    F16 = 2,
    /// 16-bit brain floating point (better range than F16).
    BF16 = 3,
    /// 32-bit signed integer.
    I32 = 4,
    /// 64-bit signed integer.
    I64 = 5,
    /// 8-bit unsigned integer.
    U8 = 6,
    /// Boolean (stored as u8).
    Bool = 7,
}

impl DType {
    /// Size of a single element in bytes.
    #[inline]
    pub const fn size_of(&self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::Bool => 1,
        }
    }

    /// Whether this dtype supports gradient computation.
    #[inline]
    pub const fn supports_grad(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    /// Whether this is a floating point type.
    #[inline]
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    /// Whether this is an integer type.
    #[inline]
    pub const fn is_integer(&self) -> bool {
        matches!(self, DType::I32 | DType::I64 | DType::U8)
    }

    /// Whether this is a signed type.
    #[inline]
    pub const fn is_signed(&self) -> bool {
        !matches!(self, DType::U8 | DType::Bool)
    }

    /// Get the default dtype for floating point operations.
    #[inline]
    pub const fn default_float() -> Self {
        DType::F32
    }

    /// Get the default dtype for integer operations.
    #[inline]
    pub const fn default_int() -> Self {
        DType::I64
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "f32"),
            DType::F64 => write!(f, "f64"),
            DType::F16 => write!(f, "f16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::I64 => write!(f, "i64"),
            DType::U8 => write!(f, "u8"),
            DType::Bool => write!(f, "bool"),
        }
    }
}

/// Marker trait for types that can be tensor elements.
///
/// This trait is sealed and cannot be implemented outside this crate.
/// All element types must be:
/// - Pod (plain old data)
/// - Copy
/// - Send + Sync (for parallel operations)
///
/// # Safety
///
/// Implementors must ensure the type is truly plain-old-data with no
/// internal pointers or non-trivial destructors.
pub trait Element:
    Copy + Clone + Send + Sync + Default + fmt::Debug + 'static + private::Sealed
{
    /// The corresponding [`DType`] for this element type.
    const DTYPE: DType;

    /// Convert from f64 (used for initialization).
    fn from_f64(v: f64) -> Self;

    /// Convert to f64 (used for display and debugging).
    fn to_f64(self) -> f64;

    /// Zero value.
    fn zero() -> Self;

    /// One value.
    fn one() -> Self;
}

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    // f16/bf16 disabled for now
    // impl Sealed for half::f16 {}
    // impl Sealed for half::bf16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u8 {}
    // bool disabled for now
    // impl Sealed for bool {}
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

impl Element for f64 {
    const DTYPE: DType = DType::F64;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self
    }

    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn one() -> Self {
        1.0
    }
}

// Note: f16/bf16 Element implementations are disabled until bytemuck support is added
// They require the `bytemuck` feature in the `half` crate
/*
impl Element for f16 {
    const DTYPE: DType = DType::F16;

    #[inline]
    fn from_f64(v: f64) -> Self {
        f16::from_f64(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn zero() -> Self {
        f16::ZERO
    }

    #[inline]
    fn one() -> Self {
        f16::ONE
    }
}

impl Element for bf16 {
    const DTYPE: DType = DType::BF16;

    #[inline]
    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    #[inline]
    fn zero() -> Self {
        bf16::ZERO
    }

    #[inline]
    fn one() -> Self {
        bf16::ONE
    }
}
*/

impl Element for i32 {
    const DTYPE: DType = DType::I32;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i32
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v as i64
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v.clamp(0.0, 255.0) as u8
    }

    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }

    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }
}

// Note: bool doesn't implement Pod, so we use a wrapper type internally
// For now, bool Element is disabled
/*
#[derive(Clone, Copy, Default, Debug)]
#[repr(transparent)]
struct BoolWrapper(u8);
unsafe impl Zeroable for BoolWrapper {}
unsafe impl Pod for BoolWrapper {}

impl Element for bool {
    const DTYPE: DType = DType::Bool;

    #[inline]
    fn from_f64(v: f64) -> Self {
        v != 0.0
    }

    #[inline]
    fn to_f64(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }

    #[inline]
    fn zero() -> Self {
        false
    }

    #[inline]
    fn one() -> Self {
        true
    }
}
*/

/// Trait for floating point element types (supports autograd).
pub trait FloatElement: Element + num_traits::Float {
    /// Square root.
    fn sqrt(self) -> Self;

    /// Exponential.
    fn exp(self) -> Self;

    /// Natural logarithm.
    fn ln(self) -> Self;

    /// Power.
    fn powf(self, n: Self) -> Self;

    /// Absolute value.
    fn abs(self) -> Self;

    /// Minimum.
    fn min(self, other: Self) -> Self;

    /// Maximum.
    fn max(self, other: Self) -> Self;
}

impl FloatElement for f32 {
    #[inline]
    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f32::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f32::ln(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f32::powf(self, n)
    }
    #[inline]
    fn abs(self) -> Self {
        f32::abs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }
}

impl FloatElement for f64 {
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn powf(self, n: Self) -> Self {
        f64::powf(self, n)
    }
    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }
    #[inline]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }
    #[inline]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_of(), 4);
        assert_eq!(DType::F64.size_of(), 8);
        assert_eq!(DType::F16.size_of(), 2);
        assert_eq!(DType::Bool.size_of(), 1);
    }

    #[test]
    fn test_dtype_properties() {
        assert!(DType::F32.is_float());
        assert!(DType::F32.supports_grad());
        assert!(!DType::I32.is_float());
        assert!(!DType::I32.supports_grad());
    }

    #[test]
    fn test_element_conversions() {
        assert_eq!(f32::from_f64(3.14), 3.14f32);
        assert_eq!(i32::from_f64(3.7), 3);
        // bool Element disabled for now
        // assert_eq!(bool::from_f64(1.0), true);
        // assert_eq!(bool::from_f64(0.0), false);
    }
}
