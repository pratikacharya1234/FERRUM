//! Error types for FERRUM operations.
//!
//! All fallible operations in FERRUM return [`Result<T>`] which uses [`FerrumError`]
//! as the error type. This provides rich context for debugging while maintaining
//! zero-cost abstractions in the success path.

use thiserror::Error;

use crate::{DType, Device, Shape};

/// Result type alias for FERRUM operations.
pub type Result<T> = std::result::Result<T, FerrumError>;

/// Comprehensive error type for all FERRUM operations.
///
/// Errors are designed to be actionable - each variant provides enough context
/// to understand what went wrong and how to fix it.
///
/// Note: Large fields like Shape are boxed to keep the error type small,
/// following Rust best practices for Result ergonomics.
#[derive(Error, Debug, Clone)]
pub enum FerrumError {
    /// Shape mismatch in an operation (e.g., matmul with incompatible dimensions).
    #[error("Shape mismatch: {operation} expects {expected}, got {actual}")]
    ShapeMismatch {
        operation: &'static str,
        expected: String,
        actual: String,
    },

    /// Broadcasting failed between two shapes.
    #[error("Cannot broadcast shapes {lhs:?} and {rhs:?}")]
    BroadcastError {
        lhs: Box<Shape>,
        rhs: Box<Shape>,
    },

    /// Data type mismatch between tensors.
    #[error("DType mismatch: {operation} expects {expected:?}, got {actual:?}")]
    DTypeMismatch {
        operation: &'static str,
        expected: DType,
        actual: DType,
    },

    /// Operation not supported for the given dtype.
    #[error("Operation '{operation}' not supported for dtype {dtype:?}")]
    UnsupportedDType {
        operation: &'static str,
        dtype: DType,
    },

    /// Device mismatch between tensors.
    #[error("Device mismatch: {operation} requires tensors on same device, got {device1:?} and {device2:?}")]
    DeviceMismatch {
        operation: &'static str,
        device1: Device,
        device2: Device,
    },

    /// Invalid axis specification.
    #[error("Invalid axis {axis} for tensor with {ndim} dimensions")]
    InvalidAxis { axis: i64, ndim: usize },

    /// Index out of bounds.
    #[error("Index {index} out of bounds for dimension {dim} with size {size}")]
    IndexOutOfBounds { index: i64, dim: usize, size: usize },

    /// Invalid shape specification.
    #[error("Invalid shape: {message}")]
    InvalidShape { message: String },

    /// Cannot reshape tensor to target shape.
    #[error("Cannot reshape tensor with {src_numel} elements to shape {target:?} ({target_numel} elements)")]
    ReshapeError {
        src_numel: usize,
        target: Box<Shape>,
        target_numel: usize,
    },

    /// Storage is not contiguous when required.
    #[error("Operation '{operation}' requires contiguous storage")]
    NonContiguous { operation: &'static str },

    /// Arithmetic error (division by zero, etc.).
    #[error("Arithmetic error in '{operation}': {message}")]
    ArithmeticError {
        operation: &'static str,
        message: String,
    },

    /// Memory allocation failure.
    #[error("Failed to allocate {bytes} bytes on {device:?}")]
    AllocationError { bytes: usize, device: Device },

    /// Internal invariant violation (bug in FERRUM).
    #[error("Internal error: {message}. This is a bug, please report it.")]
    InternalError { message: String },

    /// Feature not yet implemented.
    #[error("Not implemented: {feature}")]
    NotImplemented { feature: String },

    /// Gradient computation error.
    #[error("Autograd error: {message}")]
    AutogradError { message: String },

    /// Serialization/deserialization error.
    #[error("Serialization error: {message}")]
    SerializationError { message: String },
}

impl FerrumError {
    /// Create a shape mismatch error.
    pub fn shape_mismatch(
        operation: &'static str,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::ShapeMismatch {
            operation,
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create an internal error (indicates a bug).
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Create a not-implemented error.
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = FerrumError::shape_mismatch("matmul", "[128, 256]", "[64, 128]");
        assert!(err.to_string().contains("matmul"));
        assert!(err.to_string().contains("[128, 256]"));
    }
}
