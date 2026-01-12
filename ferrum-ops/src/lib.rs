//! # FERRUM Ops
//!
//! Optimized tensor operations with CPU SIMD and parallel execution.
//!
//! This crate provides high-performance implementations of common tensor operations.
//! Currently supports CPU with future support planned for CUDA and Metal.

pub mod binary;
pub mod matmul;
pub mod reduce;
pub mod unary;

/// Re-exports for convenience.
pub mod prelude {
    pub use crate::binary::*;
    pub use crate::matmul::*;
    pub use crate::reduce::*;
    pub use crate::unary::*;
}
