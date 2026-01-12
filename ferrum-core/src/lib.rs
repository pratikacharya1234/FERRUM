//! # FERRUM Core
//!
//! Core tensor primitives and foundational types for the FERRUM deep learning framework.
//!
//! This crate provides:
//! - [`Tensor`] - N-dimensional array with automatic memory management
//! - [`DType`] - Data type enumeration (F32, F64, F16, BF16, I32, I64, Bool)
//! - [`Device`] - Device abstraction (CPU, future: CUDA, Metal)
//! - [`Storage`] - Reference-counted memory storage
//! - [`Shape`] - Tensor dimensions and stride calculations
//!
//! ## Design Principles
//!
//! 1. **Zero-copy views**: Slicing and reshaping create views, not copies
//! 2. **Type safety**: Operations validate dtypes at runtime with clear errors
//! 3. **Memory efficiency**: Arc-based storage sharing with copy-on-write semantics
//! 4. **Rust idioms**: Iterator support, Display impl, From/Into conversions
//!
//! ## Example
//!
//! ```rust
//! use ferrum_core::prelude::*;
//!
//! let x = Tensor::randn([128, 256], DType::F32, Device::Cpu);
//! let y = Tensor::zeros([256, 64], DType::F32, Device::Cpu);
//! let z = x.matmul(&y).unwrap();
//! assert_eq!(z.shape(), &[128, 64]);
//! ```

pub mod autograd_ops;
pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod storage;
pub mod tensor;

pub mod prelude {
    //! Convenient re-exports for common usage.
    pub use crate::autograd_ops::{AutogradTensor, NoGradGuard, no_grad};
    pub use crate::device::Device;
    pub use crate::dtype::DType;
    pub use crate::error::{FerrumError, Result};
    pub use crate::shape::Shape;
    pub use crate::tensor::Tensor;
}

pub use prelude::*;
