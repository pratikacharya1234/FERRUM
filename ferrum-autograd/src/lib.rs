//! # FERRUM Autograd
//!
//! Automatic differentiation engine using tape-based reverse-mode AD.
//!
//! This crate provides the computational graph infrastructure for training neural networks:
//!
//! - **Tape-based recording**: Operations are recorded to a gradient tape
//! - **Reverse-mode AD**: Efficient backpropagation for many-inputs-to-scalar
//! - **Dynamic graphs**: Build different graphs on each forward pass (like PyTorch)
//!
//! ## Architecture
//!
//! ```text
//! Forward Pass:
//!   x ──┬── Add ──┬── MatMul ──► loss
//!       │         │
//!       └─ y ─────┘
//!
//! Gradient Tape:
//!   [MatMul_backward] ◄── [Add_backward] ◄── seed=1.0
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use ferrum_autograd::GradientTape;
//! use ferrum_core::prelude::*;
//!
//! // Enable gradient tracking
//! let mut x = Tensor::randn([2, 3], DType::F32, Device::Cpu);
//! x.set_requires_grad(true);
//!
//! // Create tape and record operations  
//! let tape = GradientTape::new();
//! // ... operations would be recorded here
//! ```
//!
//! ## Design Principles
//!
//! 1. **PyTorch-like API**: Familiar interface for ML practitioners
//! 2. **Lazy execution**: Backward pass only computes what's needed
//! 3. **Memory efficient**: Drop intermediate values when possible
//! 4. **Thread-safe**: Tape can be shared across threads (with care)

pub mod backward;
pub mod function;
pub mod gradcheck;
pub mod graph;
pub mod tape;

pub mod prelude {
    //! Convenient re-exports.
    pub use crate::backward::backward;
    pub use crate::function::Function;
    pub use crate::gradcheck::{check_gradients, quick_gradcheck};
    pub use crate::graph::{ComputationGraph, Node, NodeId};
    pub use crate::tape::GradientTape;
}

pub use prelude::*;
