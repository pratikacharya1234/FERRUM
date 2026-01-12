//! # FERRUM
//!
//! **Production-grade deep learning framework in pure Rust.**
//!
//! ```text
//!    ███████╗███████╗██████╗ ██████╗ ██╗   ██╗███╗   ███╗
//!    ██╔════╝██╔════╝██╔══██╗██╔══██╗██║   ██║████╗ ████║
//!    █████╗  █████╗  ██████╔╝██████╔╝██║   ██║██╔████╔██║
//!    ██╔══╝  ██╔══╝  ██╔══██╗██╔══██╗██║   ██║██║╚██╔╝██║
//!    ██║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║
//!    ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
//!                  Pure Rust Deep Learning
//! ```
//!
//! ## Features
//!
//! - **Pure Rust**: No Python dependency, no C++ build complexity
//! - **High Performance**: SIMD-optimized operations, parallel execution
//! - **PyTorch-like API**: Familiar interface for ML practitioners
//! - **Modular Design**: Use only what you need
//! - **Memory Safe**: Leverage Rust's ownership model
//! - **Dynamic Graphs**: Build-by-run computation graphs
//! - **GPU Support**: CUDA acceleration (with `cuda` feature)
//! - **Distributed Training**: Multi-GPU and multi-node training
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use ferrum::prelude::*;
//!
//! // Create tensors
//! let x = Tensor::randn(&[128, 784], DType::F32, Device::Cpu);
//! let y = Tensor::zeros(&[128, 10], DType::F32, Device::Cpu);
//!
//! // Build a model
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU::new())
//!     .add(Linear::new(256, 10));
//!
//! // Forward pass
//! let output = model.forward(&x)?;
//!
//! // Compute loss
//! let loss = output.sub(&y)?.pow(2.0)?.mean()?;
//! println!("Loss: {}", loss.item()?);
//! ```
//!
//! ## Module Overview
//!
//! | Module | Description |
//! |--------|-------------|
//! | `ferrum-core` | Tensor primitives, shapes, dtypes |
//! | `ferrum-autograd` | Automatic differentiation |
//! | `ferrum-ops` | Optimized tensor operations |
//! | `ferrum-nn` | Neural network layers |
//! | `ferrum-optim` | Optimizers (SGD, Adam) |
//! | `ferrum-serialize` | Model save/load |
//! | `ferrum-data` | DataLoader, datasets, samplers |
//! | `ferrum-distributed` | Distributed training (DDP) |
//! | `ferrum-cuda` | CUDA GPU support |

#![doc(html_logo_url = "https://raw.githubusercontent.com/pratikacharya1234/FERRUM/main/assets/logo.svg")]
#![warn(missing_docs)]

// Re-export core functionality
pub use ferrum_autograd as autograd;
pub use ferrum_core as core;
pub use ferrum_data as data;
pub use ferrum_distributed as distributed;
pub use ferrum_nn as nn;
pub use ferrum_ops as ops;
pub use ferrum_optim as optim;
pub use ferrum_serialize as serialize;

#[cfg(feature = "cuda")]
pub use ferrum_cuda as cuda;

/// Prelude module with commonly used types.
///
/// ```rust,ignore
/// use ferrum::prelude::*;
/// ```
pub mod prelude {
    // Core types
    pub use ferrum_core::{DType, Device, FerrumError, Result, Shape, Tensor};

    // Autograd - both module and key types
    pub use ferrum_autograd::{backward, GradientTape};
    
    // Autograd operations and traits (for backward)
    pub use ferrum_core::autograd_ops::{AutogradTensor, NoGradGuard, no_grad};

    // Neural network layers
    pub use ferrum_nn::{Linear, Module, ReLU, Sequential, Sigmoid, Tanh};
    
    // Additional activations
    pub use ferrum_nn::{GELU, SiLU, Softmax, LogSoftmax, LeakyReLU, ELU};
    
    // Normalization layers
    pub use ferrum_nn::{LayerNorm, BatchNorm1d, Dropout};
    
    // Embedding layer
    pub use ferrum_nn::Embedding;

    // Loss functions
    pub use ferrum_nn::{
        bce_loss, cross_entropy_loss, l1_loss, log_softmax, mse_loss, nll_loss, smooth_l1_loss,
        softmax,
    };

    // Optimizers
    pub use ferrum_optim::{Adam, AdamConfig, Optimizer, SGDConfig, SGD};
    
    // Learning rate schedulers
    pub use ferrum_optim::{
        LRScheduler, StepLR, MultiStepLR, ExponentialLR,
        CosineAnnealingLR, CosineAnnealingWarmRestarts,
        LinearWarmupLR, OneCycleLR, ReduceLROnPlateau, PlateauMode,
    };

    // Serialization
    pub use ferrum_serialize::{load, save};
    
    // Data loading
    pub use ferrum_data::{DataLoader, Dataset, TensorDataset, Subset, train_test_split};
    pub use ferrum_data::sampler::{RandomSampler, SequentialSampler, DistributedSampler};
    
    // Distributed training
    pub use ferrum_distributed::{
        init_process_group, get_rank, get_world_size, is_main_process, barrier,
        Backend, ProcessGroup, DistributedDataParallel, ReduceOp,
    };
}

// Re-export autograd_ops for advanced usage
pub use ferrum_core::autograd_ops;

/// Print FERRUM banner.
pub fn banner() {
    println!(
        r#"
   ███████╗███████╗██████╗ ██████╗ ██╗   ██╗███╗   ███╗
   ██╔════╝██╔════╝██╔══██╗██╔══██╗██║   ██║████╗ ████║
   █████╗  █████╗  ██████╔╝██████╔╝██║   ██║██╔████╔██║
   ██╔══╝  ██╔══╝  ██╔══██╗██╔══██╗██║   ██║██║╚██╔╝██║
   ██║     ███████╗██║  ██║██║  ██║╚██████╔╝██║ ╚═╝ ██║
   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝
               Pure Rust Deep Learning v{}
"#,
        env!("CARGO_PKG_VERSION")
    );
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_basic_tensor_ops() {
        let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let b = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_simple_model() {
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLU::new())
            .add(Linear::new(5, 2));

        let input = Tensor::randn([4, 10], DType::F32, Device::Cpu);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 2]);
    }
}
