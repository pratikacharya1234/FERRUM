//! # FERRUM Data
//!
//! Data loading utilities for deep learning, inspired by PyTorch DataLoader.
//!
//! ## Features
//!
//! - `Dataset` trait for custom datasets
//! - `DataLoader` with batching, shuffling, and parallel loading
//! - Common transforms (normalize, augmentation)
//! - Samplers (random, sequential, weighted)
//!
//! ## Example
//!
//! ```rust,ignore
//! use ferrum_data::{Dataset, DataLoader, RandomSampler};
//!
//! struct MyDataset { /* ... */ }
//!
//! impl Dataset for MyDataset {
//!     fn len(&self) -> usize { 1000 }
//!     fn get(&self, index: usize) -> (Tensor, Tensor) { /* ... */ }
//! }
//!
//! let dataset = MyDataset::new();
//! let loader = DataLoader::new(dataset)
//!     .batch_size(32)
//!     .shuffle(true)
//!     .num_workers(4);
//!
//! for (inputs, targets) in loader {
//!     // Training loop
//! }
//! ```

pub mod dataset;
pub mod dataloader;
pub mod sampler;
pub mod transform;

pub use dataset::{Dataset, TensorDataset, MapDataset, ConcatDataset, Subset, train_test_split};
pub use dataloader::DataLoader;
pub use sampler::{Sampler, RandomSampler, SequentialSampler, BatchSampler, DistributedSampler, SubsetRandomSampler, WeightedRandomSampler};
pub use transform::{Transform, Compose, Normalize};

/// Prelude module with common imports.
pub mod prelude {
    pub use crate::dataset::*;
    pub use crate::dataloader::*;
    pub use crate::sampler::*;
    pub use crate::transform::*;
}
