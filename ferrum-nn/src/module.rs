//! Module trait for neural network layers.
//!
//! The [`Module`] trait is the base abstraction for all neural network components.

use ferrum_core::{Result, Tensor};

/// Base trait for neural network modules.
///
/// All layers and models implement this trait, enabling:
/// - Forward pass computation
/// - Parameter enumeration for optimizers
/// - Training/evaluation mode switching
pub trait Module: Send + Sync {
    /// Perform forward pass.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get all learnable parameters.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }

    /// Set training mode.
    fn train(&mut self) {
        // Default: no-op
    }

    /// Set evaluation mode.
    fn eval(&mut self) {
        // Default: no-op
    }

    /// Check if in training mode.
    fn is_training(&self) -> bool {
        true
    }

    /// Get module name for debugging.
    fn name(&self) -> &str {
        "Module"
    }

    /// Count total number of parameters.
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}
