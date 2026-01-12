//! Optimizer trait.

use ferrum_core::{Result, Tensor};

/// Base trait for all optimizers.
pub trait Optimizer {
    /// Perform a single optimization step.
    fn step(&mut self) -> Result<()>;

    /// Zero all parameter gradients.
    fn zero_grad(&mut self);

    /// Get current learning rate.
    fn learning_rate(&self) -> f64;

    /// Set learning rate.
    fn set_learning_rate(&mut self, lr: f64);

    /// Get parameter groups (for per-layer learning rates).
    fn param_groups(&self) -> &[Vec<Tensor>];
}
