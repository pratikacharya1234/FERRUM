//! Normalization layers.

use ferrum_core::{DType, Device, Result, Tensor};
use crate::module::Module;

/// Layer Normalization.
///
/// Applies layer normalization over a mini-batch of inputs as described in:
/// "Layer Normalization" (Ba et al., 2016)
///
/// y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
#[derive(Debug)]
pub struct LayerNorm {
    /// Normalized shape (last N dimensions).
    normalized_shape: Vec<usize>,
    /// Learnable scale parameter gamma.
    weight: Tensor,
    /// Learnable shift parameter beta.
    bias: Tensor,
    /// Epsilon for numerical stability.
    eps: f64,
}

impl LayerNorm {
    /// Create a new LayerNorm layer.
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> Self {
        // Initialize gamma to ones and beta to zeros
        let weight = Tensor::ones(normalized_shape.clone(), DType::F32, Device::Cpu);
        let bias = Tensor::zeros(normalized_shape.clone(), DType::F32, Device::Cpu);
        
        Self {
            normalized_shape,
            weight,
            bias,
            eps,
        }
    }

    /// Get the weight parameter.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Get the bias parameter.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}

impl Default for LayerNorm {
    fn default() -> Self {
        Self::new(vec![512], 1e-5)
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        layer_norm(input, &self.weight, &self.bias, self.eps)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

/// Batch Normalization for 2D inputs.
///
/// Applies batch normalization over a mini-batch as described in:
/// "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)
#[derive(Debug)]
pub struct BatchNorm1d {
    /// Number of features.
    num_features: usize,
    /// Learnable scale parameter.
    weight: Tensor,
    /// Learnable shift parameter.
    bias: Tensor,
    /// Running mean (for inference).
    running_mean: Tensor,
    /// Running variance (for inference).
    running_var: Tensor,
    /// Momentum for running stats.
    momentum: f64,
    /// Epsilon for numerical stability.
    eps: f64,
    /// Whether in training mode.
    training: bool,
}

impl BatchNorm1d {
    /// Create a new BatchNorm1d layer.
    pub fn new(num_features: usize) -> Self {
        Self::with_eps(num_features, 1e-5, 0.1)
    }

    /// Create with custom epsilon and momentum.
    pub fn with_eps(num_features: usize, eps: f64, momentum: f64) -> Self {
        Self {
            num_features,
            weight: Tensor::ones([num_features], DType::F32, Device::Cpu),
            bias: Tensor::zeros([num_features], DType::F32, Device::Cpu),
            running_mean: Tensor::zeros([num_features], DType::F32, Device::Cpu),
            running_var: Tensor::ones([num_features], DType::F32, Device::Cpu),
            momentum,
            eps,
            training: true,
        }
    }

    /// Set training mode.
    pub fn set_training(&mut self, mode: bool) {
        self.training = mode;
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.training {
            // During training, compute batch statistics
            let mean = input.mean()?;
            let centered = input.sub(&mean.expand(input.shape_obj().clone())?)?;
            let var = centered.pow(2.0)?.mean()?;
            
            // Normalize
            let std = var.add_scalar(self.eps)?.sqrt()?;
            let normalized = centered.div(&std.expand(input.shape_obj().clone())?)?;
            
            // Scale and shift
            let scaled = normalized.mul(&self.weight.expand(input.shape_obj().clone())?)?;
            scaled.add(&self.bias.expand(input.shape_obj().clone())?)
        } else {
            // During inference, use running statistics
            let centered = input.sub(&self.running_mean.expand(input.shape_obj().clone())?)?;
            let std = self.running_var.add_scalar(self.eps)?.sqrt()?;
            let normalized = centered.div(&std.expand(input.shape_obj().clone())?)?;
            
            let scaled = normalized.mul(&self.weight.expand(input.shape_obj().clone())?)?;
            scaled.add(&self.bias.expand(input.shape_obj().clone())?)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "BatchNorm1d"
    }
}

/// Dropout layer for regularization.
///
/// During training, randomly zeroes elements with probability p.
#[derive(Debug, Clone)]
pub struct Dropout {
    /// Dropout probability.
    p: f64,
    /// Whether in training mode.
    training: bool,
}

impl Dropout {
    /// Create a new Dropout layer.
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p < 1.0, "Dropout probability must be in [0, 1)");
        Self { p, training: true }
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.training && self.p > 0.0 {
            // During training, scale by (1 - p) to approximate dropout
            // TODO: Implement proper dropout with random mask when Tensor::rand is available
            input.mul_scalar(1.0 - self.p)
        } else {
            // During inference, return input unchanged
            Ok(input.clone())
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Dropout"
    }
}

// ============================================================================
// Functional API
// ============================================================================

/// Apply layer normalization.
pub fn layer_norm(input: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    // Compute mean and variance over last dimension(s)
    let mean = input.mean()?;
    let centered = input.sub(&mean.expand(input.shape_obj().clone())?)?;
    let var = centered.pow(2.0)?.mean()?;
    
    // Normalize
    let std = var.add_scalar(eps)?.sqrt()?;
    let normalized = centered.div(&std.expand(input.shape_obj().clone())?)?;
    
    // Scale and shift
    let scaled = normalized.mul(&weight.expand(input.shape_obj().clone())?)?;
    scaled.add(&bias.expand(input.shape_obj().clone())?)
}

/// Apply batch normalization.
pub fn batch_norm(
    input: &Tensor,
    running_mean: Option<&Tensor>,
    running_var: Option<&Tensor>,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    training: bool,
    eps: f64,
) -> Result<Tensor> {
    let (mean, var) = if training {
        // Compute batch statistics
        let mean = input.mean()?;
        let centered = input.sub(&mean.expand(input.shape_obj().clone())?)?;
        let var = centered.pow(2.0)?.mean()?;
        (mean, var)
    } else {
        // Use running statistics
        let mean = running_mean
            .cloned()
            .unwrap_or_else(|| Tensor::zeros(input.shape(), input.dtype(), input.device()));
        let var = running_var
            .cloned()
            .unwrap_or_else(|| Tensor::ones(input.shape(), input.dtype(), input.device()));
        (mean, var)
    };

    // Normalize
    let centered = input.sub(&mean.expand(input.shape_obj().clone())?)?;
    let std = var.add_scalar(eps)?.sqrt()?;
    let normalized = centered.div(&std.expand(input.shape_obj().clone())?)?;

    // Scale and shift
    match (weight, bias) {
        (Some(w), Some(b)) => {
            let scaled = normalized.mul(&w.expand(input.shape_obj().clone())?)?;
            scaled.add(&b.expand(input.shape_obj().clone())?)
        }
        (Some(w), None) => normalized.mul(&w.expand(input.shape_obj().clone())?),
        (None, Some(b)) => normalized.add(&b.expand(input.shape_obj().clone())?),
        (None, None) => Ok(normalized),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(vec![4], 1e-5);
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4], Device::Cpu).unwrap();
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4]);
    }

    #[test]
    fn test_dropout() {
        let mut dropout = Dropout::new(0.5);
        let input = Tensor::ones([4], DType::F32, Device::Cpu);
        
        // Training mode
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4]);
        
        // Eval mode
        dropout.eval();
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4]);
    }
}
