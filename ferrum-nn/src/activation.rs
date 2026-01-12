//! Activation function layers.

use ferrum_core::{Result, Tensor};

use crate::module::Module;

/// ReLU activation: max(0, x)
#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.relu()
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.sigmoid()
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Tanh activation
#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Self
    }
}

impl Module for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        input.tanh()
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Leaky ReLU: max(negative_slope * x, x)
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    negative_slope: f64,
}

impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self {
            negative_slope: 0.01,
        }
    }
}

impl Module for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: Implement proper leaky relu kernel
        // For now, use a simple approximation
        let positive = input.relu()?;
        let negative = input.neg()?.relu()?.mul_scalar(-self.negative_slope)?;
        positive.add(&negative)
    }

    fn name(&self) -> &str {
        "LeakyReLU"
    }
}

/// Softmax activation (applied along last dimension)
#[derive(Debug, Clone)]
pub struct Softmax {
    dim: i64,
}

impl Softmax {
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self { dim: -1 }
    }
}

impl Module for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        softmax(input, self.dim)
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}

/// Log Softmax activation
#[derive(Debug, Clone)]
pub struct LogSoftmax {
    dim: i64,
}

impl LogSoftmax {
    pub fn new(dim: i64) -> Self {
        Self { dim }
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self { dim: -1 }
    }
}

impl Module for LogSoftmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        log_softmax(input, self.dim)
    }

    fn name(&self) -> &str {
        "LogSoftmax"
    }
}

/// GELU activation: x * Φ(x) where Φ is the standard Gaussian CDF
#[derive(Debug, Clone, Default)]
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        gelu(input)
    }

    fn name(&self) -> &str {
        "GELU"
    }
}

/// SiLU (Swish) activation: x * sigmoid(x)
#[derive(Debug, Clone, Default)]
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        Self
    }
}

impl Module for SiLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        silu(input)
    }

    fn name(&self) -> &str {
        "SiLU"
    }
}

/// ELU activation: x if x > 0 else alpha * (exp(x) - 1)
#[derive(Debug, Clone)]
pub struct ELU {
    alpha: f64,
}

impl ELU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

impl Module for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // ELU: x if x > 0 else alpha * (exp(x) - 1)
        let positive = input.relu()?;
        let negative_mask = input.neg()?.relu()?.neg()?; // Gets negative values as negative
        let exp_minus_one = negative_mask.exp()?.add_scalar(-1.0)?;
        positive.add(&exp_minus_one.mul_scalar(self.alpha)?)
    }

    fn name(&self) -> &str {
        "ELU"
    }
}

// ============================================================================
// Functional API
// ============================================================================

/// Softmax function.
/// Note: For better numerical stability, consider subtracting max first.
pub fn softmax(input: &Tensor, _dim: i64) -> Result<Tensor> {
    // Simple softmax without max subtraction
    // TODO: Add max subtraction for numerical stability when max() is implemented
    let exp_x = input.exp()?;
    let sum = exp_x.sum()?;
    exp_x.div(&sum.expand(input.shape_obj().clone())?)
}

/// Log softmax function.
pub fn log_softmax(input: &Tensor, _dim: i64) -> Result<Tensor> {
    // log_softmax(x) = x - log(sum(exp(x)))
    // Simple implementation
    let exp_x = input.exp()?;
    let sum_exp = exp_x.sum()?;
    let log_sum_exp = sum_exp.log()?;
    input.sub(&log_sum_exp.expand(input.shape_obj().clone())?)
}

/// GELU activation function.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let sqrt_2_over_pi = 0.7978845608028654; // sqrt(2/π)
    let x_cubed = input.pow(3.0)?;
    let inner = input.add(&x_cubed.mul_scalar(0.044715)?)?;
    let tanh_val = inner.mul_scalar(sqrt_2_over_pi)?.tanh()?;
    let one_plus_tanh = tanh_val.add_scalar(1.0)?;
    input.mul(&one_plus_tanh)?.mul_scalar(0.5)
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(input: &Tensor) -> Result<Tensor> {
    let sigmoid = input.sigmoid()?;
    input.mul(&sigmoid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{DType, Device};

    #[test]
    fn test_relu() {
        let relu = ReLU::new();
        let input = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], [4], Device::Cpu).unwrap();
        let output = relu.forward(&input).unwrap();
        let data = output.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::zeros([1], DType::F32, Device::Cpu);
        let output = sigmoid.forward(&input).unwrap();
        let val = output.item().unwrap();
        assert!((val - 0.5).abs() < 1e-6);
    }
}
