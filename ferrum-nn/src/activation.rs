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
        // LeakyReLU: x if x > 0 else negative_slope * x
        let data = input.to_vec::<f32>()?;
        let output: Vec<f32> = data.iter()
            .map(|&x| if x > 0.0 { x } else { self.negative_slope as f32 * x })
            .collect();
        Tensor::from_slice(&output, input.shape().to_vec(), input.device())
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
        let data = input.to_vec::<f32>()?;
        let output: Vec<f32> = data.iter()
            .map(|&x| if x > 0.0 { x } else { self.alpha as f32 * (x.exp() - 1.0) })
            .collect();
        Tensor::from_slice(&output, input.shape().to_vec(), input.device())
    }

    fn name(&self) -> &str {
        "ELU"
    }
}

// ============================================================================
// Functional API
// ============================================================================

/// Softmax function along specified dimension.
/// Numerically stable: subtracts max before exp.
pub fn softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    let shape = input.shape();
    let ndim = shape.len() as i64;
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;

    // For last-dim softmax (most common), use optimized path
    if dim == (ndim - 1) as usize {
        let data = input.to_vec::<f32>()?;
        let mut output = vec![0.0f32; data.len()];
        let outer: usize = shape[..dim].iter().product();
        let inner = shape[dim];

        for o in 0..outer {
            let base = o * inner;
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..inner {
                max_val = max_val.max(data[base + j]);
            }
            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for j in 0..inner {
                let v = (data[base + j] - max_val).exp();
                output[base + j] = v;
                sum += v;
            }
            // Normalize
            for j in 0..inner {
                output[base + j] /= sum;
            }
        }
        Tensor::from_slice(&output, shape.to_vec(), input.device())
    } else {
        // Generic dim softmax
        let data = input.to_vec::<f32>()?;
        let mut output = vec![0.0f32; data.len()];
        let dim_size = shape[dim];
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // Find max
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    max_val = max_val.max(data[idx]);
                }
                // Compute exp and sum
                let mut sum = 0.0f32;
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    let v = (data[idx] - max_val).exp();
                    output[idx] = v;
                    sum += v;
                }
                // Normalize
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    output[idx] /= sum;
                }
            }
        }
        Tensor::from_slice(&output, shape.to_vec(), input.device())
    }
}

/// Log softmax function along specified dimension.
pub fn log_softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    let sm = softmax(input, dim)?;
    sm.log()
}

/// GELU activation function.
pub fn gelu(input: &Tensor) -> Result<Tensor> {
    // GELU: x * Φ(x) where Φ is the standard Gaussian CDF
    // Approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let data = input.to_vec::<f32>()?;
    let sqrt_2_over_pi = 0.7978845608028654f32;
    let output: Vec<f32> = data.iter()
        .map(|&x| {
            let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        })
        .collect();
    Tensor::from_slice(&output, input.shape().to_vec(), input.device())
}

/// SiLU (Swish) activation: x * sigmoid(x)
pub fn silu(input: &Tensor) -> Result<Tensor> {
    let data = input.to_vec::<f32>()?;
    let output: Vec<f32> = data.iter()
        .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
        .collect();
    Tensor::from_slice(&output, input.shape().to_vec(), input.device())
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
