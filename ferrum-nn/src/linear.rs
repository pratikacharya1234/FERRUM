//! Linear (fully connected) layer.

use ferrum_core::{DType, Device, Result, Tensor};

use crate::init;
use crate::module::Module;

/// Linear layer: y = xW + b
///
/// # Example
///
/// ```rust,ignore
/// let linear = Linear::new(784, 256);
/// let output = linear.forward(&input)?;  // [batch, 784] -> [batch, 256]
/// ```
pub struct Linear {
    /// Weight matrix [in_features, out_features].
    ///
    /// Stored input-major so forward is `input.matmul(&weight)` directly.
    /// Autograd tracks gradients by tensor id, and transposed views share
    /// their base tensor's id, so a `weight.t()` in forward would deposit a
    /// transposed gradient on `weight` — this layout avoids that entirely.
    weight: Tensor,
    /// Optional bias [out_features]
    bias: Option<Tensor>,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Training mode flag
    training: bool,
}

impl Linear {
    /// Create a new linear layer with default initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Create a linear layer with optional bias.
    pub fn with_bias(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Kaiming uniform initialization (He initialization)
        let weight = init::kaiming_uniform(
            &[in_features, out_features],
            in_features,
            DType::F32,
            Device::Cpu,
        )
        .with_requires_grad(true);

        let bias = if bias {
            // Uniform initialization for bias
            let bound = 1.0 / (in_features as f64).sqrt();
            Some(
                Tensor::uniform([out_features], -bound, bound, DType::F32, Device::Cpu)
                    .with_requires_grad(true),
            )
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
            training: true,
        }
    }

    /// Move the layer's parameters to a device.
    ///
    /// Call before handing `parameters()` to an optimizer — the optimizer
    /// shares storage with the tensors it is given, so parameters moved
    /// afterwards would leave the optimizer updating the old copies.
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        self.weight = self.weight.to_device(device)?;
        if let Some(bias) = self.bias.take() {
            self.bias = Some(bias.to_device(device)?);
        }
        Ok(())
    }

    /// Get input features.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Get output features.
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input: [batch, in_features]
        // weight: [in_features, out_features]
        // output = input @ weight + bias
        let output = input.matmul(&self.weight)?;

        if let Some(ref bias) = self.bias {
            output.add(bias)
        } else {
            Ok(output)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
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
        "Linear"
    }
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_creation() {
        let linear = Linear::new(10, 5);
        assert_eq!(linear.in_features(), 10);
        assert_eq!(linear.out_features(), 5);
        assert_eq!(linear.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_linear_forward() {
        let linear = Linear::new(10, 5);
        let input = Tensor::randn([2, 10], DType::F32, Device::Cpu);
        let output = linear.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5]);
    }

    #[test]
    fn test_linear_no_bias() {
        let linear = Linear::with_bias(10, 5, false);
        assert_eq!(linear.parameters().len(), 1); // weight only
    }
}
