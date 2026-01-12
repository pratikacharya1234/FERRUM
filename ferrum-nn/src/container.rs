//! Container modules for composing layers.

use ferrum_core::{Result, Tensor};

use crate::module::Module;

/// Sequential container: applies layers in order.
///
/// # Example
///
/// ```rust,ignore
/// let model = Sequential::new()
///     .add(Linear::new(784, 256))
///     .add(ReLU::new())
///     .add(Linear::new(256, 10));
/// ```
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
    training: bool,
}

impl Sequential {
    /// Create an empty sequential container.
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            training: true,
        }
    }

    /// Add a layer to the sequence.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Add a boxed layer.
    pub fn add_boxed(mut self, layer: Box<dyn Module>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Get number of layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &str {
        "Sequential"
    }
}

impl std::fmt::Debug for Sequential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential(")?;
        for (i, layer) in self.layers.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", layer.name())?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ReLU};
    use ferrum_core::{DType, Device};

    #[test]
    fn test_sequential() {
        let model = Sequential::new()
            .add(Linear::new(10, 5))
            .add(ReLU::new())
            .add(Linear::new(5, 2));

        assert_eq!(model.len(), 3);

        let input = Tensor::randn([2, 10], DType::F32, Device::Cpu);
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
    }
}
