//! Data transforms for preprocessing.

use ferrum_core::{Result, Tensor};

/// A transform that can be applied to data.
pub trait Transform: Send + Sync {
    /// Apply the transform to a tensor.
    fn apply(&self, input: &Tensor) -> Result<Tensor>;
}

/// Compose multiple transforms.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    /// Create a new compose transform.
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        let mut result = input.clone();
        for t in &self.transforms {
            result = t.apply(&result)?;
        }
        Ok(result)
    }
}

/// Normalize a tensor with mean and std.
pub struct Normalize {
    mean: f64,
    std: f64,
}

impl Normalize {
    /// Create a new normalize transform.
    pub fn new(mean: f64, std: f64) -> Self {
        Self { mean, std }
    }

    /// Standard normalization (mean=0, std=1).
    pub fn standard() -> Self {
        Self::new(0.0, 1.0)
    }

    /// ImageNet normalization for RGB images.
    pub fn imagenet() -> Self {
        // Approximate ImageNet normalization
        Self::new(0.485, 0.229)
    }
}

impl Transform for Normalize {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        // (x - mean) / std
        input.add_scalar(-self.mean)?.mul_scalar(1.0 / self.std)
    }
}

/// Scale tensor values to a range.
pub struct Scale {
    min_val: f64,
    max_val: f64,
}

impl Scale {
    /// Create a new scale transform.
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self { min_val, max_val }
    }

    /// Scale to [0, 1].
    pub fn unit() -> Self {
        Self::new(0.0, 1.0)
    }

    /// Scale to [-1, 1].
    pub fn symmetric() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl Transform for Scale {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        // Assume input is in [0, 1] and scale to [min_val, max_val]
        let range = self.max_val - self.min_val;
        input.mul_scalar(range)?.add_scalar(self.min_val)
    }
}

/// Add Gaussian noise to tensor.
pub struct GaussianNoise {
    std: f64,
}

impl GaussianNoise {
    /// Create a new Gaussian noise transform.
    pub fn new(std: f64) -> Self {
        Self { std }
    }
}

impl Transform for GaussianNoise {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: Implement proper noise addition when randn is available
        // For now, return input unchanged
        Ok(input.clone())
    }
}

/// Identity transform (no-op).
pub struct Identity;

impl Transform for Identity {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

/// Convert to one-hot encoding.
pub struct ToOneHot {
    num_classes: usize,
}

impl ToOneHot {
    /// Create a new one-hot encoding transform.
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Transform for ToOneHot {
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        // TODO: Implement one-hot encoding
        // For now, return input unchanged
        Ok(input.clone())
    }
}

/// Lambda transform - apply a custom function.
pub struct Lambda<F> {
    func: F,
}

impl<F> Lambda<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    /// Create a new lambda transform.
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Transform for Lambda<F>
where
    F: Fn(&Tensor) -> Result<Tensor> + Send + Sync,
{
    fn apply(&self, input: &Tensor) -> Result<Tensor> {
        (self.func)(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{DType, Device};

    #[test]
    fn test_normalize() {
        let norm = Normalize::new(0.5, 0.5);
        let input = Tensor::ones([4], DType::F32, Device::Cpu);
        let output = norm.apply(&input).unwrap();
        
        // (1 - 0.5) / 0.5 = 1.0
        let data = output.to_vec::<f32>().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_compose() {
        let t1 = Box::new(Normalize::new(0.0, 2.0)) as Box<dyn Transform>;
        let t2 = Box::new(Scale::new(-1.0, 1.0)) as Box<dyn Transform>;
        let compose = Compose::new(vec![t1, t2]);
        
        let input = Tensor::ones([4], DType::F32, Device::Cpu);
        let output = compose.apply(&input).unwrap();
        assert_eq!(output.shape(), &[4]);
    }

    #[test]
    fn test_identity() {
        let id = Identity;
        let input = Tensor::ones([4], DType::F32, Device::Cpu);
        let output = id.apply(&input).unwrap();
        assert_eq!(output.to_vec::<f32>().unwrap(), input.to_vec::<f32>().unwrap());
    }
}
