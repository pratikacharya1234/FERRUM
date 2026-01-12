//! Parameter initialization methods.

use ferrum_core::{DType, Device, Tensor};

/// Xavier/Glorot uniform initialization.
///
/// Samples from U[-bound, bound] where bound = sqrt(6 / (fan_in + fan_out))
pub fn xavier_uniform(
    shape: &[usize],
    fan_in: usize,
    fan_out: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    Tensor::uniform(shape.to_vec(), -bound, bound, dtype, device)
}

/// Xavier/Glorot normal initialization.
///
/// Samples from N(0, std) where std = sqrt(2 / (fan_in + fan_out))
pub fn xavier_normal(
    shape: &[usize],
    fan_in: usize,
    fan_out: usize,
    dtype: DType,
    device: Device,
) -> Tensor {
    let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
    Tensor::normal(shape.to_vec(), 0.0, std, dtype, device)
}

/// Kaiming/He uniform initialization (for ReLU networks).
///
/// Samples from U[-bound, bound] where bound = sqrt(6 / fan_in)
pub fn kaiming_uniform(shape: &[usize], fan_in: usize, dtype: DType, device: Device) -> Tensor {
    let bound = (6.0 / fan_in as f64).sqrt();
    Tensor::uniform(shape.to_vec(), -bound, bound, dtype, device)
}

/// Kaiming/He normal initialization (for ReLU networks).
///
/// Samples from N(0, std) where std = sqrt(2 / fan_in)
pub fn kaiming_normal(shape: &[usize], fan_in: usize, dtype: DType, device: Device) -> Tensor {
    let std = (2.0 / fan_in as f64).sqrt();
    Tensor::normal(shape.to_vec(), 0.0, std, dtype, device)
}

/// Orthogonal initialization.
///
/// Fills tensor with (semi) orthogonal matrix using QR decomposition.
pub fn orthogonal(shape: &[usize], gain: f64, dtype: DType, device: Device) -> Tensor {
    // Simplified: use normal init for now
    // TODO: Implement proper orthogonal init with QR decomposition
    let tensor = Tensor::randn(shape.to_vec(), dtype, device);
    tensor.mul_scalar(gain).unwrap()
}

/// Constant initialization.
pub fn constant(shape: &[usize], value: f64, dtype: DType, device: Device) -> Tensor {
    Tensor::full(shape.to_vec(), value, dtype, device)
}

/// Zero initialization.
pub fn zeros(shape: &[usize], dtype: DType, device: Device) -> Tensor {
    Tensor::zeros(shape.to_vec(), dtype, device)
}

/// One initialization.
pub fn ones(shape: &[usize], dtype: DType, device: Device) -> Tensor {
    Tensor::ones(shape.to_vec(), dtype, device)
}
