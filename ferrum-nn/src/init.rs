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
    if shape.len() < 2 {
        return Tensor::randn(shape.to_vec(), dtype, device).mul_scalar(gain).unwrap();
    }
    
    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();
    let flat_shape = [rows, cols];
    
    // Generate random normal matrix
    let mat = Tensor::randn(flat_shape, dtype, device);
    let mat_data = mat.to_vec::<f32>().unwrap();
    
    // Simple QR via Gram-Schmidt (sufficient for init)
    let mut q = vec![0.0f32; rows * cols];
    
    for j in 0..cols {
        // Copy column j
        for i in 0..rows {
            q[i * cols + j] = mat_data[i * cols + j];
        }
        // Subtract projections
        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..rows {
                dot += q[i * cols + k] * q[i * cols + j];
            }
            for i in 0..rows {
                q[i * cols + j] -= dot * q[i * cols + k];
            }
        }
        // Normalize
        let mut norm = 0.0f32;
        for i in 0..rows {
            norm += q[i * cols + j] * q[i * cols + j];
        }
        norm = norm.sqrt().max(1e-8);
        for i in 0..rows {
            q[i * cols + j] /= norm;
        }
    }
    
    // Reshape to original shape
    let q_tensor = Tensor::from_slice(&q, flat_shape, device).unwrap();
    let mut result = q_tensor;
    if shape.len() > 2 {
        let mut full_shape = vec![rows];
        full_shape.extend_from_slice(&shape[1..]);
        result = result.reshape(full_shape).unwrap();
    }
    result.mul_scalar(gain).unwrap()
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
