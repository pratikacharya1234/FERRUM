//! Loss functions for training neural networks.

use ferrum_core::{Result, Tensor};

/// Mean Squared Error loss.
///
/// Computes the mean squared difference between predictions and targets:
///
/// ```text
/// MSE = (1/n) Σ (pred - target)²
/// ```
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = predictions.sub(targets)?;
    let squared = diff.pow(2.0)?;
    squared.mean()
}

/// Binary Cross Entropy loss.
///
/// Computes the cross entropy between binary predictions and targets:
///
/// ```text
/// BCE = -(1/n) Σ [y*log(p) + (1-y)*log(1-p)]
/// ```
pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let eps = 1e-7;
    let pred_clamped = predictions.add_scalar(eps)?;
    let one_minus_pred = pred_clamped.neg()?.add_scalar(1.0 + eps)?;
    let term1 = targets.mul(&pred_clamped.log()?)?;
    let one_minus_target = targets.neg()?.add_scalar(1.0)?;
    let term2 = one_minus_target.mul(&one_minus_pred.log()?)?;
    let loss = term1.add(&term2)?.mean()?.neg()?;
    Ok(loss)
}

/// L1 Loss (Mean Absolute Error).
///
/// Computes the mean absolute difference:
///
/// ```text
/// L1 = (1/n) Σ |pred - target|
/// ```
pub fn l1_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = predictions.sub(targets)?;
    let data = diff.to_vec::<f32>()?;
    let abs_data: Vec<f32> = data.iter().map(|&x| x.abs()).collect();
    let abs_tensor = Tensor::from_slice(&abs_data, diff.shape().to_vec(), diff.device())?;
    abs_tensor.mean()
}

/// Smooth L1 Loss (Huber loss with delta=1).
///
/// Combines properties of L1 and L2 loss:
/// - Uses L2 for small errors (|x| < 1)
/// - Uses L1 for large errors (|x| >= 1)
///
/// ```text
/// SmoothL1(x) = 0.5*x² if |x| < 1
///             = |x| - 0.5 otherwise
/// ```
pub fn smooth_l1_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let diff = predictions.sub(targets)?;
    let data = diff.to_vec::<f32>()?;
    let smooth_data: Vec<f32> = data.iter().map(|&x| {
        let abs_x = x.abs();
        if abs_x < 1.0 {
            0.5 * x * x
        } else {
            abs_x - 0.5
        }
    }).collect();
    let smooth_tensor = Tensor::from_slice(&smooth_data, diff.shape().to_vec(), diff.device())?;
    smooth_tensor.mean()
}

/// Cross Entropy Loss (without softmax).
///
/// Expects log probabilities as input (use with log_softmax).
///
/// ```text
/// CE = -(1/n) Σ targets * log(predictions)
/// ```
pub fn cross_entropy_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let product = targets.mul(log_probs)?;
    product.mean()?.neg()
}

/// Negative Log Likelihood loss.
///
/// Commonly used with log_softmax for classification:
///
/// ```text
/// NLL = -(1/n) Σ log(p[target_class])
/// ```
///
/// # Arguments
///
/// * `log_probs` - Log probabilities from log_softmax, shape [batch, classes]
/// * `targets` - Class indices, shape [batch]
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs_data = log_probs.to_vec::<f32>()?;
    let targets_data = targets.to_vec::<i64>()?;
    let batch_size = targets_data.len();
    let num_classes = log_probs.shape()[1];
    
    let mut total_loss = 0.0f32;
    for (i, &target_idx) in targets_data.iter().enumerate() {
        let idx = target_idx as usize;
        if idx < num_classes {
            total_loss += log_probs_data[i * num_classes + idx];
        }
    }
    
    let loss = -total_loss / batch_size as f32;
    Tensor::from_slice(&[loss], [1], log_probs.device())
}

/// Softmax function along specified dimension.
///
/// Numerically stable: subtracts max before exp.
pub fn softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    let shape = input.shape();
    let ndim = shape.len() as i64;
    let dim = if dim < 0 { ndim + dim } else { dim } as usize;

    let data = input.to_vec::<f32>()?;
    let mut output = vec![0.0f32; data.len()];
    let dim_size = shape[dim];
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();

    for o in 0..outer {
        for i in 0..inner {
            let mut max_val = f32::NEG_INFINITY;
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                max_val = max_val.max(data[idx]);
            }
            let mut sum = 0.0f32;
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                let v = (data[idx] - max_val).exp();
                output[idx] = v;
                sum += v;
            }
            for d in 0..dim_size {
                let idx = o * dim_size * inner + d * inner + i;
                output[idx] /= sum;
            }
        }
    }
    Tensor::from_slice(&output, shape.to_vec(), input.device())
}

/// Log Softmax function along a dimension.
///
/// Computes: log_softmax(x_i) = x_i - log(Σ exp(x_j))
pub fn log_softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    let sm = softmax(input, dim)?;
    sm.log()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{DType, Device};

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        let target = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val = loss.item().unwrap();
        assert!(loss_val < 1e-6);
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        let target = Tensor::zeros([3], DType::F32, Device::Cpu);
        let loss = l1_loss(&pred, &target).unwrap();
        let loss_val = loss.item().unwrap();
        assert!((loss_val - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 0.5], [3], Device::Cpu).unwrap();
        let target = Tensor::zeros([3], DType::F32, Device::Cpu);
        let loss = smooth_l1_loss(&pred, &target).unwrap();
        let loss_val = loss.item().unwrap();
        // smooth_l1(1.0) = 0.5, smooth_l1(2.0) = 1.5, smooth_l1(0.5) = 0.125
        let expected = (0.5 + 1.5 + 0.125) / 3.0;
        assert!((loss_val - expected).abs() < 0.01);
    }

    #[test]
    fn test_nll_loss() {
        let log_probs = Tensor::from_slice(&[-0.5f32, -1.0, -2.0, -1.0, -0.5, -2.0], [2, 3], Device::Cpu).unwrap();
        let targets = Tensor::from_slice(&[0i64, 1], [2], Device::Cpu).unwrap();
        let loss = nll_loss(&log_probs, &targets).unwrap();
        let loss_val = loss.item().unwrap();
        // -(-0.5 + -0.5) / 2 = 0.5
        assert!((loss_val - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        let output = softmax(&input, 0).unwrap();
        let sum = output.sum().unwrap().item().unwrap();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
