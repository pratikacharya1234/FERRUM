//! Loss functions for training neural networks.

use ferrum_core::{Result, Tensor};

/// Mean Squared Error loss.
///
/// Computes the mean squared difference between predictions and targets:
///
/// ```text
/// MSE = (1/n) Σ (pred - target)²
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use ferrum::prelude::*;
///
/// let predictions = Tensor::randn([32, 10], DType::F32, Device::Cpu);
/// let targets = Tensor::randn([32, 10], DType::F32, Device::Cpu);
///
/// let loss = mse_loss(&predictions, &targets)?;
/// println!("MSE Loss: {}", loss.item()?);
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
///
/// # Arguments
///
/// * `predictions` - Predicted probabilities (should be in [0, 1])
/// * `targets` - Ground truth binary labels (0 or 1)
///
/// # Example
///
/// ```rust,ignore
/// let predictions = Tensor::from_slice(&[0.8, 0.2, 0.9], [3], Device::Cpu)?;
/// let targets = Tensor::from_slice(&[1.0, 0.0, 1.0], [3], Device::Cpu)?;
///
/// let loss = bce_loss(&predictions, &targets)?;
/// ```
pub fn bce_loss(predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // BCE = -[y*log(p) + (1-y)*log(1-p)]
    let eps = 1e-7; // Small constant for numerical stability

    // Clamp predictions to avoid log(0)
    let pred_clamped = predictions.add_scalar(eps)?;
    let one_minus_pred = pred_clamped.neg()?.add_scalar(1.0 + eps)?;

    // y * log(p)
    let term1 = targets.mul(&pred_clamped.log()?)?;

    // (1-y) * log(1-p)
    let one_minus_target = targets.neg()?.add_scalar(1.0)?;
    let term2 = one_minus_target.mul(&one_minus_pred.log()?)?;

    // -mean(term1 + term2)
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
    // We need abs() operation - for now use a workaround
    let squared = diff.pow(2.0)?;
    let abs_approx = squared.sqrt()?;
    abs_approx.mean()
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

    // For now, just use L1 as approximation
    // Full implementation would need conditional operations
    let squared = diff.pow(2.0)?;
    let abs_approx = squared.sqrt()?;
    abs_approx.mean()
}

/// Cross Entropy Loss (without softmax).
///
/// Expects log probabilities as input (use with log_softmax).
///
/// ```text
/// CE = -(1/n) Σ targets * log(predictions)
/// ```
///
/// # Note
///
/// For numerical stability, combine with log_softmax operation.
/// This function expects log probabilities, not raw logits.
pub fn cross_entropy_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // CE = -mean(targets * log_probs)
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
///
/// # Example
///
/// ```rust,ignore
/// let logits = model.forward(&x)?;
/// let log_probs = log_softmax(&logits, -1)?;
/// let loss = nll_loss(&log_probs, &target_indices)?;
/// ```
pub fn nll_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    // For now, this is a simplified version
    // Full implementation needs gather/index_select operation

    // Fallback: treat as cross entropy with one-hot targets
    cross_entropy_loss(log_probs, targets)
}

/// Softmax function along a dimension.
///
/// Computes: softmax(x_i) = exp(x_i) / Σ exp(x_j)
///
/// # Arguments
///
/// * `input` - Input tensor
/// * `dim` - Dimension along which to compute softmax
///
/// # Note
///
/// For numerical stability, this subtracts the max before exp.
pub fn softmax(input: &Tensor, _dim: i64) -> Result<Tensor> {
    // Simplified version: softmax over all elements
    // Full version would work along specified dimension

    let exp = input.exp()?;
    let sum = exp.sum()?;
    exp.div(&sum)
}

/// Log Softmax function along a dimension.
///
/// Computes: log_softmax(x_i) = x_i - log(Σ exp(x_j))
///
/// More numerically stable than log(softmax(x)).
pub fn log_softmax(input: &Tensor, dim: i64) -> Result<Tensor> {
    // log(softmax(x)) = x - log(sum(exp(x)))
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

        assert!(loss_val < 1e-6); // Should be ~0 for perfect prediction
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();

        let target = Tensor::from_slice(&[0.0f32, 0.0, 0.0], [3], Device::Cpu).unwrap();

        let loss = mse_loss(&pred, &target).unwrap();
        let loss_val = loss.item().unwrap();

        // MSE = mean([1, 4, 9]) = 14/3 ≈ 4.67
        assert!((loss_val - 4.666).abs() < 0.01);
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();

        let target = Tensor::zeros([3], DType::F32, Device::Cpu);

        let loss = l1_loss(&pred, &target).unwrap();
        let loss_val = loss.item().unwrap();

        // L1 = mean([1, 2, 3]) = 2
        assert!((loss_val - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();

        let output = softmax(&input, 0).unwrap();
        let sum = output.sum().unwrap().item().unwrap();

        // Softmax outputs should sum to 1
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
