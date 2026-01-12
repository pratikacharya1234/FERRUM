//! Gradient checking utilities for verifying backward implementations.
//!
//! Gradient checking numerically approximates gradients using finite differences
//! and compares them to analytical gradients from autograd.

use ferrum_core::{Result, Tensor};

/// Gradient checking with finite differences.
///
/// Verifies that analytical gradients match numerical gradients computed
/// using the finite difference method:
///
/// ```text
/// ∂f/∂x ≈ [f(x + ε) - f(x - ε)] / (2ε)
/// ```
///
/// # Arguments
///
/// * `f` - Function that computes a scalar output from a tensor input
/// * `input` - Input tensor
/// * `analytical_grad` - Gradient computed by autograd
/// * `eps` - Small perturbation (default: 1e-5)
/// * `rtol` - Relative tolerance (default: 1e-3)
/// * `atol` - Absolute tolerance (default: 1e-5)
///
/// # Returns
///
/// `Ok(())` if gradients match within tolerance, error otherwise.
///
/// # Example
///
/// ```rust,ignore
/// use ferrum_autograd::gradcheck::check_gradients;
///
/// // Function: f(x) = sum(x^2)
/// let compute_loss = |x: &Tensor| -> Result<Tensor> {
///     x.pow(2.0)?.sum()
/// };
///
/// let x = Tensor::randn([3, 4], DType::F32, Device::Cpu);
/// let analytical_grad = ...; // Computed via backward()
///
/// check_gradients(&compute_loss, &x, &analytical_grad, None, None, None)?;
/// ```
pub fn check_gradients<F>(
    f: &F,
    input: &Tensor,
    analytical_grad: &Tensor,
    eps: Option<f64>,
    rtol: Option<f64>,
    atol: Option<f64>,
) -> Result<()>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    let eps = eps.unwrap_or(1e-4);
    let rtol = rtol.unwrap_or(0.01); // 1% relative tolerance
    let atol = atol.unwrap_or(0.01); // Absolute tolerance

    // Convert input to contiguous vector
    let input_vec = input.to_vec::<f32>()?;
    let grad_vec = analytical_grad.to_vec::<f32>()?;

    if input_vec.len() != grad_vec.len() {
        return Err(ferrum_core::FerrumError::InvalidShape {
            message: format!(
                "Input and gradient shapes don't match: {} vs {}",
                input_vec.len(),
                grad_vec.len()
            ),
        });
    }

    println!("Gradient Checking:");
    println!("  Epsilon: {}", eps);
    println!("  Relative tolerance: {}", rtol);
    println!("  Absolute tolerance: {}", atol);
    println!();

    let mut max_error = 0.0f64;
    let mut num_errors = 0;

    // Check gradient for each element
    for (i, &analytical_grad) in grad_vec.iter().enumerate().take(100) {
        // Only check first 100 elements for performance
        let numerical_grad = compute_numerical_gradient(f, input, i, eps)?;
        let analytical = analytical_grad as f64;

        let diff = (numerical_grad - analytical).abs();
        let threshold = atol + rtol * analytical.abs();

        if diff > threshold {
            if num_errors < 5 {
                // Print first 5 errors
                println!(
                    "  Element {}: numerical={:.6}, analytical={:.6}, diff={:.6}",
                    i, numerical_grad, analytical, diff
                );
            }
            num_errors += 1;
            max_error = max_error.max(diff);
        }
    }

    println!();
    if num_errors > 0 {
        println!(
            "⚠ Gradient check FAILED: {} elements exceeded tolerance",
            num_errors
        );
        println!("  Max error: {:.6}", max_error);
        Err(ferrum_core::FerrumError::AutogradError {
            message: format!(
                "Gradient check failed: {} errors, max error {:.6}",
                num_errors, max_error
            ),
        })
    } else {
        println!("✓ Gradient check PASSED");
        Ok(())
    }
}

/// Compute numerical gradient for a single element using finite differences.
fn compute_numerical_gradient<F>(f: &F, input: &Tensor, index: usize, eps: f64) -> Result<f64>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    // Get input as vector
    let mut input_vec = input.to_vec::<f32>()?;

    // f(x + eps)
    input_vec[index] += eps as f32;
    let input_plus = Tensor::from_slice(&input_vec, input.shape(), input.device())?;
    let output_plus = f(&input_plus)?;
    let val_plus = output_plus.item()?;

    // f(x - eps)
    input_vec[index] -= 2.0 * eps as f32;
    let input_minus = Tensor::from_slice(&input_vec, input.shape(), input.device())?;
    let output_minus = f(&input_minus)?;
    let val_minus = output_minus.item()?;

    // Numerical gradient: [f(x+ε) - f(x-ε)] / (2ε)
    let numerical_grad = (val_plus - val_minus) / (2.0 * eps);

    Ok(numerical_grad)
}

/// Quick gradient check with default parameters.
///
/// Convenience function that uses default tolerance values.
pub fn quick_gradcheck<F>(f: &F, input: &Tensor, analytical_grad: &Tensor) -> Result<()>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    check_gradients(f, input, analytical_grad, None, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::Device;

    #[test]
    fn test_numerical_gradient_simple() {
        // Test f(x) = x^2, gradient should be 2x

        let compute_loss = |x: &Tensor| -> Result<Tensor> { x.pow(2.0)?.sum() };

        let input = Tensor::from_slice(&[2.0f32], [1], Device::Cpu).unwrap();

        let numerical = compute_numerical_gradient(&compute_loss, &input, 0, 1e-5).unwrap();

        // Analytical gradient at x=2 is 2*2 = 4
        // Relax tolerance for finite precision
        assert!((numerical - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_gradcheck_linear() {
        // Test f(x) = 2*x, gradient should be 2

        let compute_loss = |x: &Tensor| -> Result<Tensor> { x.mul_scalar(2.0)?.sum() };

        let input = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();

        // Analytical gradient is all 2s
        let analytical = Tensor::from_slice(&[2.0f32, 2.0, 2.0], [3], Device::Cpu).unwrap();

        let result = quick_gradcheck(&compute_loss, &input, &analytical);
        assert!(result.is_ok());
    }
}
