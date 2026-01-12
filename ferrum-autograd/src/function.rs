//! Differentiable function trait and common implementations.
//!
//! A `Function` defines both the forward and backward pass for an operation.
//! The backward pass computes gradients of the inputs given gradients of the outputs.

use ferrum_core::{DType, Result, Shape, Tensor};

/// Trait for differentiable functions.
///
/// Implement this trait to define custom autograd operations.
/// The backward method computes ∂L/∂input given ∂L/∂output.
///
/// ## Example
///
/// ```rust,ignore
/// struct MySigmoid;
///
/// impl Function for MySigmoid {
///     fn name(&self) -> &'static str { "sigmoid" }
///
///     fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
///         let output = &saved[0];  // sigmoid(x) was saved
///         // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
///         let grad = grad_output.mul(output)?.mul(&output.neg()?.add_scalar(1.0)?)?;
///         Ok(vec![Some(grad)])
///     }
/// }
/// ```
pub trait Function: Send + Sync {
    /// Human-readable name for debugging.
    fn name(&self) -> &'static str;

    /// Compute gradients for inputs given output gradient.
    ///
    /// # Arguments
    ///
    /// * `saved_tensors` - Tensors saved during forward for backward computation
    /// * `grad_output` - Gradient of the loss with respect to the output
    ///
    /// # Returns
    ///
    /// A vector of optional gradients for each input. `None` if an input
    /// doesn't require gradients.
    fn backward(
        &self,
        saved_tensors: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Option<Tensor>>>;

    /// Clone this function into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Function>;
}

// ============================================================================
// ARITHMETIC OPERATIONS
// ============================================================================

/// Addition backward: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1
#[derive(Clone)]
pub struct AddBackward {
    /// Shape of first input (for broadcasting reduction).
    pub a_shape: Shape,
    /// Shape of second input.
    pub b_shape: Shape,
}

impl Function for AddBackward {
    fn name(&self) -> &'static str {
        "add_backward"
    }

    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // Gradient flows through unchanged, but may need reduction for broadcasting
        let grad_a = reduce_gradient(grad_output, &self.a_shape)?;
        let grad_b = reduce_gradient(grad_output, &self.b_shape)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Subtraction backward: ∂(a-b)/∂a = 1, ∂(a-b)/∂b = -1
#[derive(Clone)]
pub struct SubBackward {
    pub a_shape: Shape,
    pub b_shape: Shape,
}

impl Function for SubBackward {
    fn name(&self) -> &'static str {
        "sub_backward"
    }

    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = reduce_gradient(grad_output, &self.a_shape)?;
        let grad_b = reduce_gradient(&grad_output.neg()?, &self.b_shape)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Multiplication backward: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
#[derive(Clone)]
pub struct MulBackward;

impl Function for MulBackward {
    fn name(&self) -> &'static str {
        "mul_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);
        let grad_a = reduce_gradient(&grad_output.mul(b)?, a.shape_obj())?;
        let grad_b = reduce_gradient(&grad_output.mul(a)?, b.shape_obj())?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Division backward: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
#[derive(Clone)]
pub struct DivBackward;

impl Function for DivBackward {
    fn name(&self) -> &'static str {
        "div_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);

        // ∂L/∂a = ∂L/∂out * 1/b
        let grad_a = reduce_gradient(&grad_output.div(b)?, a.shape_obj())?;

        // ∂L/∂b = ∂L/∂out * (-a/b²)
        let b_sq = b.pow(2.0)?;
        let grad_b = reduce_gradient(&grad_output.mul(&a.neg()?)?.div(&b_sq)?, b.shape_obj())?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Negation backward: ∂(-a)/∂a = -1
#[derive(Clone)]
pub struct NegBackward;

impl Function for NegBackward {
    fn name(&self) -> &'static str {
        "neg_backward"
    }

    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.neg()?)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

// ============================================================================
// MATRIX OPERATIONS
// ============================================================================

/// Matrix multiplication backward.
///
/// For C = A @ B:
/// - ∂L/∂A = ∂L/∂C @ Bᵀ
/// - ∂L/∂B = Aᵀ @ ∂L/∂C
#[derive(Clone)]
pub struct MatMulBackward;

impl Function for MatMulBackward {
    fn name(&self) -> &'static str {
        "matmul_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);

        // grad_a = grad_output @ b.T
        let grad_a = grad_output.matmul(&b.t()?)?;

        // grad_b = a.T @ grad_output
        let grad_b = a.t()?.matmul(grad_output)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

/// ReLU backward: ∂relu(x)/∂x = 1 if x > 0 else 0
#[derive(Clone)]
pub struct ReLUBackward;

impl Function for ReLUBackward {
    fn name(&self) -> &'static str {
        "relu_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];

        // Gradient is grad_output where input > 0, else 0
        // We create a mask by checking where input > 0
        let mask = create_relu_mask(input)?;
        let grad = grad_output.mul(&mask)?;

        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Sigmoid backward: ∂σ(x)/∂x = σ(x)(1 - σ(x))
#[derive(Clone)]
pub struct SigmoidBackward;

impl Function for SigmoidBackward {
    fn name(&self) -> &'static str {
        "sigmoid_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // We save the output σ(x), not the input x
        let sigmoid_out = &saved[0];

        // d/dx = σ(x) * (1 - σ(x))
        let one_minus_sigmoid = sigmoid_out.neg()?.add_scalar(1.0)?;
        let grad = grad_output.mul(sigmoid_out)?.mul(&one_minus_sigmoid)?;

        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Tanh backward: ∂tanh(x)/∂x = 1 - tanh²(x)
#[derive(Clone)]
pub struct TanhBackward;

impl Function for TanhBackward {
    fn name(&self) -> &'static str {
        "tanh_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // We save tanh(x)
        let tanh_out = &saved[0];

        // d/dx = 1 - tanh²(x)
        let tanh_sq = tanh_out.pow(2.0)?;
        let grad = grad_output.mul(&tanh_sq.neg()?.add_scalar(1.0)?)?;

        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// Sum backward: gradient is broadcast to original shape
#[derive(Clone)]
pub struct SumBackward {
    pub input_shape: Shape,
}

impl Function for SumBackward {
    fn name(&self) -> &'static str {
        "sum_backward"
    }

    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // grad_output is scalar, expand to input shape
        let grad = grad_output.expand(self.input_shape.clone())?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Mean backward: gradient is 1/n broadcast to original shape
#[derive(Clone)]
pub struct MeanBackward {
    pub input_shape: Shape,
}

impl Function for MeanBackward {
    fn name(&self) -> &'static str {
        "mean_backward"
    }

    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let numel = self.input_shape.numel();
        let scale = 1.0 / numel as f64;
        let grad = grad_output
            .mul_scalar(scale)?
            .expand(self.input_shape.clone())?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

// ============================================================================
// ELEMENT-WISE MATH
// ============================================================================

/// Exp backward: ∂exp(x)/∂x = exp(x)
#[derive(Clone)]
pub struct ExpBackward;

impl Function for ExpBackward {
    fn name(&self) -> &'static str {
        "exp_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // We save exp(x)
        let exp_out = &saved[0];
        let grad = grad_output.mul(exp_out)?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Log backward: ∂log(x)/∂x = 1/x
#[derive(Clone)]
pub struct LogBackward;

impl Function for LogBackward {
    fn name(&self) -> &'static str {
        "log_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        let grad = grad_output.div(input)?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Power backward: ∂(x^n)/∂x = n * x^(n-1)
#[derive(Clone)]
pub struct PowBackward {
    pub exponent: f64,
}

impl Function for PowBackward {
    fn name(&self) -> &'static str {
        "pow_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        // d/dx = n * x^(n-1)
        let grad = grad_output
            .mul_scalar(self.exponent)?
            .mul(&input.pow(self.exponent - 1.0)?)?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

/// Sqrt backward: ∂√x/∂x = 1/(2√x)
#[derive(Clone)]
pub struct SqrtBackward;

impl Function for SqrtBackward {
    fn name(&self) -> &'static str {
        "sqrt_backward"
    }

    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // We save sqrt(x)
        let sqrt_out = &saved[0];
        // d/dx = 1 / (2 * sqrt(x))
        let grad = grad_output.div(&sqrt_out.mul_scalar(2.0)?)?;
        Ok(vec![Some(grad)])
    }

    fn clone_box(&self) -> Box<dyn Function> {
        Box::new(self.clone())
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Reduce gradient to match a target shape (for broadcasting).
///
/// When a tensor was broadcast during forward, its gradient must be
/// summed along the broadcast dimensions.
fn reduce_gradient(grad: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    let grad_shape = grad.shape();
    let target_dims = target_shape.dims();

    // If shapes match, no reduction needed
    if grad_shape == target_dims {
        return Ok(grad.clone());
    }

    // Get total elements
    let grad_numel: usize = grad_shape.iter().product();
    let target_numel: usize = target_dims.iter().product();

    // Special case: target is scalar
    if target_numel == 1 {
        return grad.sum()?.reshape(target_shape.clone());
    }

    // For broadcasting reduction, we need to sum over the broadcast dimensions
    // This is a simplified implementation that handles common cases

    // If target has fewer dimensions, we need to reduce leading dims
    // AND possibly reduce dimensions that were broadcast (size 1 in target)

    // Calculate how many elements get summed into each target element
    let reduction_factor = grad_numel / target_numel;

    if reduction_factor == 1 {
        // Just a reshape
        return grad.reshape(target_shape.clone());
    }

    // For 2D gradient -> smaller target shape cases (common in broadcasting):
    // Example: grad [4, 1] -> target [1] means sum across batch
    // Example: grad [4, 4] -> target [1, 4] means sum across dim 0

    // Get data and manually compute the reduced gradient
    match grad.dtype() {
        ferrum_core::DType::F32 => {
            let grad_data = grad.to_vec::<f32>()?;
            let mut target_data = vec![0.0f32; target_numel];

            // Map each gradient element to its corresponding target element
            // This handles arbitrary broadcasting patterns
            for (i, &g) in grad_data.iter().enumerate() {
                let target_idx = compute_broadcast_index(i, &grad_shape, target_dims);
                target_data[target_idx] += g;
            }

            Tensor::from_slice(&target_data, target_dims, grad.device())
        }
        ferrum_core::DType::F64 => {
            let grad_data = grad.to_vec::<f64>()?;
            let mut target_data = vec![0.0f64; target_numel];

            for (i, &g) in grad_data.iter().enumerate() {
                let target_idx = compute_broadcast_index(i, &grad_shape, target_dims);
                target_data[target_idx] += g;
            }

            Tensor::from_slice(&target_data, target_dims, grad.device())
        }
        _ => Err(ferrum_core::FerrumError::UnsupportedDType {
            operation: "reduce_gradient",
            dtype: grad.dtype(),
        }),
    }
}

/// Compute the index in the target shape that corresponds to a flat index in the gradient.
/// This handles broadcasting by mapping larger indices to smaller ones.
fn compute_broadcast_index(flat_idx: usize, grad_shape: &[usize], target_shape: &[usize]) -> usize {
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    // Convert flat index to multi-dimensional coordinates in grad shape
    let mut grad_coords = vec![0usize; grad_ndim];
    let mut remaining = flat_idx;
    for i in (0..grad_ndim).rev() {
        grad_coords[i] = remaining % grad_shape[i];
        remaining /= grad_shape[i];
    }

    // Map to target coordinates, accounting for dimension offset and broadcasting
    let offset = grad_ndim - target_ndim;
    let mut target_idx = 0;
    let mut stride = 1;

    for i in (0..target_ndim).rev() {
        let grad_coord = grad_coords[i + offset];
        // If target dim is 1 (broadcast), map to 0; otherwise use the grad coord
        let target_coord = if target_shape[i] == 1 {
            0
        } else {
            grad_coord % target_shape[i]
        };
        target_idx += target_coord * stride;
        stride *= target_shape[i];
    }

    target_idx
}

/// Create a mask tensor for ReLU backward.
fn create_relu_mask(input: &Tensor) -> Result<Tensor> {
    let mut mask = Tensor::zeros(input.shape_obj().clone(), input.dtype(), input.device());

    // Set mask to 1 where input > 0
    match input.dtype() {
        DType::F32 => {
            let input_data = input.to_vec::<f32>()?;
            let mut mask_data = mask.to_vec::<f32>()?;
            for (m, &x) in mask_data.iter_mut().zip(input_data.iter()) {
                *m = if x > 0.0 { 1.0 } else { 0.0 };
            }
            mask = Tensor::from_slice(&mask_data, input.shape(), input.device())?;
        }
        DType::F64 => {
            let input_data = input.to_vec::<f64>()?;
            let mut mask_data = mask.to_vec::<f64>()?;
            for (m, &x) in mask_data.iter_mut().zip(input_data.iter()) {
                *m = if x > 0.0 { 1.0 } else { 0.0 };
            }
            mask = Tensor::from_slice(&mask_data, input.shape(), input.device())?;
        }
        _ => {
            return Err(ferrum_core::FerrumError::UnsupportedDType {
                operation: "relu_backward",
                dtype: input.dtype(),
            });
        }
    }

    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::Device;

    #[test]
    fn test_relu_backward() {
        let input = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], [4], Device::Cpu).unwrap();

        let grad_output = Tensor::ones([4], DType::F32, Device::Cpu);

        let relu_back = ReLUBackward;
        let grads = relu_back.backward(&[input], &grad_output).unwrap();

        let grad = grads[0].as_ref().unwrap();
        let grad_data = grad.to_vec::<f32>().unwrap();

        // Gradient should be 0 where input <= 0, 1 otherwise
        assert_eq!(grad_data, vec![0.0, 0.0, 1.0, 1.0]);
    }
}
