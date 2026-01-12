//! Autograd-aware tensor operations.
//!
//! This module provides the interface between tensor operations and the
//! automatic differentiation system. It uses a callback-based design to
//! avoid circular dependencies between ferrum-core and ferrum-autograd.

use std::cell::RefCell;
use std::sync::Arc;

use crate::{Result, Shape, Tensor};

// ============================================================================
// BACKWARD FUNCTION TRAIT (defined here so ferrum-autograd can implement)
// ============================================================================

/// Trait for backward functions that compute gradients.
pub trait BackwardFn: Send + Sync {
    /// Name of this backward function for debugging.
    fn name(&self) -> &'static str;
    
    /// Compute gradients of inputs given output gradient.
    ///
    /// # Arguments
    /// * `saved_tensors` - Tensors saved during forward pass
    /// * `grad_output` - Gradient of loss w.r.t. output of the forward operation
    ///
    /// # Returns
    /// Vector of optional gradients for each input (None if input doesn't require grad)
    fn backward(
        &self,
        saved_tensors: &[Tensor],
        grad_output: &Tensor,
    ) -> Result<Vec<Option<Tensor>>>;
    
    /// Clone into a boxed trait object.
    fn clone_box(&self) -> Box<dyn BackwardFn>;
}

// ============================================================================
// AUTOGRAD CONTEXT (callback system for recording operations)
// ============================================================================

/// Callback type for recording operations to the computation graph.
pub type RecordCallback = Arc<
    dyn Fn(
        Box<dyn BackwardFn>,  // backward function
        &[u64],               // input tensor IDs
        u64,                  // output tensor ID
        Vec<Tensor>,          // saved tensors for backward
    )
    + Send
    + Sync,
>;

/// Callback type for backward pass.
pub type BackwardCallback = Arc<dyn Fn(&Tensor) -> Result<()> + Send + Sync>;

/// Thread-local autograd context.
struct AutogradContext {
    /// Callback to record operations (set by ferrum-autograd).
    record_fn: Option<RecordCallback>,
    /// Callback for backward pass (set by ferrum-autograd).
    backward_fn: Option<BackwardCallback>,
    /// Whether autograd is enabled.
    enabled: bool,
    /// Counter for tensor IDs.
    tensor_id_counter: u64,
}

impl AutogradContext {
    fn new() -> Self {
        Self {
            record_fn: None,
            backward_fn: None,
            enabled: false,
            tensor_id_counter: 0,
        }
    }
    
    fn next_tensor_id(&mut self) -> u64 {
        self.tensor_id_counter += 1;
        self.tensor_id_counter
    }
}

thread_local! {
    static AUTOGRAD_CTX: RefCell<AutogradContext> = RefCell::new(AutogradContext::new());
}

// ============================================================================
// PUBLIC API FOR AUTOGRAD REGISTRATION
// ============================================================================

/// Register the autograd callbacks (called by ferrum-autograd when tape is created).
pub fn register_autograd_callbacks(
    record_fn: RecordCallback,
    backward_fn: BackwardCallback,
) {
    AUTOGRAD_CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.record_fn = Some(record_fn);
        ctx.backward_fn = Some(backward_fn);
        ctx.enabled = true;
    });
}

/// Unregister autograd callbacks.
pub fn unregister_autograd_callbacks() {
    AUTOGRAD_CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.record_fn = None;
        ctx.backward_fn = None;
        ctx.enabled = false;
    });
}

/// Check if autograd is currently enabled.
pub fn is_autograd_enabled() -> bool {
    AUTOGRAD_CTX.with(|ctx| ctx.borrow().enabled)
}

/// Temporarily disable autograd (for inference).
pub fn set_autograd_enabled(enabled: bool) -> bool {
    AUTOGRAD_CTX.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let was_enabled = ctx.enabled;
        ctx.enabled = enabled;
        was_enabled
    })
}

/// Get a new tensor ID.
pub fn new_tensor_id() -> u64 {
    AUTOGRAD_CTX.with(|ctx| ctx.borrow_mut().next_tensor_id())
}

/// Record an operation to the computation graph.
///
/// The output_tensor_id should be the tensor_id of the result tensor.
pub fn record_operation(
    backward_fn: Box<dyn BackwardFn>,
    input_tensor_ids: &[u64],
    output_tensor_id: u64,
    saved_tensors: Vec<Tensor>,
) {
    // First, check if enabled and get the callback (if any) without holding borrow
    let record_fn = AUTOGRAD_CTX.with(|ctx| {
        let ctx = ctx.borrow();
        if ctx.enabled {
            ctx.record_fn.clone()
        } else {
            None
        }
    });
    
    // Call the callback outside of the borrow
    if let Some(record) = record_fn {
        record(backward_fn, input_tensor_ids, output_tensor_id, saved_tensors);
    }
}

/// Execute backward pass on a tensor.
pub fn execute_backward(tensor: &Tensor) -> Result<()> {
    // Get callback without holding borrow
    let backward_fn = AUTOGRAD_CTX.with(|ctx| {
        ctx.borrow().backward_fn.clone()
    });

    if let Some(backward) = backward_fn {
        backward(tensor)
    } else {
        // No autograd registered - just set gradient to ones for leaf tensors
        if tensor.requires_grad() {
            tensor.set_grad(Some(Tensor::ones(
                tensor.shape_obj().clone(),
                tensor.dtype(),
                tensor.device(),
            )));
        }
        Ok(())
    }
}

// ============================================================================
// NO_GRAD CONTEXT MANAGER
// ============================================================================

/// RAII guard for temporarily disabling gradient computation.
pub struct NoGradGuard {
    was_enabled: bool,
}

impl NoGradGuard {
    /// Create a new no-grad context.
    pub fn new() -> Self {
        let was_enabled = set_autograd_enabled(false);
        Self { was_enabled }
    }
}

impl Default for NoGradGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for NoGradGuard {
    fn drop(&mut self) {
        set_autograd_enabled(self.was_enabled);
    }
}

/// Execute a closure with gradient computation disabled.
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoGradGuard::new();
    f()
}

// ============================================================================
// TENSOR BACKWARD IMPLEMENTATION
// ============================================================================

/// Trait providing backward() method for Tensor.
pub trait AutogradTensor {
    /// Compute gradients via backpropagation.
    ///
    /// This initiates the backward pass from this tensor (typically a scalar loss).
    /// Gradients are accumulated in all tensors that have `requires_grad=true`.
    fn backward(&self) -> Result<()>;
}

impl AutogradTensor for Tensor {
    fn backward(&self) -> Result<()> {
        // Verify this is a scalar
        if self.numel() != 1 {
            return Err(crate::error::FerrumError::InvalidShape {
                message: format!(
                    "backward() can only be called on scalar tensors, got shape {:?}",
                    self.shape()
                ),
            });
        }

        execute_backward(self)
    }
}

// ============================================================================
// COMMON BACKWARD FUNCTION IMPLEMENTATIONS
// ============================================================================

/// Addition backward: gradients flow through unchanged (with broadcast reduction).
#[derive(Clone)]
pub struct AddBackward {
    pub a_shape: Shape,
    pub b_shape: Shape,
}

impl BackwardFn for AddBackward {
    fn name(&self) -> &'static str { "add_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape)?;
        let grad_b = reduce_to_shape(grad_output, &self.b_shape)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Subtraction backward.
#[derive(Clone)]
pub struct SubBackward {
    pub a_shape: Shape,
    pub b_shape: Shape,
}

impl BackwardFn for SubBackward {
    fn name(&self) -> &'static str { "sub_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad_a = reduce_to_shape(grad_output, &self.a_shape)?;
        let grad_b = reduce_to_shape(&grad_output.neg()?, &self.b_shape)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Multiplication backward: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
#[derive(Clone)]
pub struct MulBackward;

impl BackwardFn for MulBackward {
    fn name(&self) -> &'static str { "mul_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);
        let grad_a = reduce_to_shape(&grad_output.mul(b)?, a.shape_obj())?;
        let grad_b = reduce_to_shape(&grad_output.mul(a)?, b.shape_obj())?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Division backward: ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = -a/b²
#[derive(Clone)]
pub struct DivBackward;

impl BackwardFn for DivBackward {
    fn name(&self) -> &'static str { "div_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);
        let grad_a = reduce_to_shape(&grad_output.div(b)?, a.shape_obj())?;
        let b_sq = b.pow(2.0)?;
        let grad_b = reduce_to_shape(&grad_output.mul(&a.neg()?)?.div(&b_sq)?, b.shape_obj())?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Matrix multiplication backward.
#[derive(Clone)]
pub struct MatMulBackward;

impl BackwardFn for MatMulBackward {
    fn name(&self) -> &'static str { "matmul_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let (a, b) = (&saved[0], &saved[1]);
        let grad_a = grad_output.matmul(&b.t()?)?;
        let grad_b = a.t()?.matmul(grad_output)?;
        Ok(vec![Some(grad_a), Some(grad_b)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Sum backward: gradient expands to input shape.
#[derive(Clone)]
pub struct SumBackward {
    pub input_shape: Shape,
}

impl BackwardFn for SumBackward {
    fn name(&self) -> &'static str { "sum_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let grad = grad_output.expand(self.input_shape.clone())?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Mean backward: gradient is 1/n expanded to input shape.
#[derive(Clone)]
pub struct MeanBackward {
    pub input_shape: Shape,
}

impl BackwardFn for MeanBackward {
    fn name(&self) -> &'static str { "mean_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let n = self.input_shape.numel() as f64;
        let grad = grad_output.mul_scalar(1.0 / n)?.expand(self.input_shape.clone())?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Sum along dimension backward: gradient expands back to input shape.
#[derive(Clone)]
pub struct SumDimBackward {
    pub input_shape: Shape,
    pub dim: usize,
    pub keepdim: bool,
}

impl BackwardFn for SumDimBackward {
    fn name(&self) -> &'static str { "sum_dim_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // If keepdim was false, we need to unsqueeze before expanding
        let grad = if self.keepdim {
            grad_output.expand(self.input_shape.clone())?
        } else {
            // Unsqueeze the gradient at the reduced dimension
            let grad_unsqueezed = grad_output.unsqueeze(self.dim as i64)?;
            grad_unsqueezed.expand(self.input_shape.clone())?
        };
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Concatenation backward: split gradient back to original tensors.
#[derive(Clone)]
pub struct CatBackward {
    pub dim: usize,
    pub sizes: Vec<usize>,
}

impl BackwardFn for CatBackward {
    fn name(&self) -> &'static str { "cat_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        // Split the gradient along the concatenation dimension
        let mut grads = Vec::with_capacity(self.sizes.len());
        let mut offset = 0;
        
        for &size in &self.sizes {
            // Use narrow to extract the slice
            let grad_slice = grad_output.narrow(self.dim, offset, size)?;
            grads.push(Some(grad_slice));
            offset += size;
        }
        
        Ok(grads)
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Power backward: ∂(x^n)/∂x = n * x^(n-1)
#[derive(Clone)]
pub struct PowBackward {
    pub exponent: f64,
}

impl BackwardFn for PowBackward {
    fn name(&self) -> &'static str { "pow_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        let grad = grad_output
            .mul_scalar(self.exponent)?
            .mul(&input.pow(self.exponent - 1.0)?)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Exp backward: ∂exp(x)/∂x = exp(x)
#[derive(Clone)]
pub struct ExpBackward;

impl BackwardFn for ExpBackward {
    fn name(&self) -> &'static str { "exp_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let exp_output = &saved[0]; // We save exp(x), not x
        let grad = grad_output.mul(exp_output)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Log backward: ∂log(x)/∂x = 1/x
#[derive(Clone)]
pub struct LogBackward;

impl BackwardFn for LogBackward {
    fn name(&self) -> &'static str { "log_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        let grad = grad_output.div(input)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// ReLU backward: gradient passes through where input > 0.
#[derive(Clone)]
pub struct ReLUBackward;

impl BackwardFn for ReLUBackward {
    fn name(&self) -> &'static str { "relu_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let input = &saved[0];
        // Create mask: 1 where input > 0, 0 elsewhere
        let mask = create_positive_mask(input)?;
        let grad = grad_output.mul(&mask)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Sigmoid backward: ∂σ(x)/∂x = σ(x) * (1 - σ(x))
#[derive(Clone)]
pub struct SigmoidBackward;

impl BackwardFn for SigmoidBackward {
    fn name(&self) -> &'static str { "sigmoid_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let sig_out = &saved[0]; // We save sigmoid(x), not x
        let one_minus_sig = sig_out.neg()?.add_scalar(1.0)?;
        let grad = grad_output.mul(sig_out)?.mul(&one_minus_sig)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Tanh backward: ∂tanh(x)/∂x = 1 - tanh²(x)
#[derive(Clone)]
pub struct TanhBackward;

impl BackwardFn for TanhBackward {
    fn name(&self) -> &'static str { "tanh_backward" }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let tanh_out = &saved[0]; // We save tanh(x), not x
        let tanh_sq = tanh_out.pow(2.0)?;
        let grad = grad_output.mul(&tanh_sq.neg()?.add_scalar(1.0)?)?;
        Ok(vec![Some(grad)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

/// Negation backward: ∂(-x)/∂x = -1
#[derive(Clone)]
pub struct NegBackward;

impl BackwardFn for NegBackward {
    fn name(&self) -> &'static str { "neg_backward" }
    
    fn backward(&self, _saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(grad_output.neg()?)])
    }
    
    fn clone_box(&self) -> Box<dyn BackwardFn> {
        Box::new(self.clone())
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Reduce gradient to match target shape (for broadcasting).
fn reduce_to_shape(grad: &Tensor, target_shape: &Shape) -> Result<Tensor> {
    let grad_shape = grad.shape_obj();
    
    // If shapes already match, return as-is
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }
    
    let grad_dims = grad_shape.dims();
    let target_dims = target_shape.dims();
    
    // Handle scalar target case - sum everything
    if target_dims.is_empty() {
        return grad.sum();
    }
    
    // If gradient is scalar, broadcast it up to target shape
    if grad_dims.is_empty() {
        return grad.expand(target_shape.clone());
    }
    
    // For shapes with same number of dimensions but different sizes due to broadcasting,
    // we need to sum along the broadcast dimensions
    let mut result = grad.clone();
    
    // Handle extra leading dimensions in gradient
    let offset = grad_dims.len().saturating_sub(target_dims.len());
    
    // Sum over extra leading dimensions one at a time
    for _ in 0..offset {
        // Sum along axis 0 repeatedly
        result = sum_along_axis(&result, 0)?;
    }
    
    // Now handle dimensions that were broadcast (size 1 in target)
    let result_dims = result.shape();
    if result_dims.len() == target_dims.len() {
        // Sum along dimensions where target has size 1 but result doesn't
        for i in (0..target_dims.len()).rev() {
            if target_dims[i] == 1 && result.shape()[i] != 1 {
                result = sum_along_axis(&result, i)?;
            }
        }
    }
    
    // Final check - if shapes still don't match, try reshape
    if result.shape() != target_dims && result.numel() == target_shape.numel() {
        result = result.reshape(target_shape.clone())?;
    }
    
    Ok(result)
}

/// Sum along a specific axis, keeping dimensions.
fn sum_along_axis(tensor: &Tensor, axis: usize) -> Result<Tensor> {
    use crate::dtype::DType;
    use crate::shape::Shape;
    
    let shape = tensor.shape();
    if axis >= shape.len() {
        return Err(crate::error::FerrumError::InvalidShape {
            message: format!("axis {} out of bounds for shape {:?}", axis, shape),
        });
    }
    
    // Calculate output shape with dimension at axis reduced to 1
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape[axis] = 1;
    let new_shape = Shape::from(new_shape);
    
    // Calculate strides for iteration
    let pre_axis_size: usize = shape[..axis].iter().product();
    let axis_size = shape[axis];
    let post_axis_size: usize = shape[axis + 1..].iter().product();
    if post_axis_size == 0 || pre_axis_size == 0 {
        // Handle edge case of empty dimensions
        return tensor.reshape(new_shape);
    }
    
    match tensor.dtype() {
        DType::F32 => {
            let src = tensor.to_vec::<f32>()?;
            let mut dst = vec![0.0f32; pre_axis_size * post_axis_size];
            
            for pre in 0..pre_axis_size {
                for post in 0..post_axis_size {
                    let mut sum = 0.0f32;
                    for a in 0..axis_size {
                        let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                        sum += src[idx];
                    }
                    let out_idx = pre * post_axis_size + post;
                    dst[out_idx] = sum;
                }
            }
            
            Tensor::from_slice(&dst, new_shape, tensor.device())
        }
        DType::F64 => {
            let src = tensor.to_vec::<f64>()?;
            let mut dst = vec![0.0f64; pre_axis_size * post_axis_size];
            
            for pre in 0..pre_axis_size {
                for post in 0..post_axis_size {
                    let mut sum = 0.0f64;
                    for a in 0..axis_size {
                        let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                        sum += src[idx];
                    }
                    let out_idx = pre * post_axis_size + post;
                    dst[out_idx] = sum;
                }
            }
            
            Tensor::from_slice(&dst, new_shape, tensor.device())
        }
        _ => {
            Err(crate::error::FerrumError::UnsupportedDType {
                operation: "sum_along_axis",
                dtype: tensor.dtype(),
            })
        }
    }
}

/// Create a mask tensor with 1 where input > 0, 0 elsewhere.
fn create_positive_mask(input: &Tensor) -> Result<Tensor> {
    use crate::dtype::DType;
    
    match input.dtype() {
        DType::F32 => {
            let data = input.to_vec::<f32>()?;
            let mask: Vec<f32> = data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
            Tensor::from_slice(&mask, input.shape(), input.device())
        }
        DType::F64 => {
            let data = input.to_vec::<f64>()?;
            let mask: Vec<f64> = data.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect();
            Tensor::from_slice(&mask, input.shape(), input.device())
        }
        dtype => Err(crate::error::FerrumError::UnsupportedDType {
            operation: "relu_backward",
            dtype,
        }),
    }
}
