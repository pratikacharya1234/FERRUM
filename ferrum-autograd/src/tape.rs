//! Gradient tape for recording operations.
//!
//! The gradient tape provides a context manager-like API for recording
//! operations and computing gradients.

use std::cell::RefCell;
use std::sync::Arc;

use parking_lot::Mutex;

use ferrum_core::{autograd_ops, Result, Tensor};

use crate::graph::ComputationGraph;

// Thread-local gradient tape for implicit recording.
thread_local! {
    static CURRENT_TAPE: RefCell<Option<Arc<GradientTape>>> = const { RefCell::new(None) };
}

/// Gradient tape for recording differentiable operations.
///
/// The tape records operations during the forward pass and enables
/// reverse-mode automatic differentiation via the `backward` method.
///
/// ## Usage
///
/// There are two ways to use the tape:
///
/// ### 1. Explicit Recording (recommended)
///
/// ```rust,ignore
/// let tape = GradientTape::new();
/// let x = Tensor::randn(&[2, 3], DType::F32, Device::Cpu).requires_grad(true);
///
/// let y = tape.watch(&x);  // Start tracking
/// let z = y.matmul(&w);
/// let loss = z.sum();
///
/// let grads = tape.gradient(&loss, &[&x])?;
/// ```
///
/// ### 2. Implicit Recording
///
/// ```rust,ignore
/// GradientTape::with_tape(|tape| {
///     let x = Tensor::randn(&[2, 3], DType::F32, Device::Cpu).requires_grad(true);
///     let loss = x.pow(2.0).sum();
///     tape.backward(&loss)
/// })?;
/// ```
pub struct GradientTape {
    /// Underlying computation graph.
    graph: Mutex<ComputationGraph>,
    /// Whether the tape is currently recording.
    recording: Mutex<bool>,
}

impl GradientTape {
    /// Create a new gradient tape.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            graph: Mutex::new(ComputationGraph::new()),
            recording: Mutex::new(true),
        })
    }

    /// Check if the tape is currently recording.
    pub fn is_recording(&self) -> bool {
        *self.recording.lock()
    }

    /// Stop recording (operations won't be tracked).
    pub fn stop_recording(&self) {
        *self.recording.lock() = false;
    }

    /// Resume recording.
    pub fn start_recording(&self) {
        *self.recording.lock() = true;
    }

    /// Execute a closure with recording disabled.
    pub fn no_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let was_recording = self.is_recording();
        self.stop_recording();
        let result = f();
        if was_recording {
            self.start_recording();
        }
        result
    }

    /// Record an operation in the tape.
    ///
    /// This is called internally by tensor operations when requires_grad is true.
    pub fn record(
        &self,
        function: Box<dyn crate::function::Function>,
        input_tensor_ids: &[u64],
        output_tensor_id: u64,
        saved_tensors: Vec<Tensor>,
    ) {
        if !self.is_recording() {
            return;
        }
        self.graph
            .lock()
            .record(function, input_tensor_ids, output_tensor_id, saved_tensors);
    }

    /// Compute gradients of a scalar output with respect to inputs.
    ///
    /// # Arguments
    ///
    /// * `output` - The scalar tensor to differentiate (typically loss)
    /// * `inputs` - Tensors to compute gradients for
    ///
    /// # Returns
    ///
    /// A vector of gradients, one for each input tensor.
    pub fn gradient(&self, output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        // Verify output is scalar
        if output.numel() != 1 {
            return Err(ferrum_core::FerrumError::InvalidShape {
                message: format!(
                    "backward() requires scalar output, got shape {:?}",
                    output.shape()
                ),
            });
        }

        // Run backward pass
        crate::backward::backward(output, &self.graph.lock())?;

        // Collect gradients for requested inputs
        let grads: Vec<Tensor> = inputs
            .iter()
            .map(|t| {
                t.grad()
                    .unwrap_or_else(|| Tensor::zeros(t.shape_obj().clone(), t.dtype(), t.device()))
            })
            .collect();

        Ok(grads)
    }

    /// Clear the tape (remove all recorded operations).
    ///
    /// Call this between training iterations to free memory.
    pub fn clear(&self) {
        self.graph.lock().clear();
    }

    /// Get the number of recorded operations.
    pub fn len(&self) -> usize {
        self.graph.lock().len()
    }

    /// Check if the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.graph.lock().is_empty()
    }

    /// Run a function with this tape as the current tape.
    ///
    /// This also registers the tape with the global autograd context so that
    /// tensor operations will automatically be recorded.
    pub fn with_tape<F, R>(f: F) -> R
    where
        F: FnOnce(&Arc<GradientTape>) -> R,
    {
        let tape = GradientTape::new();
        
        // Set thread-local tape
        CURRENT_TAPE.with(|current| {
            *current.borrow_mut() = Some(tape.clone());
        });
        
        // Register with global autograd context
        let tape_for_record = tape.clone();
        let tape_for_backward = tape.clone();
        
        let record_fn: autograd_ops::RecordCallback = Arc::new(move |backward_fn, input_ids, output_id, saved| {
            // Adapt BackwardFn to Function
            let adapted = BackwardFnAdapter(backward_fn);
            tape_for_record.record(Box::new(adapted), input_ids, output_id, saved);
        });
        
        let backward_fn: autograd_ops::BackwardCallback = Arc::new(move |tensor| {
            // Run backward using the tape's graph
            tape_for_backward.backward(tensor)
        });
        
        autograd_ops::register_autograd_callbacks(record_fn, backward_fn);
        
        // Run the user's code
        let result = f(&tape);
        
        // Cleanup
        autograd_ops::unregister_autograd_callbacks();
        CURRENT_TAPE.with(|current| {
            *current.borrow_mut() = None;
        });
        
        result
    }

    /// Compute gradients via backward pass (internal method called from autograd_ops).
    fn backward(&self, output: &Tensor) -> Result<()> {
        crate::backward::backward(output, &self.graph.lock())
    }

    /// Get the current thread-local tape (if any).
    pub fn current() -> Option<Arc<GradientTape>> {
        CURRENT_TAPE.with(|current| current.borrow().clone())
    }
}

/// Adapter to use BackwardFn (from ferrum-core) as Function (from ferrum-autograd).
struct BackwardFnAdapter(Box<dyn autograd_ops::BackwardFn>);

impl crate::function::Function for BackwardFnAdapter {
    fn name(&self) -> &'static str {
        self.0.name()
    }
    
    fn backward(&self, saved: &[Tensor], grad_output: &Tensor) -> Result<Vec<Option<Tensor>>> {
        self.0.backward(saved, grad_output)
    }
    
    fn clone_box(&self) -> Box<dyn crate::function::Function> {
        Box::new(BackwardFnAdapter(self.0.clone_box()))
    }
}

impl Default for GradientTape {
    fn default() -> Self {
        Self {
            graph: Mutex::new(ComputationGraph::new()),
            recording: Mutex::new(true),
        }
    }
}

impl std::fmt::Debug for GradientTape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientTape")
            .field("recording", &self.is_recording())
            .field("num_ops", &self.len())
            .finish()
    }
}

/// Context manager for disabling gradient computation.
///
/// # Example
///
/// ```rust,ignore
/// let x = Tensor::randn(&[2, 3], DType::F32, Device::Cpu).requires_grad(true);
///
/// // Gradients are tracked here
/// let y = x.pow(2.0);
///
/// no_grad(|| {
///     // Gradients are not tracked here
///     let z = x.pow(3.0);
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    if let Some(tape) = GradientTape::current() {
        tape.no_grad(f)
    } else {
        f()
    }
}

/// Check if gradient computation is currently enabled.
pub fn is_grad_enabled() -> bool {
    GradientTape::current()
        .map(|t| t.is_recording())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_creation() {
        let tape = GradientTape::new();
        assert!(tape.is_recording());
        assert!(tape.is_empty());
    }

    #[test]
    fn test_no_grad() {
        let tape = GradientTape::new();

        tape.no_grad(|| {
            assert!(!tape.is_recording());
        });

        assert!(tape.is_recording());
    }

    #[test]
    fn test_with_tape() {
        GradientTape::with_tape(|tape| {
            assert!(GradientTape::current().is_some());
            assert!(tape.is_recording());
        });

        assert!(GradientTape::current().is_none());
    }
}
