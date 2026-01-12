//! Distributed Data Parallel (DDP) wrapper.

use std::sync::Arc;
use parking_lot::Mutex;

use ferrum_core::Tensor;
use crate::collectives::ReduceOp;
use crate::error::Result;
use crate::process_group::ProcessGroup;

/// Distributed Data Parallel wrapper for models.
///
/// Wraps a model to enable data parallel training across multiple processes.
/// Gradients are synchronized via all-reduce after backward pass.
///
/// # Example
///
/// ```ignore
/// use ferrum_distributed::{DistributedDataParallel, ProcessGroup, Backend};
/// use ferrum_nn::Module;
///
/// let model = MyModel::new();
/// let pg = ProcessGroup::new(Backend::Gloo, 0, 4)?;
/// let ddp_model = DistributedDataParallel::new(model, pg);
///
/// // Training loop
/// for batch in dataloader {
///     let output = ddp_model.forward(&input);
///     let loss = criterion(&output, &target);
///     loss.backward();
///     ddp_model.sync_gradients();
///     optimizer.step();
/// }
/// ```
pub struct DistributedDataParallel<M> {
    /// The wrapped module.
    module: M,
    /// Process group for communication.
    process_group: Arc<ProcessGroup>,
    /// Bucket size for gradient bucketing (bytes).
    bucket_size_mb: usize,
    /// Whether to broadcast buffers.
    broadcast_buffers: bool,
    /// Whether to find unused parameters.
    find_unused_parameters: bool,
}

impl<M> DistributedDataParallel<M> {
    /// Create a new DDP wrapper.
    pub fn new(module: M, process_group: Arc<ProcessGroup>) -> Self {
        Self {
            module,
            process_group,
            bucket_size_mb: 25, // Default 25MB buckets like PyTorch
            broadcast_buffers: true,
            find_unused_parameters: false,
        }
    }

    /// Set the bucket size for gradient bucketing (in MB).
    pub fn bucket_size_mb(mut self, size: usize) -> Self {
        self.bucket_size_mb = size;
        self
    }

    /// Enable/disable buffer broadcasting.
    pub fn broadcast_buffers(mut self, broadcast: bool) -> Self {
        self.broadcast_buffers = broadcast;
        self
    }

    /// Enable/disable finding unused parameters.
    pub fn find_unused_parameters(mut self, find: bool) -> Self {
        self.find_unused_parameters = find;
        self
    }

    /// Get reference to the wrapped module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Get mutable reference to the wrapped module.
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Get the process group.
    pub fn process_group(&self) -> &ProcessGroup {
        &self.process_group
    }

    /// Synchronize gradients across all processes.
    ///
    /// This performs an all-reduce on all gradients and divides by world size.
    pub fn sync_gradients(&self, gradients: &mut [Tensor]) -> Result<()> {
        for grad in gradients {
            self.process_group.all_reduce(grad, ReduceOp::Average)?;
        }
        Ok(())
    }

    /// Broadcast model parameters from rank 0.
    ///
    /// Call this before training to ensure all processes start with the same weights.
    pub fn broadcast_parameters(&self, parameters: &mut [Tensor]) -> Result<()> {
        for param in parameters {
            self.process_group.broadcast(param, 0)?;
        }
        Ok(())
    }
}

// Forward Module trait if M implements it
impl<M> DistributedDataParallel<M>
where
    M: Clone,
{
    /// Get a clone of the inner module.
    pub fn into_inner(self) -> M {
        self.module
    }
}

/// Gradient bucket for efficient all-reduce.
#[derive(Debug)]
pub struct GradientBucket {
    /// Flattened gradients in this bucket.
    data: Vec<f32>,
    /// Parameter indices in this bucket.
    param_indices: Vec<usize>,
    /// Size of each parameter's gradient.
    sizes: Vec<usize>,
    /// Whether the bucket is ready for all-reduce.
    ready: bool,
}

impl GradientBucket {
    /// Create a new empty bucket.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            param_indices: Vec::new(),
            sizes: Vec::new(),
            ready: false,
        }
    }

    /// Add a gradient to the bucket.
    pub fn add_gradient(&mut self, param_idx: usize, grad: &[f32]) {
        self.param_indices.push(param_idx);
        self.sizes.push(grad.len());
        self.data.extend_from_slice(grad);
    }

    /// Mark bucket as ready for all-reduce.
    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    /// Check if bucket is ready.
    pub fn is_ready(&self) -> bool {
        self.ready
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

impl Default for GradientBucket {
    fn default() -> Self {
        Self::new()
    }
}

/// Gradient synchronization hook.
pub struct GradientHook {
    /// Buckets for gradient accumulation.
    buckets: Vec<Mutex<GradientBucket>>,
    /// Process group for communication.
    process_group: Arc<ProcessGroup>,
}

impl GradientHook {
    /// Create a new gradient hook.
    pub fn new(process_group: Arc<ProcessGroup>, num_buckets: usize) -> Self {
        let buckets = (0..num_buckets)
            .map(|_| Mutex::new(GradientBucket::new()))
            .collect();
        
        Self {
            buckets,
            process_group,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend;

    struct DummyModel {
        value: i32,
    }

    impl DummyModel {
        fn new() -> Self {
            Self { value: 42 }
        }
    }

    #[test]
    fn test_ddp_creation() {
        let model = DummyModel::new();
        let pg = Arc::new(ProcessGroup::new(Backend::Gloo, 0, 1).unwrap());
        let ddp = DistributedDataParallel::new(model, pg);
        
        assert_eq!(ddp.module().value, 42);
    }

    #[test]
    fn test_gradient_bucket() {
        let mut bucket = GradientBucket::new();
        bucket.add_gradient(0, &[1.0, 2.0, 3.0]);
        bucket.add_gradient(1, &[4.0, 5.0]);
        
        assert_eq!(bucket.data.len(), 5);
        assert_eq!(bucket.param_indices.len(), 2);
        assert!(!bucket.is_ready());
        
        bucket.mark_ready();
        assert!(bucket.is_ready());
    }
}
