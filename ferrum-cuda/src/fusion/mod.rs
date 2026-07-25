//! NVRTC Kernel Fusion Compiler.
//!
//! This module provides automatic fusion of element-wise operations
//! into single CUDA kernels, reducing kernel launch overhead and
//! memory traffic.
//!
//! # Architecture
//!
//! ```text
//! Autograd Graph
//!     │
//!     ▼
//! FusionPartitioner::partition()
//!     │
//!     ▼
//! Vec<FusionRegion>
//!     │
//!     ▼
//! CodeGen::generate() for each region
//!     │
//!     ▼
//! NvrtcCompiler::compile_source()
//!     │
//!     ▼
//! CompiledKernel (cached by source hash)
//! ```

pub mod codegen;
pub mod compiler;
pub mod ir;
pub mod kernels;
pub mod partition;

pub use codegen::CodeGen;
pub use compiler::{CompiledKernel, FusionError, NvrtcCompiler};
pub use ir::{FusionGraph, FusionNode, FusionOp};
pub use partition::{FusionPartitioner, FusionRegion};

use std::sync::Arc;

use crate::cuda_device::CudaDevice;
use crate::error::CudaResult;

/// High-level fusion API.
pub struct FusionEngine {
    compiler: NvrtcCompiler,
}

impl FusionEngine {
    /// Create a new fusion engine.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            compiler: NvrtcCompiler::new(device),
        }
    }

    /// Compile a fusion graph into a launchable kernel.
    pub fn compile(
        &self,
        graph: &FusionGraph,
        kernel_name: &str,
    ) -> Result<CompiledKernel, FusionError> {
        self.compiler.compile_graph(graph, kernel_name)
    }

    /// Compile a blessed kernel by name.
    pub fn compile_blessed(&self, name: &str) -> Result<CompiledKernel, FusionError> {
        let source = kernels::blessed_kernels()
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, s)| *s)
            .ok_or_else(|| FusionError::CompileError(format!("Unknown blessed kernel: {}", name)))?;

        self.compiler.compile_source(source, name)
    }

    /// Compile and cache all blessed kernels.
    pub fn warmup(&self) -> Vec<Result<CompiledKernel, FusionError>> {
        kernels::blessed_kernels()
            .iter()
            .map(|(name, source)| self.compiler.compile_source(source, name))
            .collect()
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        self.compiler.cache_stats()
    }
}

/// Fuse and compile a chain of element-wise operations.
///
/// This is the main entry point for kernel fusion. Given a sequence
/// of operations, it:
/// 1. Builds a fusion graph
/// 2. Partitions into fusion regions
/// 3. Compiles each region into a CUDA kernel
/// 4. Returns the compiled kernels
///
/// # Example
/// ```rust,ignore
/// // Instead of 4 separate kernel launches:
/// let tmp1 = a.add(b);
/// let tmp2 = tmp1.mul(c);
/// let tmp3 = tmp2.relu();
/// let out = tmp3.add(d);
///
/// // Fuse into 1 kernel:
/// let fused = fuse_and_compile(&[a, b, c, d], &[Add, Mul, Relu, Add]);
/// ```
pub fn fuse_and_compile(
    device: &Arc<CudaDevice>,
    input_shapes: &[Vec<usize>],
    ops: &[FusionOp],
    output_shape: Vec<usize>,
) -> Result<CompiledKernel, FusionError> {
    let engine = FusionEngine::new(device.clone());

    // Build fusion graph
    let mut graph = FusionGraph::new();

    // Add inputs
    let input_indices: Vec<usize> = input_shapes
        .iter()
        .map(|shape| graph.add_input(shape.clone()))
        .collect();

    // Add operations
    let mut current = input_indices[0];
    for (i, op) in ops.iter().enumerate() {
        let inputs = if op.is_binary() {
            vec![current, input_indices[i + 1]]
        } else {
            vec![current]
        };
        current = graph.add_op(op.clone(), inputs, output_shape.clone());
    }

    graph.mark_output(current);

    // Compile
    engine.compile(&graph, "fused_kernel")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_engine_creation() {
        if let Ok(device) = crate::cuda_device::CudaDevice::new(0) {
            let device = Arc::new(device);
            let engine = FusionEngine::new(device);
            let (cached, _) = engine.cache_stats();
            assert_eq!(cached, 0);
        }
    }

    #[test]
    fn test_fuse_and_compile() {
        if let Ok(device) = crate::cuda_device::CudaDevice::new(0) {
            let device = Arc::new(device);
            let result = fuse_and_compile(
                &device,
                &[vec![100], vec![100], vec![100], vec![100]],
                &[FusionOp::Add, FusionOp::Mul, FusionOp::Relu, FusionOp::Add],
                vec![100],
            );
            // Will succeed only if NVRTC is available
            if let Ok(kernel) = result {
                assert!(!kernel.function.is_null());
            }
        }
    }
}
