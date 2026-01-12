//! Collective communication operations.

use ferrum_core::Tensor;
use crate::error::Result;

/// Reduction operations for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Compute product of all values.
    Product,
    /// Take minimum value.
    Min,
    /// Take maximum value.
    Max,
    /// Compute average (sum / world_size).
    Average,
}

impl ReduceOp {
    /// Apply reduction operation to two scalars.
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match self {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
            ReduceOp::Average => a + b, // Will divide by world_size after
        }
    }
}

/// Collective operations interface.
pub trait Collectives {
    /// Broadcast tensor from root to all processes.
    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()>;

    /// All-reduce: reduce and distribute result to all processes.
    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()>;

    /// Reduce to root process.
    fn reduce(&self, tensor: &mut Tensor, dst: usize, op: ReduceOp) -> Result<()>;

    /// All-gather: gather tensors from all processes to all processes.
    fn all_gather(&self, output: &mut [Tensor], input: &Tensor) -> Result<()>;

    /// Gather tensors to root process.
    fn gather(&self, output: Option<&mut [Tensor]>, input: &Tensor, dst: usize) -> Result<()>;

    /// Scatter tensors from root to all processes.
    fn scatter(&self, output: &mut Tensor, input: Option<&[Tensor]>, src: usize) -> Result<()>;

    /// Reduce-scatter: reduce and scatter result.
    fn reduce_scatter(&self, output: &mut Tensor, input: &[Tensor], op: ReduceOp) -> Result<()>;

    /// Barrier synchronization.
    fn barrier(&self) -> Result<()>;
}

/// Simulated collective operations for testing.
pub struct SimulatedCollectives {
    rank: usize,
    world_size: usize,
}

impl SimulatedCollectives {
    /// Create new simulated collectives.
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }
}

impl Collectives for SimulatedCollectives {
    fn broadcast(&self, _tensor: &mut Tensor, _root: usize) -> Result<()> {
        // In single-process mode, broadcast is a no-op
        Ok(())
    }

    fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        // In single-process mode, all_reduce is identity or average
        if op == ReduceOp::Average {
            // Divide by world_size (which is 1, so no change)
        }
        let _ = tensor;
        Ok(())
    }

    fn reduce(&self, _tensor: &mut Tensor, _dst: usize, _op: ReduceOp) -> Result<()> {
        Ok(())
    }

    fn all_gather(&self, output: &mut [Tensor], input: &Tensor) -> Result<()> {
        if !output.is_empty() {
            output[0] = input.clone();
        }
        Ok(())
    }

    fn gather(&self, output: Option<&mut [Tensor]>, input: &Tensor, _dst: usize) -> Result<()> {
        if let Some(out) = output {
            if !out.is_empty() {
                out[0] = input.clone();
            }
        }
        Ok(())
    }

    fn scatter(&self, output: &mut Tensor, input: Option<&[Tensor]>, _src: usize) -> Result<()> {
        if let Some(inp) = input {
            if !inp.is_empty() {
                *output = inp[0].clone();
            }
        }
        Ok(())
    }

    fn reduce_scatter(&self, output: &mut Tensor, input: &[Tensor], _op: ReduceOp) -> Result<()> {
        if !input.is_empty() {
            *output = input[0].clone();
        }
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        // No-op in single process
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_ops() {
        assert_eq!(ReduceOp::Sum.apply(3.0, 4.0), 7.0);
        assert_eq!(ReduceOp::Product.apply(3.0, 4.0), 12.0);
        assert_eq!(ReduceOp::Min.apply(3.0, 4.0), 3.0);
        assert_eq!(ReduceOp::Max.apply(3.0, 4.0), 4.0);
    }

    #[test]
    fn test_simulated_barrier() {
        let collectives = SimulatedCollectives::new(0, 1);
        assert!(collectives.barrier().is_ok());
    }
}
