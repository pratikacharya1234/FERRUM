//! Process group for distributed communication.

use std::sync::Arc;
use parking_lot::Mutex;

use crate::backend::{Backend, BackendConfig};
use crate::collectives::{Collectives, ReduceOp, SimulatedCollectives};
use crate::error::{DistributedError, Result};
use ferrum_core::Tensor;

/// A process group for collective communication.
pub struct ProcessGroup {
    /// Backend used for communication.
    backend: Backend,
    /// Rank of this process.
    rank: usize,
    /// Total number of processes.
    world_size: usize,
    /// Backend configuration.
    config: BackendConfig,
    /// Collective operations implementation.
    collectives: Arc<Mutex<Box<dyn Collectives + Send>>>,
}

impl ProcessGroup {
    /// Create a new process group.
    pub fn new(backend: Backend, rank: usize, world_size: usize) -> Result<Self> {
        if rank >= world_size {
            return Err(DistributedError::InvalidRank(rank, world_size));
        }

        if !backend.is_available() {
            return Err(DistributedError::UnsupportedBackend(backend.name().to_string()));
        }

        let collectives: Box<dyn Collectives + Send> = Box::new(SimulatedCollectives::new(rank, world_size));

        Ok(Self {
            backend,
            rank,
            world_size,
            config: BackendConfig::default(),
            collectives: Arc::new(Mutex::new(collectives)),
        })
    }

    /// Create with custom configuration.
    pub fn with_config(backend: Backend, rank: usize, world_size: usize, config: BackendConfig) -> Result<Self> {
        let mut pg = Self::new(backend, rank, world_size)?;
        pg.config = config;
        Ok(pg)
    }

    /// Get the backend.
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get this process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Check if this is the main process (rank 0).
    pub fn is_main(&self) -> bool {
        self.rank == 0
    }

    /// Broadcast tensor from root to all processes.
    pub fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        self.collectives.lock().broadcast(tensor, root)
    }

    /// All-reduce: reduce tensor across all processes and distribute result.
    pub fn all_reduce(&self, tensor: &mut Tensor, op: ReduceOp) -> Result<()> {
        self.collectives.lock().all_reduce(tensor, op)
    }

    /// Reduce tensor to destination process.
    pub fn reduce(&self, tensor: &mut Tensor, dst: usize, op: ReduceOp) -> Result<()> {
        self.collectives.lock().reduce(tensor, dst, op)
    }

    /// All-gather: gather tensors from all processes.
    pub fn all_gather(&self, output: &mut [Tensor], input: &Tensor) -> Result<()> {
        self.collectives.lock().all_gather(output, input)
    }

    /// Gather tensors to destination process.
    pub fn gather(&self, output: Option<&mut [Tensor]>, input: &Tensor, dst: usize) -> Result<()> {
        self.collectives.lock().gather(output, input, dst)
    }

    /// Scatter tensors from source process.
    pub fn scatter(&self, output: &mut Tensor, input: Option<&[Tensor]>, src: usize) -> Result<()> {
        self.collectives.lock().scatter(output, input, src)
    }

    /// Barrier synchronization.
    pub fn barrier(&self) -> Result<()> {
        self.collectives.lock().barrier()
    }
}

/// Create a new process group (sub-group of world).
pub fn new_group(ranks: &[usize]) -> Result<ProcessGroup> {
    let world_size = ranks.len();
    // Find our rank in the new group
    let current_rank = crate::get_rank();
    let new_rank = ranks.iter()
        .position(|&r| r == current_rank)
        .ok_or_else(|| DistributedError::InvalidConfiguration(
            "Current process not in group".to_string()
        ))?;
    
    ProcessGroup::new(Backend::Gloo, new_rank, world_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_group_creation() {
        let pg = ProcessGroup::new(Backend::Gloo, 0, 4).unwrap();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 4);
        assert!(pg.is_main());
    }

    #[test]
    fn test_invalid_rank() {
        let result = ProcessGroup::new(Backend::Gloo, 5, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_barrier() {
        let pg = ProcessGroup::new(Backend::Gloo, 0, 1).unwrap();
        assert!(pg.barrier().is_ok());
    }
}
