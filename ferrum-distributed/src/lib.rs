//! Distributed training support for Ferrum.
//!
//! This module provides abstractions for distributed training, similar to
//! PyTorch's `torch.distributed` module.
//!
//! # Features
//!
//! - Process groups for collective communication
//! - All-reduce, broadcast, scatter, gather operations
//! - Distributed data parallel (DDP) wrapper
//! - Gradient averaging across workers
//!
//! # Example
//!
//! ```ignore
//! use ferrum_distributed::{init_process_group, ProcessGroup, Backend};
//!
//! // Initialize distributed environment
//! let pg = init_process_group(Backend::Gloo, 0, 4).unwrap();
//!
//! // Get rank and world size
//! println!("Rank: {}, World Size: {}", pg.rank(), pg.world_size());
//!
//! // Perform all-reduce on tensors
//! let tensor = Tensor::ones(&[10]);
//! pg.all_reduce(&tensor, ReduceOp::Sum);
//! ```

pub mod backend;
pub mod collectives;
pub mod ddp;
pub mod error;
pub mod process_group;

pub use backend::{Backend, BackendConfig};
pub use collectives::ReduceOp;
pub use ddp::DistributedDataParallel;
pub use error::{DistributedError, Result};
pub use process_group::ProcessGroup;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use parking_lot::Mutex;
use std::sync::Arc;

/// Global state for distributed training.
static INITIALIZED: AtomicBool = AtomicBool::new(false);
static DEFAULT_GROUP: Mutex<Option<Arc<ProcessGroup>>> = Mutex::new(None);

/// Initialize the default process group.
///
/// # Arguments
/// * `backend` - Communication backend to use
/// * `rank` - Rank of this process
/// * `world_size` - Total number of processes
pub fn init_process_group(backend: Backend, rank: usize, world_size: usize) -> Result<Arc<ProcessGroup>> {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(DistributedError::AlreadyInitialized);
    }

    let pg = Arc::new(ProcessGroup::new(backend, rank, world_size)?);
    *DEFAULT_GROUP.lock() = Some(pg.clone());
    Ok(pg)
}

/// Initialize process group from environment variables.
///
/// Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from environment.
pub fn init_process_group_from_env() -> Result<Arc<ProcessGroup>> {
    let rank: usize = std::env::var("RANK")
        .unwrap_or_else(|_| "0".into())
        .parse()
        .map_err(|_| DistributedError::InvalidConfiguration("Invalid RANK".into()))?;
    
    let world_size: usize = std::env::var("WORLD_SIZE")
        .unwrap_or_else(|_| "1".into())
        .parse()
        .map_err(|_| DistributedError::InvalidConfiguration("Invalid WORLD_SIZE".into()))?;
    
    let backend = std::env::var("DIST_BACKEND")
        .unwrap_or_else(|_| "gloo".into());
    
    let backend = match backend.as_str() {
        "nccl" => Backend::Nccl,
        "gloo" => Backend::Gloo,
        "mpi" => Backend::Mpi,
        _ => Backend::Gloo,
    };

    init_process_group(backend, rank, world_size)
}

/// Get the default process group.
pub fn get_world() -> Option<Arc<ProcessGroup>> {
    DEFAULT_GROUP.lock().clone()
}

/// Check if distributed training is initialized.
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// Destroy the default process group and cleanup.
pub fn destroy_process_group() {
    *DEFAULT_GROUP.lock() = None;
    INITIALIZED.store(false, Ordering::SeqCst);
}

/// Get the rank of the current process.
pub fn get_rank() -> usize {
    get_world().map(|pg| pg.rank()).unwrap_or(0)
}

/// Get the world size.
pub fn get_world_size() -> usize {
    get_world().map(|pg| pg.world_size()).unwrap_or(1)
}

/// Check if this is the main process (rank 0).
pub fn is_main_process() -> bool {
    get_rank() == 0
}

/// Barrier synchronization across all processes.
pub fn barrier() -> Result<()> {
    if let Some(pg) = get_world() {
        pg.barrier()
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_destroy() {
        // Clean state
        destroy_process_group();
        
        assert!(!is_initialized());
        
        let pg = init_process_group(Backend::Gloo, 0, 1).unwrap();
        assert!(is_initialized());
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);
        
        destroy_process_group();
        assert!(!is_initialized());
    }

    #[test]
    fn test_get_rank_world_size() {
        destroy_process_group();
        
        assert_eq!(get_rank(), 0);
        assert_eq!(get_world_size(), 1);
        
        init_process_group(Backend::Gloo, 2, 4).unwrap();
        
        assert_eq!(get_rank(), 2);
        assert_eq!(get_world_size(), 4);
        
        destroy_process_group();
    }
}
