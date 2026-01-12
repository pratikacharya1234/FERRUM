//! CUDA error types.

use std::fmt;

/// Result type for CUDA operations.
pub type CudaResult<T> = Result<T, CudaError>;

/// CUDA-specific errors.
#[derive(Debug)]
pub enum CudaError {
    /// CUDA is not available on this system.
    NotAvailable,
    /// Invalid device ID.
    InvalidDevice { device_id: usize, available: usize },
    /// Out of GPU memory.
    OutOfMemory { requested: usize, available: usize },
    /// Kernel launch failed.
    KernelLaunchFailed { name: &'static str, message: String },
    /// Invalid memory access.
    InvalidMemoryAccess { address: usize },
    /// Synchronization failed.
    SyncFailed { message: String },
    /// Driver error with code.
    DriverError { code: i32, message: String },
    /// Stream error.
    StreamError { message: String },
    /// Invalid argument.
    InvalidArgument { message: String },
    /// Not implemented.
    NotImplemented { feature: String },
    /// Generic error.
    Other { message: String },
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaError::NotAvailable => write!(f, "CUDA is not available on this system"),
            CudaError::InvalidDevice { device_id, available } => {
                write!(f, "Invalid device ID {}, only {} devices available", device_id, available)
            }
            CudaError::OutOfMemory { requested, available } => {
                write!(f, "Out of GPU memory: requested {} bytes, {} available", requested, available)
            }
            CudaError::KernelLaunchFailed { name, message } => {
                write!(f, "Kernel '{}' launch failed: {}", name, message)
            }
            CudaError::InvalidMemoryAccess { address } => {
                write!(f, "Invalid GPU memory access at 0x{:x}", address)
            }
            CudaError::SyncFailed { message } => {
                write!(f, "CUDA synchronization failed: {}", message)
            }
            CudaError::DriverError { code, message } => {
                write!(f, "CUDA driver error ({}): {}", code, message)
            }
            CudaError::StreamError { message } => {
                write!(f, "CUDA stream error: {}", message)
            }
            CudaError::InvalidArgument { message } => {
                write!(f, "Invalid argument: {}", message)
            }
            CudaError::NotImplemented { feature } => {
                write!(f, "Feature not implemented: {}", feature)
            }
            CudaError::Other { message } => write!(f, "CUDA error: {}", message),
        }
    }
}

impl std::error::Error for CudaError {}

impl From<CudaError> for ferrum_core::FerrumError {
    fn from(err: CudaError) -> Self {
        ferrum_core::FerrumError::InternalError {
            message: format!("CUDA error: {}", err),
        }
    }
}
