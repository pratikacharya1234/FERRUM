//! CUDA backend for FERRUM deep learning framework.
//!
//! This module provides GPU acceleration using CUDA. It includes:
//! - Device management and memory allocation
//! - GPU tensor operations
//! - Optimized kernels for common operations
//!
//! # Features
//!
//! - `cuda` - Enable actual CUDA support (requires CUDA toolkit)
//! - `simulate` - Use CPU simulation for testing without GPU
//!
//! # Example
//!
//! ```rust,ignore
//! use ferrum_cuda::CudaDevice;
//!
//! // Initialize CUDA
//! let device = CudaDevice::new(0)?;
//! println!("GPU: {}", device.name());
//!
//! // Allocate memory
//! let buffer = device.alloc(1024 * 1024)?; // 1MB
//! ```

pub mod cuda_device;
pub mod cuda_memory;
pub mod error;
pub mod kernels;
pub mod stream;
pub mod tensor;

pub use cuda_device::{CudaDevice, CudaDeviceManager};
pub use cuda_memory::{CudaBuffer, MemoryPool};
pub use error::{CudaError, CudaResult};
pub use stream::{CudaEvent, CudaStream, StreamPool};
pub use tensor::CudaTensor;

/// Check if CUDA is available on this system.
pub fn cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_device::init_cuda().is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        // In simulate mode, pretend CUDA is available
        #[cfg(feature = "simulate")]
        return true;
        #[cfg(not(feature = "simulate"))]
        return false;
    }
}

/// Get the number of available CUDA devices.
pub fn device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        cuda_device::get_device_count().unwrap_or(0)
    }
    #[cfg(not(feature = "cuda"))]
    {
        #[cfg(feature = "simulate")]
        return 1; // Simulate one GPU
        #[cfg(not(feature = "simulate"))]
        return 0;
    }
}

/// Synchronize all CUDA devices.
pub fn synchronize() -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        cuda_device::synchronize_all()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Ok(())
    }
}
