//! CUDA backend for the FERRUM deep learning framework.
//!
//! FERRUM is GPU-first by design. This crate provides the CUDA backend:
//! GPU device management, memory allocation, kernel execution, and
//! stream-based asynchronous operations.
//!
//! # Architecture
//!
//! FERRUM links directly against the CUDA Driver API (`libcuda.so.1`
//! on Linux, `nvcuda.dll` on Windows) via FFI — no CUDA toolkit
//! required at compile time. The driver API is loaded dynamically
//! at runtime (`dlopen`/`dlsym`), so the same binary works on
//! machines with and without NVIDIA GPUs.
//!
//! # Example
//!
//! ```rust,ignore
//! use ferrum_cuda::{CudaDevice, CudaDeviceManager};
//!
//! // Initialize CUDA and enumerate GPUs
//! let manager = CudaDeviceManager::new()?;
//! manager.initialize()?;
//! println!("Found {} CUDA device(s)", manager.device_count());
//!
//! // Allocate GPU memory
//! let device = manager.get_device(0)?;
//! let buffer = device.alloc(1024 * 1024)?; // 1 MB on GPU
//! ```

pub mod backend;
pub mod cuda_device;
pub mod cuda_memory;
pub mod cublas;
#[cfg(feature = "cudarc")]
pub mod cudarc_backend;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod kernel_launch;
pub mod kernels;
pub mod profiler;
pub mod stream;
pub mod tensor;

pub use backend::{register_gpu_backend, CudaComputeBackend};
pub use cuda_device::{CudaDevice, CudaDeviceManager, DeviceProperties};
pub use cuda_memory::{CudaBuffer, MemoryPool};
pub use cublas::{CublasHandle, cublas_matmul_f32, cublas_matmul_f16};
pub use error::{CudaError, CudaResult};
pub use fusion::{FusionEngine, FusionGraph, FusionOp, CompiledKernel, NvrtcCompiler};
pub use graph::{CudaGraph, GraphRunner};
pub use profiler::{Profiler, ProfilerConfig, KernelProfile};
pub use stream::{CudaEvent, CudaStream, StreamPool};
pub use tensor::CudaTensor;

/// Check if CUDA is available on this system.
///
/// This probes the CUDA Driver API at runtime via FFI. No
/// compile-time CUDA toolkit dependency is required.
pub fn cuda_available() -> bool {
    cuda_device::init_cuda().is_ok()
}

/// Get the number of available CUDA devices.
pub fn device_count() -> usize {
    cuda_device::get_device_count().unwrap_or(0)
}

/// Synchronize all CUDA devices (blocks until all GPU work completes).
pub fn synchronize() -> CudaResult<()> {
    cuda_device::synchronize_all()
}