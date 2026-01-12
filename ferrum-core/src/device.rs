//! Device abstraction for compute backends.
//!
//! FERRUM is designed to support multiple compute backends:
//!
//! | Device | Status | Notes |
//! |--------|--------|-------|
//! | CPU    | ✅ Stable | SIMD-optimized via Rayon |
//! | CUDA   | 🚧 Planned | NVIDIA GPUs |
//! | Metal  | 🚧 Planned | Apple Silicon |
//! | Vulkan | 🚧 Planned | Cross-platform GPU |
//!
//! ## Usage
//!
//! ```rust
//! use ferrum_core::Device;
//!
//! // Default device
//! let device = Device::Cpu;
//!
//! // Future: GPU selection
//! // let device = Device::Cuda(0);  // First CUDA device
//! ```
//!
//! ## Device Placement
//!
//! Operations between tensors on different devices will return
//! [`FerrumError::DeviceMismatch`]. Use [`Tensor::to_device`] for transfers.

use std::fmt;

/// Compute device for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Device {
    /// CPU with optional SIMD acceleration.
    #[default]
    Cpu,
    /// NVIDIA CUDA device (index specifies which GPU).
    #[allow(dead_code)]
    Cuda(usize),
    /// Apple Metal device.
    #[allow(dead_code)]
    Metal(usize),
}

impl Device {
    /// Check if this is a CPU device.
    #[inline]
    pub const fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Check if this is a CUDA device.
    #[inline]
    pub const fn is_cuda(&self) -> bool {
        matches!(self, Device::Cuda(_))
    }

    /// Check if this is a Metal device.
    #[inline]
    pub const fn is_metal(&self) -> bool {
        matches!(self, Device::Metal(_))
    }

    /// Check if this is any GPU device.
    #[inline]
    pub const fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }

    /// Get the device index (0 for CPU).
    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            Device::Cpu => 0,
            Device::Cuda(i) | Device::Metal(i) => *i,
        }
    }

    /// Get the default device (CPU).
    #[inline]
    pub const fn default() -> Self {
        Device::Cpu
    }

    /// Check if CUDA is available at runtime.
    #[inline]
    pub fn cuda_is_available() -> bool {
        // TODO: Implement CUDA detection
        false
    }

    /// Check if Metal is available at runtime.
    #[inline]
    pub fn metal_is_available() -> bool {
        // TODO: Implement Metal detection
        cfg!(target_os = "macos")
    }

    /// Get the number of available CUDA devices.
    #[inline]
    pub fn cuda_device_count() -> usize {
        // TODO: Implement CUDA device enumeration
        0
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "cpu"),
            Device::Cuda(i) => write!(f, "cuda:{}", i),
            Device::Metal(i) => write!(f, "metal:{}", i),
        }
    }
}

/// Trait for device-specific allocators.
///
/// This is used internally by the storage system to allocate memory
/// on different devices.
pub trait DeviceAllocator: Send + Sync {
    /// Allocate `size` bytes of memory.
    ///
    /// # Safety
    ///
    /// The returned pointer must be valid for `size` bytes and properly aligned.
    unsafe fn allocate(&self, size: usize) -> *mut u8;

    /// Deallocate memory previously allocated with `allocate`.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated by this allocator with the same `size`.
    /// - `ptr` must not be used after this call.
    unsafe fn deallocate(&self, ptr: *mut u8, size: usize);

    /// Copy `size` bytes from `src` to `dst`.
    ///
    /// # Safety
    ///
    /// - Both pointers must be valid for `size` bytes.
    /// - Regions may overlap (will use memmove semantics).
    unsafe fn copy(&self, src: *const u8, dst: *mut u8, size: usize);

    /// Zero `size` bytes starting at `ptr`.
    ///
    /// # Safety
    ///
    /// `ptr` must be valid for `size` bytes.
    unsafe fn zero(&self, ptr: *mut u8, size: usize);
}

/// CPU allocator using the system allocator.
pub struct CpuAllocator;

impl DeviceAllocator for CpuAllocator {
    unsafe fn allocate(&self, size: usize) -> *mut u8 {
        if size == 0 {
            return std::ptr::NonNull::dangling().as_ptr();
        }
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        ptr
    }

    unsafe fn deallocate(&self, ptr: *mut u8, size: usize) {
        if size == 0 {
            return;
        }
        let layout = std::alloc::Layout::from_size_align(size, 64).unwrap();
        std::alloc::dealloc(ptr, layout);
    }

    unsafe fn copy(&self, src: *const u8, dst: *mut u8, size: usize) {
        std::ptr::copy(src, dst, size);
    }

    unsafe fn zero(&self, ptr: *mut u8, size: usize) {
        std::ptr::write_bytes(ptr, 0, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_properties() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_gpu());
        assert!(Device::Cuda(0).is_cuda());
        assert!(Device::Cuda(0).is_gpu());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Metal(1).to_string(), "metal:1");
    }

    #[test]
    fn test_cpu_allocator() {
        let allocator = CpuAllocator;
        unsafe {
            let ptr = allocator.allocate(1024);
            assert!(!ptr.is_null());
            allocator.zero(ptr, 1024);
            allocator.deallocate(ptr, 1024);
        }
    }
}
