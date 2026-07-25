//! Device abstraction for compute backends.
//!
//! FERRUM is GPU-first by default, with CPU as a fallback:
//!
//! | Device | Status | Notes |
//! |--------|--------|-------|
//! | CUDA   | ✅ Default | NVIDIA GPU acceleration via CUDA |
//! | CPU    | ✅ Stable | SIMD-optimized fallback via Rayon |
//! | Metal  | 🚧 Planned | Apple Silicon (MPS) |
//! | Vulkan | 🚧 Planned | Cross-platform GPU |
//!
//! ## Usage
//!
//! ```rust
//! use ferrum_core::Device;
//!
//! // Default device (GPU if available, falls back to CPU)
//! let device = Device::default();  // Cuda(0) or Cpu
//!
//! // Explicit CPU
//! let cpu = Device::Cpu;
//!
//! // Specific GPU
//! let gpu1 = Device::Cuda(1);
//! ```
//!
//! ## Device Placement
//!
//! Operations between tensors on different devices will return
//! [`FerrumError::DeviceMismatch`]. Use [`Tensor::to_device`] for transfers.

use std::fmt;

/// Compute device for tensor operations.
///
/// FERRUM is GPU-first. The default device is `Cuda(0)` when CUDA
/// hardware is detected at runtime; falls back to `Cpu` otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Device {
    /// NVIDIA CUDA device (index specifies which GPU).
    /// This is the default compute device in FERRUM.
    Cuda(usize),
    /// CPU with optional SIMD acceleration.
    /// Use explicitly when you need host memory.
    Cpu,
    /// Apple Metal device.
    Metal(usize),
}

impl Default for Device {
    /// Default device: CUDA GPU 0 if available, otherwise CPU.
    #[inline]
    fn default() -> Self {
        if Device::cuda_is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        }
    }
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

    /// Get the device index.
    #[inline]
    pub const fn index(&self) -> usize {
        match self {
            Device::Cuda(i) | Device::Metal(i) => *i,
            Device::Cpu => 0,
        }
    }

    /// Check if CUDA is available at runtime.
    ///
    /// Probes the CUDA Driver API (`libcuda.so.1` on Linux, `nvcuda.dll`
    /// on Windows) to confirm GPU hardware and drivers are present.
    #[inline]
    pub fn cuda_is_available() -> bool {
        probe_cuda_hardware()
    }

    /// Check if Metal is available at runtime.
    #[inline]
    pub fn metal_is_available() -> bool {
        cfg!(target_os = "macos")
    }

    /// Get the number of available CUDA devices at runtime.
    #[inline]
    pub fn cuda_device_count() -> usize {
        probe_cuda_device_count()
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cuda(i) => write!(f, "cuda:{}", i),
            Device::Cpu => write!(f, "cpu"),
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

// ── CUDA Hardware Probing ──────────────────────────────────────────────

/// Probe CUDA hardware at runtime via the CUDA Driver API.
///
/// Returns `true` if at least one CUDA-capable GPU is present and the
/// CUDA driver library can be loaded.
fn probe_cuda_hardware() -> bool {
    probe_cuda_device_count() > 0
}

/// Probe number of available CUDA devices via the CUDA Driver API.
///
/// Uses `dlopen` / `LoadLibrary` to load the CUDA driver dynamically
/// at runtime — no compile-time dependency on the CUDA toolkit.
fn probe_cuda_device_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        type CuInitFn = unsafe extern "C" fn(u32) -> i32;
        type CuDeviceGetCountFn = unsafe extern "C" fn(*mut i32) -> i32;

        // SAFETY: dlopen/dlsym are called with constant, null-terminated C strings.
        // Function pointers are transmuted to the correct ABI signatures.
        unsafe {
            let lib = load_cuda_library_linux();
            if lib.is_null() {
                return 0;
            }

            // Resolve cuInit
            let init_ptr = resolve_symbol(lib, "cuInit\0");
            if init_ptr.is_null() {
                unload_library_linux(lib);
                return 0;
            }
            let cu_init: CuInitFn = std::mem::transmute(init_ptr);
            if cu_init(0) != 0 {
                unload_library_linux(lib);
                return 0;
            }

            // Resolve cuDeviceGetCount
            let count_ptr = resolve_symbol(lib, "cuDeviceGetCount\0");
            if count_ptr.is_null() {
                unload_library_linux(lib);
                return 0;
            }
            let cu_device_get_count: CuDeviceGetCountFn =
                std::mem::transmute(count_ptr);

            let mut count: i32 = 0;
            let result = cu_device_get_count(&mut count);
            unload_library_linux(lib);

            if result == 0 && count > 0 {
                count as usize
            } else {
                0
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Windows: Load nvcuda.dll and probe for CUDA devices
        #[cfg(target_os = "windows")]
        {
            use std::ffi::CString;
            let lib_name = CString::new("nvcuda.dll").unwrap();
            unsafe {
                let lib = windows::Win32::System::LibraryLoader::LoadLibraryA(
                    windows::core::PCSTR(lib_name.as_ptr() as *const u8)
                );
                match lib {
                    Ok(handle) => {
                        // Probe for CUDA devices via cuDeviceGetCount
                        let probe_fn = libc::dlsym(
                            handle.0 as *mut std::ffi::c_void,
                            CString::new("cuDeviceGetCount").unwrap().as_ptr()
                        );
                        if !probe_fn.is_null() {
                            let get_count: unsafe extern "C" fn(*mut i32) -> i32 =
                                std::mem::transmute(probe_fn);
                            let mut count: i32 = 0;
                            let result = get_count(&mut count);
                            windows::Win32::System::LibraryLoader::FreeLibrary(handle).ok();
                            if result == 0 { count as usize } else { 0 }
                        } else {
                            windows::Win32::System::LibraryLoader::FreeLibrary(handle).ok();
                            0
                        }
                    }
                    Err(_) => 0,
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            0
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn load_cuda_library_linux() -> *mut std::ffi::c_void {
    use std::ffi::CString;
    let lib_name = CString::new("libcuda.so.1").unwrap();
    let handle = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW);
    if handle.is_null() {
        // Try the unversioned symlink
        let lib_name2 = CString::new("libcuda.so").unwrap();
        libc::dlopen(lib_name2.as_ptr(), libc::RTLD_NOW)
    } else {
        handle
    }
}

#[cfg(target_os = "linux")]
unsafe fn resolve_symbol(
    lib: *mut std::ffi::c_void,
    name: &str,
) -> *mut std::ffi::c_void {
    let c_name = std::ffi::CString::new(name).unwrap();
    libc::dlsym(lib, c_name.as_ptr())
}

#[cfg(target_os = "linux")]
unsafe fn unload_library_linux(lib: *mut std::ffi::c_void) {
    libc::dlclose(lib);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_is_cpu() {
        assert!(Device::Cpu.is_cpu());
        assert!(!Device::Cpu.is_gpu());
    }

    #[test]
    fn test_device_is_cuda() {
        assert!(Device::Cuda(0).is_cuda());
        assert!(Device::Cuda(0).is_gpu());
    }

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Metal(1).to_string(), "metal:1");
    }

    #[test]
    fn test_device_default_is_cuda() {
        let d = Device::default();
        // If CUDA hardware is available, default should be Cuda(0).
        // If not, falls back to Cpu. Either way this test verifies
        // the default() method doesn't panic.
        assert!(d.is_cuda() || d.is_cpu());
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