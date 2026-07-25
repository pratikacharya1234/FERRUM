//! CUDA GPU memory management via the CUDA Driver API.
//!
//! All GPU memory allocation (`cuMemAlloc`, `cuMemFree`) and data
//! transfer (`cuMemcpyHtoD`, `cuMemcpyDtoH`, `cuMemcpyDtoD`) is
//! performed through the dynamically-loaded Driver API.

use std::ffi::c_void;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::cuda_device::CudaDevice;
use crate::error::{CudaError, CudaResult};

// ── CUDA Driver API Memory Function Types ─────────────────────────────

type CuMemAllocFn = unsafe extern "C" fn(*mut u64, usize) -> i32;
type CuMemFreeFn = unsafe extern "C" fn(u64) -> i32;
type CuMemcpyHtoDFn = unsafe extern "C" fn(u64, *const c_void, usize) -> i32;
type CuMemcpyDtoHFn = unsafe extern "C" fn(*mut c_void, u64, usize) -> i32;
type CuMemcpyDtoDFn = unsafe extern "C" fn(u64, u64, usize) -> i32;
type CuMemsetD8Fn = unsafe extern "C" fn(u64, u8, usize) -> i32;

struct MemFns {
    alloc: CuMemAllocFn,
    free: CuMemFreeFn,
    hto_d: CuMemcpyHtoDFn,
    dto_h: CuMemcpyDtoHFn,
    dto_d: CuMemcpyDtoDFn,
    memset: CuMemsetD8Fn,
}

static MEM_FNS: OnceLock<MemFns> = OnceLock::new();

fn get_mem_fns() -> CudaResult<&'static MemFns> {
    if let Some(fns) = MEM_FNS.get() {
        return Ok(fns);
    }

    let fns = unsafe {
        let lib_name = std::ffi::CString::new("libcuda.so.1").unwrap();
        let lib = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return Err(CudaError::NotAvailable);
        }

        macro_rules! res {
            ($name:expr, $t:ty) => {{
                let c = std::ffi::CString::new($name).unwrap();
                let p = libc::dlsym(lib, c.as_ptr());
                if p.is_null() {
                    return Err(CudaError::NotAvailable);
                }
                std::mem::transmute(p)
            }};
        }

        MemFns {
            alloc: res!("cuMemAlloc_v2", CuMemAllocFn),
            free: res!("cuMemFree_v2", CuMemFreeFn),
            hto_d: res!("cuMemcpyHtoD_v2", CuMemcpyHtoDFn),
            dto_h: res!("cuMemcpyDtoH_v2", CuMemcpyDtoHFn),
            dto_d: res!("cuMemcpyDtoD_v2", CuMemcpyDtoDFn),
            memset: res!("cuMemsetD8_v2", CuMemsetD8Fn),
        }
    };

    let _ = MEM_FNS.set(fns);
    Ok(MEM_FNS.get().unwrap())
}

// ── CudaBuffer ────────────────────────────────────────────────────────

/// A buffer of GPU memory allocated via `cuMemAlloc`.
#[derive(Debug)]
pub struct CudaBuffer {
    /// CUDA device pointer (u64 handle used by Driver API).
    ptr: u64,
    /// Size in bytes.
    size: usize,
    /// Device this buffer belongs to.
    device: Arc<CudaDevice>,
    /// Whether this buffer owns its memory (will call cuMemFree on drop).
    owned: bool,
}

// SAFETY: GPU memory pointers are valid across threads when
// synchronization is handled through CUDA streams.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate a new GPU buffer via `cuMemAlloc`.
    pub fn new(device: Arc<CudaDevice>, size: usize) -> CudaResult<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: 0,
                size: 0,
                device,
                owned: true,
            });
        }

        // Ensure CUDA is initialized and context is current
        device.set_current()?;

        let fns = get_mem_fns()?;
        let mut dptr: u64 = 0;
        let result = unsafe { (fns.alloc)(&mut dptr, size) };

        if result != 0 || dptr == 0 {
            return Err(CudaError::OutOfMemory {
                requested: size,
                available: device.free_memory(),
            });
        }

        device.track_alloc(size);

        Ok(Self {
            ptr: dptr,
            size,
            device,
            owned: true,
        })
    }

    /// Create a buffer from an existing device pointer (does not take ownership).
    ///
    /// # Safety
    /// `ptr` must be a valid CUDA device pointer allocated via `cuMemAlloc`.
    pub unsafe fn from_ptr(device: Arc<CudaDevice>, ptr: u64, size: usize) -> Self {
        Self {
            ptr,
            size,
            device,
            owned: false,
        }
    }

    /// Get the raw CUDA device pointer.
    pub fn ptr(&self) -> u64 {
        self.ptr
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device this buffer is allocated on.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Copy data from host memory to GPU (H2D transfer via `cuMemcpyHtoD`).
    pub fn copy_from_host(&mut self, data: &[u8]) -> CudaResult<()> {
        if data.len() > self.size {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Data size {} exceeds buffer size {}",
                    data.len(),
                    self.size
                ),
            });
        }

        self.device.set_current()?;

        let fns = get_mem_fns()?;
        let result = unsafe {
            (fns.hto_d)(self.ptr, data.as_ptr() as *const c_void, data.len())
        };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyHtoD failed with code {result}"),
            });
        }
        Ok(())
    }

    /// Copy data from GPU to host memory (D2H transfer via `cuMemcpyDtoH`).
    pub fn copy_to_host(&self, data: &mut [u8]) -> CudaResult<()> {
        if data.len() > self.size {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Data size {} exceeds buffer size {}",
                    data.len(),
                    self.size
                ),
            });
        }

        self.device.set_current()?;

        let fns = get_mem_fns()?;
        let result = unsafe {
            (fns.dto_h)(data.as_mut_ptr() as *mut c_void, self.ptr, data.len())
        };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyDtoH failed with code {result}"),
            });
        }
        Ok(())
    }

    /// Copy data from another GPU buffer (D2D transfer via `cuMemcpyDtoD`).
    pub fn copy_from_device(&mut self, src: &CudaBuffer) -> CudaResult<()> {
        if src.size > self.size {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Source size {} exceeds destination size {}",
                    src.size,
                    self.size
                ),
            });
        }

        self.device.set_current()?;

        let fns = get_mem_fns()?;
        let result = unsafe { (fns.dto_d)(self.ptr, src.ptr, src.size) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyDtoD failed with code {result}"),
            });
        }
        Ok(())
    }

    /// Set all bytes in the buffer to zero on the GPU (via `cuMemsetD8`).
    pub fn zero(&mut self) -> CudaResult<()> {
        if self.size == 0 {
            return Ok(());
        }

        self.device.set_current()?;

        let fns = get_mem_fns()?;
        let result = unsafe { (fns.memset)(self.ptr, 0, self.size) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemsetD8 failed with code {result}"),
            });
        }
        Ok(())
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.owned && self.ptr != 0 {
            if self.device.set_current().is_ok() {
                if let Ok(fns) = get_mem_fns() {
                    let result = unsafe { (fns.free)(self.ptr) };
                    if result == 0 {
                        self.device.track_free(self.size);
                    }
                }
            }
        }
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        let mut new_buffer = CudaBuffer::new(self.device.clone(), self.size)
            .expect("Failed to allocate GPU memory for clone");
        new_buffer
            .copy_from_device(self)
            .expect("Failed to copy GPU memory");
        new_buffer
    }
}

// ── Memory Pool ───────────────────────────────────────────────────────

/// A memory pool for efficient GPU memory allocation.
///
/// Reuses previously freed buffers to reduce `cuMemAlloc`/`cuMemFree`
/// overhead. Buffers are organized by rounded-up bucket sizes.
pub struct MemoryPool {
    device: Arc<CudaDevice>,
    /// Free blocks organized by size (power of 2).
    free_blocks: Mutex<Vec<Vec<CudaBuffer>>>,
    /// Minimum block size (256 bytes).
    min_block_size: usize,
    /// Maximum cached block size (256 MB).
    max_block_size: usize,
}

impl MemoryPool {
    /// Create a new memory pool for the given device.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        let num_buckets = 21; // 256 bytes to 256 MB
        Self {
            device,
            free_blocks: Mutex::new(vec![Vec::new(); num_buckets]),
            min_block_size: 256,
            max_block_size: 256 * 1024 * 1024,
        }
    }

    /// Allocate a buffer from the pool, reusing a cached one if available.
    pub fn alloc(&self, size: usize) -> CudaResult<CudaBuffer> {
        let size = self.round_size(size);
        let bucket = self.size_to_bucket(size);

        // Try to get from free list
        {
            let mut free_blocks = self.free_blocks.lock();
            if let Some(block) = free_blocks[bucket].pop() {
                return Ok(block);
            }
        }

        // Allocate new from GPU
        CudaBuffer::new(self.device.clone(), size)
    }

    /// Return a buffer to the pool for future reuse.
    pub fn free(&self, buffer: CudaBuffer) {
        let bucket = self.size_to_bucket(buffer.size());
        if bucket < self.free_blocks.lock().len() {
            self.free_blocks.lock()[bucket].push(buffer);
        }
        // If too large, just drop it (will call cuMemFree)
    }

    /// Clear all cached memory, freeing GPU allocations.
    pub fn clear(&self) {
        let mut free_blocks = self.free_blocks.lock();
        for bucket in free_blocks.iter_mut() {
            bucket.clear();
        }
    }

    fn round_size(&self, size: usize) -> usize {
        let size = size.max(self.min_block_size);
        size.next_power_of_two()
    }

    fn size_to_bucket(&self, size: usize) -> usize {
        let min_bits = self.min_block_size.ilog2() as usize;
        let size_bits = size.ilog2() as usize;
        size_bits.saturating_sub(min_bits)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_device::init_cuda;

    #[test]
    fn test_buffer_allocation_when_gpu_available() {
        // This test only runs when a GPU is present
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let buffer = CudaBuffer::new(device, 1024).unwrap();
                assert_eq!(buffer.size(), 1024);
            }
        }
    }

    #[test]
    fn test_host_device_copy_when_gpu_available() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let mut buffer = CudaBuffer::new(device, 1024).unwrap();

                let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
                buffer.copy_from_host(&data).unwrap();

                let mut result = vec![0u8; 1024];
                buffer.copy_to_host(&mut result).unwrap();

                assert_eq!(data, result);
            }
        }
    }
}