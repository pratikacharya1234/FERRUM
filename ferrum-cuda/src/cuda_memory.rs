//! CUDA memory management.

use std::sync::Arc;
use parking_lot::Mutex;

use crate::cuda_device::CudaDevice;
use crate::error::{CudaError, CudaResult};

/// A buffer of GPU memory.
#[derive(Debug)]
pub struct CudaBuffer {
    /// Device pointer (or simulated CPU pointer).
    ptr: *mut u8,
    /// Size in bytes.
    size: usize,
    /// Device this buffer belongs to.
    device: Arc<CudaDevice>,
    /// Whether this buffer owns its memory.
    owned: bool,
}

// SAFETY: CudaBuffer is Send + Sync because GPU memory operations
// are synchronized through CUDA streams.
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaBuffer {
    /// Allocate a new GPU buffer.
    pub fn new(device: Arc<CudaDevice>, size: usize) -> CudaResult<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                size: 0,
                device,
                owned: true,
            });
        }

        // Check available memory
        if size > device.free_memory() {
            return Err(CudaError::OutOfMemory {
                requested: size,
                available: device.free_memory(),
            });
        }

        let ptr = alloc_device_memory(size)?;
        device.track_alloc(size);

        Ok(Self {
            ptr,
            size,
            device,
            owned: true,
        })
    }

    /// Create a buffer from existing device pointer (does not take ownership).
    pub unsafe fn from_ptr(device: Arc<CudaDevice>, ptr: *mut u8, size: usize) -> Self {
        Self {
            ptr,
            size,
            device,
            owned: false,
        }
    }

    /// Get the device pointer.
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Get the size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get the device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Copy data from host to device.
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

        copy_host_to_device(data.as_ptr(), self.ptr, data.len())?;
        Ok(())
    }

    /// Copy data from device to host.
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

        copy_device_to_host(self.ptr, data.as_mut_ptr(), data.len())?;
        Ok(())
    }

    /// Copy data from another device buffer.
    pub fn copy_from_device(&mut self, src: &CudaBuffer) -> CudaResult<()> {
        if src.size > self.size {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Source size {} exceeds destination size {}",
                    src.size, self.size
                ),
            });
        }

        copy_device_to_device(src.ptr, self.ptr, src.size)?;
        Ok(())
    }

    /// Set all bytes to zero.
    pub fn zero(&mut self) -> CudaResult<()> {
        memset_device(self.ptr, 0, self.size)?;
        Ok(())
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        if self.owned && !self.ptr.is_null() {
            let _ = free_device_memory(self.ptr);
            self.device.track_free(self.size);
        }
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        let mut new_buffer = CudaBuffer::new(self.device.clone(), self.size)
            .expect("Failed to allocate GPU memory for clone");
        new_buffer.copy_from_device(self)
            .expect("Failed to copy GPU memory");
        new_buffer
    }
}

// ============================================================================
// Memory allocation functions (simulated or real CUDA)
// ============================================================================

fn alloc_device_memory(size: usize) -> CudaResult<*mut u8> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemAlloc
        let _ = size;
        todo!("Real CUDA allocation")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: use aligned CPU memory
        let layout = std::alloc::Layout::from_size_align(size, 256)
            .map_err(|_| CudaError::InvalidArgument {
                message: "Invalid allocation size".to_string(),
            })?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(CudaError::OutOfMemory {
                requested: size,
                available: 0,
            });
        }
        Ok(ptr)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = size;
        Err(CudaError::NotAvailable)
    }
}

fn free_device_memory(ptr: *mut u8) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemFree
        let _ = ptr;
        todo!("Real CUDA free")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: we can't easily free without knowing the size
        // In a real implementation, we'd track this
        // For now, leak (not ideal but safe for testing)
        let _ = ptr;
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = ptr;
        Err(CudaError::NotAvailable)
    }
}

fn copy_host_to_device(src: *const u8, dst: *mut u8, size: usize) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemcpyHtoD
        let _ = (src, dst, size);
        todo!("Real CUDA H2D copy")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: just memcpy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (src, dst, size);
        Err(CudaError::NotAvailable)
    }
}

fn copy_device_to_host(src: *const u8, dst: *mut u8, size: usize) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemcpyDtoH
        let _ = (src, dst, size);
        todo!("Real CUDA D2H copy")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: just memcpy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (src, dst, size);
        Err(CudaError::NotAvailable)
    }
}

fn copy_device_to_device(src: *const u8, dst: *mut u8, size: usize) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemcpyDtoD
        let _ = (src, dst, size);
        todo!("Real CUDA D2D copy")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: just memcpy
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, size);
        }
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (src, dst, size);
        Err(CudaError::NotAvailable)
    }
}

fn memset_device(ptr: *mut u8, value: u8, size: usize) -> CudaResult<()> {
    // For all backends, a zero-size memset is a no-op and does not require a valid pointer.
    if size == 0 {
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        // Real CUDA: cuMemsetD8
        let _ = (ptr, value, size);
        todo!("Real CUDA memset")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated: just memset
        if ptr.is_null() {
            return Err(CudaError::InvalidArgument {
                message: "memset_device called with null pointer and non-zero size".to_string(),
            });
        }
        unsafe {
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (ptr, value, size);
        Err(CudaError::NotAvailable)
    }
}

// ============================================================================
// Memory pool for efficient allocation
// ============================================================================

/// A memory pool for efficient GPU memory allocation.
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
    /// Create a new memory pool.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        let num_buckets = 21; // 256 bytes to 256 MB
        Self {
            device,
            free_blocks: Mutex::new(vec![Vec::new(); num_buckets]),
            min_block_size: 256,
            max_block_size: 256 * 1024 * 1024,
        }
    }

    /// Allocate a buffer from the pool.
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

        // Allocate new
        CudaBuffer::new(self.device.clone(), size)
    }

    /// Return a buffer to the pool.
    pub fn free(&self, buffer: CudaBuffer) {
        let bucket = self.size_to_bucket(buffer.size());
        if bucket < self.free_blocks.lock().len() {
            self.free_blocks.lock()[bucket].push(buffer);
        }
        // If too large, just drop it
    }

    /// Clear all cached memory.
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
        let min_bits = self.min_block_size.trailing_zeros() as usize;
        let size_bits = size.trailing_zeros() as usize;
        size_bits.saturating_sub(min_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "simulate")]
    fn test_buffer_allocation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let buffer = CudaBuffer::new(device, 1024).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_host_device_copy() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let mut buffer = CudaBuffer::new(device, 1024).unwrap();
        
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        buffer.copy_from_host(&data).unwrap();
        
        let mut result = vec![0u8; 1024];
        buffer.copy_to_host(&mut result).unwrap();
        
        assert_eq!(data, result);
    }
}
