//! Reference-counted storage for tensor data.
//!
//! Storage provides the underlying memory management for tensors:
//!
//! - **Reference counting**: Multiple tensors can share the same storage (views)
//! - **Device-aware**: Storage knows which device its data lives on
//! - **Type-erased**: Raw bytes, interpreted by tensors based on dtype
//!
//! ## Memory Model
//!
//! ```text
//! Tensor A ─────┐
//!               │
//! Tensor B ─────┼───► Storage (Arc) ───► [data bytes...]
//!               │      │
//! Tensor C ─────┘      └─► Device, size, capacity
//! ```
//!
//! When a tensor needs to modify shared storage, it must first call
//! [`Storage::make_unique`] to ensure exclusive ownership (copy-on-write).

use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::device::{CpuAllocator, Device, DeviceAllocator};
use crate::error::{FerrumError, Result};

/// Raw storage buffer for tensor data.
///
/// This is the inner type wrapped by `Arc` in [`Storage`].
struct StorageInner {
    /// Pointer to allocated memory.
    ptr: NonNull<u8>,
    /// Size in bytes.
    size: usize,
    /// Capacity in bytes (may be larger than size for growth).
    capacity: usize,
    /// Device where data resides.
    device: Device,
}

// SAFETY: StorageInner manages raw memory that is valid and properly aligned.
// The pointer is not accessed without proper synchronization (via RwLock in Storage).
unsafe impl Send for StorageInner {}
unsafe impl Sync for StorageInner {}

impl StorageInner {
    /// Allocate new storage on the given device.
    fn new(size: usize, device: Device) -> Result<Self> {
        if size == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                size: 0,
                capacity: 0,
                device,
            });
        }

        let ptr = match device {
            Device::Cpu => {
                let allocator = CpuAllocator;
                // SAFETY: We're requesting a valid size with proper alignment
                unsafe { allocator.allocate(size) }
            }
            _ => {
                return Err(FerrumError::NotImplemented {
                    feature: format!("Storage allocation on {:?}", device),
                })
            }
        };

        let ptr = NonNull::new(ptr).ok_or(FerrumError::AllocationError {
            bytes: size,
            device,
        })?;

        Ok(Self {
            ptr,
            size,
            capacity: size,
            device,
        })
    }

    /// Create storage from existing data (takes ownership).
    ///
    /// # Safety
    ///
    /// - `data` must be a valid allocation of at least `size` bytes
    /// - `data` must have been allocated by the appropriate allocator for `device`
    /// - Caller transfers ownership of the memory to this storage
    #[allow(dead_code)]
    unsafe fn from_raw(data: *mut u8, size: usize, device: Device) -> Self {
        Self {
            ptr: NonNull::new(data).unwrap_or(NonNull::dangling()),
            size,
            capacity: size,
            device,
        }
    }

    /// Get a raw pointer to the data.
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Get a mutable raw pointer to the data.
    #[inline]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Zero-fill the storage.
    fn zero_fill(&mut self) {
        if self.size == 0 {
            return;
        }

        match self.device {
            Device::Cpu => {
                // SAFETY: We own this memory and it's valid for self.size bytes
                unsafe {
                    std::ptr::write_bytes(self.as_mut_ptr(), 0, self.size);
                }
            }
            _ => {
                // TODO: Implement for other devices
            }
        }
    }

    /// Copy data from a slice.
    fn copy_from_slice(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.size {
            return Err(FerrumError::InternalError {
                message: format!(
                    "Copy size mismatch: expected {} bytes, got {}",
                    self.size,
                    data.len()
                ),
            });
        }

        match self.device {
            Device::Cpu => {
                // SAFETY: We've verified the sizes match
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), self.as_mut_ptr(), self.size);
                }
            }
            _ => {
                return Err(FerrumError::NotImplemented {
                    feature: format!("Copy to {:?}", self.device),
                });
            }
        }

        Ok(())
    }

    /// Clone the storage data.
    fn clone_data(&self) -> Result<Self> {
        let mut new_storage = Self::new(self.size, self.device)?;

        if self.size > 0 {
            match self.device {
                Device::Cpu => {
                    // SAFETY: Both storages have the same size
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            self.as_ptr(),
                            new_storage.as_mut_ptr(),
                            self.size,
                        );
                    }
                }
                _ => {
                    return Err(FerrumError::NotImplemented {
                        feature: format!("Clone on {:?}", self.device),
                    });
                }
            }
        }

        Ok(new_storage)
    }
}

impl Drop for StorageInner {
    fn drop(&mut self) {
        if self.capacity == 0 {
            return;
        }

        match self.device {
            Device::Cpu => {
                let allocator = CpuAllocator;
                // SAFETY: This memory was allocated by CpuAllocator with the same capacity
                unsafe {
                    allocator.deallocate(self.ptr.as_ptr(), self.capacity);
                }
            }
            _ => {
                // TODO: Implement for other devices
                // For now, this would leak memory on unsupported devices,
                // but we prevent allocation on them anyway.
            }
        }
    }
}

/// Reference-counted tensor storage.
///
/// Storage is the backing memory for one or more tensors. It provides:
/// - Thread-safe reference counting (via Arc)
/// - Read/write locking for safe concurrent access
/// - Copy-on-write semantics via `make_unique`
///
/// ## Example
///
/// ```rust
/// use ferrum_core::storage::Storage;
/// use ferrum_core::device::Device;
///
/// let storage = Storage::zeros(1024, Device::Cpu).unwrap();
/// assert_eq!(storage.size(), 1024);
/// ```
#[derive(Clone)]
pub struct Storage {
    inner: Arc<RwLock<StorageInner>>,
}

impl Storage {
    /// Create uninitialized storage of the given size.
    ///
    /// # Warning
    ///
    /// The contents are uninitialized and must be written before reading.
    pub fn uninit(size: usize, device: Device) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(RwLock::new(StorageInner::new(size, device)?)),
        })
    }

    /// Create zero-filled storage.
    pub fn zeros(size: usize, device: Device) -> Result<Self> {
        let storage = Self::uninit(size, device)?;
        storage.inner.write().zero_fill();
        Ok(storage)
    }

    /// Create storage from a byte slice.
    pub fn from_bytes(data: &[u8], device: Device) -> Result<Self> {
        let storage = Self::uninit(data.len(), device)?;
        storage.inner.write().copy_from_slice(data)?;
        Ok(storage)
    }

    /// Create storage from typed data.
    pub fn from_slice<T: bytemuck::Pod>(data: &[T], device: Device) -> Result<Self> {
        let bytes = bytemuck::cast_slice(data);
        Self::from_bytes(bytes, device)
    }

    /// Get the size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.inner.read().size
    }

    /// Get the device.
    #[inline]
    pub fn device(&self) -> Device {
        self.inner.read().device
    }

    /// Check if this storage has a unique reference (not shared).
    #[inline]
    pub fn is_unique(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Get the reference count.
    #[inline]
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Ensure this storage has a unique reference.
    ///
    /// If shared, creates a copy of the data.
    /// Returns `true` if a copy was made.
    pub fn make_unique(&mut self) -> Result<bool> {
        if self.is_unique() {
            return Ok(false);
        }

        let new_inner = self.inner.read().clone_data()?;
        self.inner = Arc::new(RwLock::new(new_inner));
        Ok(true)
    }

    /// Read data as a typed slice.
    ///
    /// # Panics
    ///
    /// Panics if the size is not aligned to `T`.
    pub fn read_as<T: bytemuck::Pod>(&self) -> StorageReadGuard<'_, T> {
        StorageReadGuard {
            guard: self.inner.read(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Write data as a typed slice.
    ///
    /// # Panics
    ///
    /// Panics if the size is not aligned to `T`.
    pub fn write_as<T: bytemuck::Pod>(&self) -> StorageWriteGuard<'_, T> {
        StorageWriteGuard {
            guard: self.inner.write(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Get raw pointer for read access (use with caution).
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The storage is not modified while this pointer is in use
    /// - The pointer is not used after the storage is dropped
    #[inline]
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.inner.read().as_ptr()
    }

    /// Get raw pointer for write access (use with caution).
    ///
    /// # Safety
    ///
    /// Caller must ensure:
    /// - The storage is uniquely owned (call `make_unique` first)
    /// - No other references to the data exist
    /// - The pointer is not used after the storage is dropped
    #[inline]
    pub unsafe fn as_mut_ptr(&self) -> *mut u8 {
        self.inner.write().as_mut_ptr()
    }
}

impl std::fmt::Debug for Storage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.read();
        f.debug_struct("Storage")
            .field("size", &inner.size)
            .field("device", &inner.device)
            .field("ref_count", &Arc::strong_count(&self.inner))
            .finish()
    }
}

/// RAII guard for reading storage as typed data.
pub struct StorageReadGuard<'a, T: bytemuck::Pod> {
    guard: parking_lot::RwLockReadGuard<'a, StorageInner>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> StorageReadGuard<'a, T> {
    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        if self.guard.size == 0 {
            return &[];
        }
        // SAFETY: Storage ensures proper alignment and the lock ensures exclusive access
        unsafe {
            let ptr = self.guard.as_ptr() as *const T;
            let len = self.guard.size / std::mem::size_of::<T>();
            std::slice::from_raw_parts(ptr, len)
        }
    }
}

impl<'a, T: bytemuck::Pod> std::ops::Deref for StorageReadGuard<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

/// RAII guard for writing storage as typed data.
pub struct StorageWriteGuard<'a, T: bytemuck::Pod> {
    guard: parking_lot::RwLockWriteGuard<'a, StorageInner>,
    _marker: std::marker::PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> StorageWriteGuard<'a, T> {
    /// Get the data as a slice.
    pub fn as_slice(&self) -> &[T] {
        if self.guard.size == 0 {
            return &[];
        }
        // SAFETY: Storage ensures proper alignment and the lock ensures exclusive access
        unsafe {
            let ptr = self.guard.as_ptr() as *const T;
            let len = self.guard.size / std::mem::size_of::<T>();
            std::slice::from_raw_parts(ptr, len)
        }
    }

    /// Get the data as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.guard.size == 0 {
            return &mut [];
        }
        // SAFETY: Storage ensures proper alignment and we have exclusive write access
        unsafe {
            let ptr = self.guard.as_mut_ptr() as *mut T;
            let len = self.guard.size / std::mem::size_of::<T>();
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }
}

impl<'a, T: bytemuck::Pod> std::ops::Deref for StorageWriteGuard<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<'a, T: bytemuck::Pod> std::ops::DerefMut for StorageWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_creation() {
        let storage = Storage::zeros(1024, Device::Cpu).unwrap();
        assert_eq!(storage.size(), 1024);
        assert!(storage.device().is_cpu());
    }

    #[test]
    fn test_storage_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let storage = Storage::from_slice(&data, Device::Cpu).unwrap();

        let read = storage.read_as::<f32>();
        assert_eq!(read.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_storage_write() {
        let storage = Storage::zeros(16, Device::Cpu).unwrap();

        {
            let mut write = storage.write_as::<f32>();
            write[0] = 42.0;
            write[1] = 3.14;
        }

        let read = storage.read_as::<f32>();
        assert_eq!(read[0], 42.0);
        assert_eq!(read[1], 3.14);
    }

    #[test]
    fn test_storage_sharing() {
        let storage1 = Storage::zeros(1024, Device::Cpu).unwrap();
        let storage2 = storage1.clone();

        assert_eq!(storage1.ref_count(), 2);
        assert_eq!(storage2.ref_count(), 2);
        assert!(!storage1.is_unique());
    }

    #[test]
    fn test_storage_make_unique() {
        let storage1 = Storage::zeros(16, Device::Cpu).unwrap();
        let mut storage2 = storage1.clone();

        // Write to storage1
        storage1.write_as::<f32>()[0] = 42.0;

        // Make storage2 unique (creates a copy)
        let copied = storage2.make_unique().unwrap();
        assert!(copied);
        assert!(storage2.is_unique());

        // storage2 should have the copied data
        assert_eq!(storage2.read_as::<f32>()[0], 42.0);

        // Modifying storage2 shouldn't affect storage1
        storage2.write_as::<f32>()[0] = 100.0;
        assert_eq!(storage1.read_as::<f32>()[0], 42.0);
        assert_eq!(storage2.read_as::<f32>()[0], 100.0);
    }

    #[test]
    fn test_empty_storage() {
        let storage = Storage::zeros(0, Device::Cpu).unwrap();
        assert_eq!(storage.size(), 0);
        assert_eq!(storage.read_as::<f32>().as_slice().len(), 0);
    }
}
