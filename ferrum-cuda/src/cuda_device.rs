//! CUDA device management.

use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::RwLock;
use once_cell::sync::Lazy;

use crate::error::{CudaError, CudaResult};

/// Global CUDA initialization state.
static CUDA_INITIALIZED: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));

/// Current device per thread.
thread_local! {
    static CURRENT_DEVICE: AtomicUsize = const { AtomicUsize::new(0) };
}

/// Properties of a CUDA device.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name.
    pub name: String,
    /// Total global memory in bytes.
    pub total_memory: usize,
    /// Number of multiprocessors.
    pub multiprocessor_count: usize,
    /// CUDA compute capability major version.
    pub compute_major: u32,
    /// CUDA compute capability minor version.
    pub compute_minor: u32,
    /// Maximum threads per block.
    pub max_threads_per_block: usize,
    /// Maximum block dimensions.
    pub max_block_dim: [usize; 3],
    /// Maximum grid dimensions.
    pub max_grid_dim: [usize; 3],
    /// Warp size.
    pub warp_size: usize,
    /// Shared memory per block in bytes.
    pub shared_memory_per_block: usize,
    /// Whether unified memory is supported.
    pub unified_memory: bool,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        // Default simulated GPU properties (like RTX 3080)
        Self {
            name: "Simulated CUDA Device".to_string(),
            total_memory: 10 * 1024 * 1024 * 1024, // 10GB
            multiprocessor_count: 68,
            compute_major: 8,
            compute_minor: 6,
            max_threads_per_block: 1024,
            max_block_dim: [1024, 1024, 64],
            max_grid_dim: [2147483647, 65535, 65535],
            warp_size: 32,
            shared_memory_per_block: 49152,
            unified_memory: true,
        }
    }
}

/// CUDA device handle.
#[derive(Debug)]
pub struct CudaDevice {
    /// Device ID.
    id: usize,
    /// Device properties.
    properties: DeviceProperties,
    /// Allocated memory tracking.
    allocated_memory: AtomicUsize,
}

impl CudaDevice {
    /// Create a new CUDA device handle.
    pub fn new(device_id: usize) -> CudaResult<Self> {
        init_cuda()?;
        
        let count = get_device_count()?;
        if device_id >= count {
            return Err(CudaError::InvalidDevice {
                device_id,
                available: count,
            });
        }

        let properties = get_device_properties(device_id)?;
        
        Ok(Self {
            id: device_id,
            properties,
            allocated_memory: AtomicUsize::new(0),
        })
    }

    /// Get the device ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the device name.
    pub fn name(&self) -> &str {
        &self.properties.name
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Get total memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.properties.total_memory
    }

    /// Get currently allocated memory in bytes.
    pub fn allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }

    /// Get free memory in bytes.
    pub fn free_memory(&self) -> usize {
        self.properties.total_memory.saturating_sub(self.allocated_memory())
    }

    /// Set this device as the current device.
    pub fn set_current(&self) -> CudaResult<()> {
        set_device(self.id)
    }

    /// Synchronize this device.
    pub fn synchronize(&self) -> CudaResult<()> {
        synchronize_device(self.id)
    }

    /// Track memory allocation.
    pub(crate) fn track_alloc(&self, bytes: usize) {
        self.allocated_memory.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Track memory deallocation.
    pub(crate) fn track_free(&self, bytes: usize) {
        self.allocated_memory.fetch_sub(bytes, Ordering::Relaxed);
    }
}

/// Initialize CUDA subsystem.
pub fn init_cuda() -> CudaResult<()> {
    let mut initialized = CUDA_INITIALIZED.write();
    if *initialized {
        return Ok(());
    }

    #[cfg(feature = "cuda")]
    {
        // Real CUDA initialization would go here
        // cuInit(0);
        todo!("Real CUDA initialization")
    }

    #[cfg(feature = "simulate")]
    {
        // Simulated initialization always succeeds
        *initialized = true;
        Ok(())
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        Err(CudaError::NotAvailable)
    }
}

/// Get the number of available CUDA devices.
pub fn get_device_count() -> CudaResult<usize> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA would call cuDeviceGetCount
        todo!("Real CUDA device count")
    }

    #[cfg(feature = "simulate")]
    {
        Ok(1) // Simulate one GPU
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        Err(CudaError::NotAvailable)
    }
}

/// Get device properties.
pub fn get_device_properties(device_id: usize) -> CudaResult<DeviceProperties> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA would call cuDeviceGetAttribute
        let _ = device_id;
        todo!("Real CUDA device properties")
    }

    #[cfg(feature = "simulate")]
    {
        let mut props = DeviceProperties::default();
        props.name = format!("Simulated CUDA Device {}", device_id);
        Ok(props)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = device_id;
        Err(CudaError::NotAvailable)
    }
}

/// Set the current CUDA device.
pub fn set_device(device_id: usize) -> CudaResult<()> {
    let count = get_device_count()?;
    if device_id >= count {
        return Err(CudaError::InvalidDevice {
            device_id,
            available: count,
        });
    }

    CURRENT_DEVICE.with(|d| d.store(device_id, Ordering::Relaxed));

    #[cfg(feature = "cuda")]
    {
        // Real CUDA would call cuCtxSetCurrent
    }

    Ok(())
}

/// Get the current CUDA device.
pub fn get_device() -> usize {
    CURRENT_DEVICE.with(|d| d.load(Ordering::Relaxed))
}

/// Synchronize a specific device.
pub fn synchronize_device(device_id: usize) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Real CUDA would call cuCtxSynchronize
        let _ = device_id;
        todo!("Real CUDA synchronize")
    }

    #[cfg(feature = "simulate")]
    {
        let _ = device_id;
        Ok(()) // Simulation is always synchronized
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = device_id;
        Err(CudaError::NotAvailable)
    }
}

/// Synchronize all devices.
pub fn synchronize_all() -> CudaResult<()> {
    let count = get_device_count()?;
    for i in 0..count {
        synchronize_device(i)?;
    }
    Ok(())
}

/// Device manager for handling multiple GPUs.
pub struct CudaDeviceManager {
    devices: RwLock<Vec<CudaDevice>>,
}

impl CudaDeviceManager {
    /// Create a new device manager.
    pub fn new() -> CudaResult<Self> {
        Ok(Self {
            devices: RwLock::new(Vec::new()),
        })
    }

    /// Initialize all available devices.
    pub fn initialize(&self) -> CudaResult<()> {
        init_cuda()?;
        let count = get_device_count()?;
        let mut devices = self.devices.write();
        
        for i in 0..count {
            devices.push(CudaDevice::new(i)?);
        }
        
        Ok(())
    }

    /// Get the number of devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }

    /// Set the current device.
    pub fn set_device(&self, device_id: usize) -> CudaResult<()> {
        set_device(device_id)
    }

    /// Get the current device ID.
    pub fn current_device(&self) -> usize {
        get_device()
    }

    /// Synchronize the current device.
    pub fn synchronize(&self) -> CudaResult<()> {
        synchronize_device(get_device())
    }

    /// Get a reference to a device.
    pub fn get_device(&self, device_id: usize) -> CudaResult<CudaDevice> {
        let devices = self.devices.read();
        if device_id >= devices.len() {
            return Err(CudaError::InvalidDevice {
                device_id,
                available: devices.len(),
            });
        }
        // Clone the device since we can't return a reference through RwLock
        Ok(CudaDevice::new(device_id)?)
    }
}

impl Default for CudaDeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create device manager")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "simulate")]
    fn test_device_creation() {
        let device = CudaDevice::new(0).unwrap();
        assert_eq!(device.id(), 0);
        assert!(device.total_memory() > 0);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_device_count() {
        let count = get_device_count().unwrap();
        assert!(count >= 1);
    }
}
