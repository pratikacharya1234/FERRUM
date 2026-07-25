//! Real CUDA stream implementation using the CUDA Driver API.
//!
//! Streams enable asynchronous GPU operations:
//! - Kernel execution on one stream
//! - H2D memcpy on another
//! - D2H memcpy on a third
//!
//! This overlaps computation and data transfer, hiding latency.

use std::ffi::c_void;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::cuda_device::CudaDevice;
use crate::error::{CudaError, CudaResult};

// ── CUDA Driver API Stream Function Types ─────────────────────────────

type CuStreamCreateFn = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuStreamDestroyFn = unsafe extern "C" fn(u64) -> i32;
type CuStreamSynchronizeFn = unsafe extern "C" fn(u64) -> i32;
type CuStreamQueryFn = unsafe extern "C" fn(u64) -> i32;
type CuStreamWaitEventFn = unsafe extern "C" fn(u64, u64, u32) -> i32;
type CuEventCreateFn = unsafe extern "C" fn(*mut u64, u32) -> i32;
type CuEventDestroyFn = unsafe extern "C" fn(u64) -> i32;
type CuEventRecordFn = unsafe extern "C" fn(u64, u64) -> i32;
type CuEventSynchronizeFn = unsafe extern "C" fn(u64) -> i32;
type CuEventQueryFn = unsafe extern "C" fn(u64) -> i32;
type CuEventElapsedTimeFn = unsafe extern "C" fn(*mut f32, u64, u64) -> i32;
type CuMemcpyAsyncFn = unsafe extern "C" fn(u64, u64, usize, u64) -> i32;

/// Resolve a CUDA Driver API symbol by dynamically loading libcuda.
fn resolve_cuda_symbol<T>(name: &str) -> CudaResult<T> {
    let c_name = std::ffi::CString::new(name).unwrap();
    let lib_name = std::ffi::CString::new("libcuda.so.1").unwrap();
    unsafe {
        let lib = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW);
        if lib.is_null() {
            return Err(CudaError::NotAvailable);
        }
        let ptr = libc::dlsym(lib, c_name.as_ptr());
        libc::dlclose(lib);
        if ptr.is_null() {
            return Err(CudaError::NotAvailable);
        }
        Ok(std::mem::transmute_copy::<*mut c_void, T>(&ptr))
    }
}

// ── CUDA Stream ──────────────────────────────────────────────────────

/// A real CUDA stream for asynchronous GPU operations.
///
/// Streams allow overlapping:
/// - Kernel execution (stream A)
/// - Host→Device memcpy (stream B)
/// - Device→Host memcpy (stream C)
#[derive(Debug)]
pub struct CudaStream {
    /// CUDA stream handle (CUstream).
    handle: u64,
    /// Device this stream belongs to.
    device: Arc<CudaDevice>,
    /// Whether this is the default stream (handle 0).
    is_default: bool,
}

// SAFETY: CUDA streams are valid across threads when the CUDA context is current.
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl CudaStream {
    /// Create a new CUDA stream via `cuStreamCreate`.
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        device.set_current()?;

        let create_fn: CuStreamCreateFn = resolve_cuda_symbol("cuStreamCreate")?;
        let mut handle: u64 = 0;

        // CU_STREAM_DEFAULT = 0
        let result = unsafe { (create_fn)(&mut handle, 0) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuStreamCreate failed: {result}"),
            });
        }

        Ok(Self {
            handle,
            device,
            is_default: false,
        })
    }

    /// Get the default stream (handle 0).
    pub fn default_stream(device: Arc<CudaDevice>) -> Self {
        Self {
            handle: 0,
            device,
            is_default: true,
        }
    }

    /// Get the raw CUDA stream handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Get the device this stream belongs to.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Synchronize the stream (blocks until all queued work completes).
    pub fn synchronize(&self) -> CudaResult<()> {
        if self.handle == 0 {
            // Default stream: synchronize entire device
            return self.device.synchronize();
        }

        let sync_fn: CuStreamSynchronizeFn = resolve_cuda_symbol("cuStreamSynchronize")?;
        let result = unsafe { (sync_fn)(self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuStreamSynchronize failed: {result}"),
            });
        }
        Ok(())
    }

    /// Query whether all operations on this stream have completed.
    pub fn query(&self) -> CudaResult<bool> {
        if self.handle == 0 {
            return Ok(true);
        }

        let query_fn: CuStreamQueryFn = resolve_cuda_symbol("cuStreamQuery")?;
        let result = unsafe { (query_fn)(self.handle) };

        match result {
            0 => Ok(true),   // CUDA_SUCCESS: all operations complete
            600 => Ok(false), // CUDA_ERROR_NOT_READY: operations still pending
            _ => Err(CudaError::DriverError {
                message: format!("cuStreamQuery failed: {result}"),
            }),
        }
    }

    /// Make this stream wait for an event on another stream.
    pub fn wait_event(&self, event: &CudaEvent) -> CudaResult<()> {
        let wait_fn: CuStreamWaitEventFn = resolve_cuda_symbol("cuStreamWaitEvent")?;
        // flags = 0 (default)
        let result = unsafe { (wait_fn)(self.handle, event.handle(), 0) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuStreamWaitEvent failed: {result}"),
            });
        }
        Ok(())
    }

    /// Record an event on this stream.
    pub fn record_event(&self, event: &CudaEvent) -> CudaResult<()> {
        let record_fn: CuEventRecordFn = resolve_cuda_symbol("cuEventRecord")?;
        let result = unsafe { (record_fn)(event.handle(), self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventRecord failed: {result}"),
            });
        }
        Ok(())
    }

    /// Copy memory from host to device asynchronously on this stream.
    pub fn memcpy_h2d_async(&self, dst: u64, src: *const c_void, size: usize) -> CudaResult<()> {
        let copy_fn: CuMemcpyAsyncFn = resolve_cuda_symbol("cuMemcpyHtoDAsync_v2")?;
        let result = unsafe { (copy_fn)(dst, src as u64, size, self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyHtoDAsync failed: {result}"),
            });
        }
        Ok(())
    }

    /// Copy memory from device to host asynchronously on this stream.
    pub fn memcpy_d2h_async(&self, dst: *mut c_void, src: u64, size: usize) -> CudaResult<()> {
        let copy_fn: CuMemcpyAsyncFn = resolve_cuda_symbol("cuMemcpyDtoHAsync_v2")?;
        let result = unsafe { (copy_fn)(dst as u64, src, size, self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyDtoHAsync failed: {result}"),
            });
        }
        Ok(())
    }

    /// Copy memory from device to device asynchronously on this stream.
    pub fn memcpy_d2d_async(&self, dst: u64, src: u64, size: usize) -> CudaResult<()> {
        let copy_fn: CuMemcpyAsyncFn = resolve_cuda_symbol("cuMemcpyDtoDAsync_v2")?;
        let result = unsafe { (copy_fn)(dst, src, size, self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuMemcpyDtoDAsync failed: {result}"),
            });
        }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if !self.is_default && self.handle != 0 {
            // Best-effort cleanup
            if self.device.set_current().is_ok() {
                if let Ok(destroy_fn) = resolve_cuda_symbol::<CuStreamDestroyFn>("cuStreamDestroy_v2") {
                    unsafe { (destroy_fn)(self.handle); };
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CUDA Events
// ═══════════════════════════════════════════════════════════════════════

/// A real CUDA event for GPU synchronization and timing.
#[derive(Debug)]
pub struct CudaEvent {
    /// CUDA event handle (CUevent).
    handle: u64,
    /// Device this event belongs to.
    device: Arc<CudaDevice>,
}

// SAFETY: CUDA events are valid across threads when the CUDA context is current.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Create a new CUDA event via `cuEventCreate`.
    pub fn new(device: Arc<CudaDevice>) -> CudaResult<Self> {
        device.set_current()?;

        let create_fn: CuEventCreateFn = resolve_cuda_symbol("cuEventCreate")?;
        let mut handle: u64 = 0;

        // CU_EVENT_DEFAULT = 0
        let result = unsafe { (create_fn)(&mut handle, 0) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventCreate failed: {result}"),
            });
        }

        Ok(Self { handle, device })
    }

    /// Create a CUDA event with blocking sync (for CPU-side timing).
    pub fn with_blocking_sync(device: Arc<CudaDevice>) -> CudaResult<Self> {
        device.set_current()?;

        let create_fn: CuEventCreateFn = resolve_cuda_symbol("cuEventCreate")?;
        let mut handle: u64 = 0;

        // CU_EVENT_BLOCKING_SYNC = 1
        let result = unsafe { (create_fn)(&mut handle, 1) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventCreate failed: {result}"),
            });
        }

        Ok(Self { handle, device })
    }

    /// Get the event handle.
    pub fn handle(&self) -> u64 {
        self.handle
    }

    /// Check if the event has completed.
    pub fn query(&self) -> CudaResult<bool> {
        let query_fn: CuEventQueryFn = resolve_cuda_symbol("cuEventQuery")?;
        let result = unsafe { (query_fn)(self.handle) };

        match result {
            0 => Ok(true),   // CUDA_SUCCESS
            600 => Ok(false), // CUDA_ERROR_NOT_READY
            _ => Err(CudaError::DriverError {
                message: format!("cuEventQuery failed: {result}"),
            }),
        }
    }

    /// Wait (block) for the event to complete.
    pub fn synchronize(&self) -> CudaResult<()> {
        let sync_fn: CuEventSynchronizeFn = resolve_cuda_symbol("cuEventSynchronize")?;
        let result = unsafe { (sync_fn)(self.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventSynchronize failed: {result}"),
            });
        }
        Ok(())
    }

    /// Get elapsed time between this event and another, in milliseconds.
    pub fn elapsed_time(&self, end: &CudaEvent) -> CudaResult<f32> {
        let elapsed_fn: CuEventElapsedTimeFn = resolve_cuda_symbol("cuEventElapsedTime")?;
        let mut ms: f32 = 0.0;

        let result = unsafe { (elapsed_fn)(&mut ms, self.handle, end.handle) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventElapsedTime failed: {result}"),
            });
        }
        Ok(ms)
    }

    /// Record this event on a stream.
    pub fn record(&self, stream: &CudaStream) -> CudaResult<()> {
        let record_fn: CuEventRecordFn = resolve_cuda_symbol("cuEventRecord")?;
        let result = unsafe { (record_fn)(self.handle, stream.handle()) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuEventRecord failed: {result}"),
            });
        }
        Ok(())
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        if self.device.set_current().is_ok() {
            if let Ok(destroy_fn) = resolve_cuda_symbol::<CuEventDestroyFn>("cuEventDestroy_v2") {
                unsafe { (destroy_fn)(self.handle); };
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Stream Pool
// ═══════════════════════════════════════════════════════════════════════

/// A pool of CUDA streams for efficient reuse.
///
/// Streams are expensive to create (CUDA context synchronization),
/// so reusing them from a pool reduces overhead.
pub struct StreamPool {
    device: Arc<CudaDevice>,
    compute_streams: Mutex<Vec<CudaStream>>,
    copy_streams: Mutex<Vec<CudaStream>>,
    max_streams: usize,
}

impl StreamPool {
    /// Create a new stream pool with separate compute and copy streams.
    pub fn new(device: Arc<CudaDevice>, max_streams: usize) -> Self {
        Self {
            device,
            compute_streams: Mutex::new(Vec::new()),
            copy_streams: Mutex::new(Vec::new()),
            max_streams,
        }
    }

    /// Get a compute stream from the pool.
    pub fn get_compute(&self) -> CudaResult<CudaStream> {
        let mut streams = self.compute_streams.lock();
        if let Some(stream) = streams.pop() {
            Ok(stream)
        } else {
            CudaStream::new(self.device.clone())
        }
    }

    /// Return a compute stream to the pool.
    pub fn put_compute(&self, stream: CudaStream) {
        let mut streams = self.compute_streams.lock();
        if streams.len() < self.max_streams {
            streams.push(stream);
        }
    }

    /// Get a copy stream from the pool.
    pub fn get_copy(&self) -> CudaResult<CudaStream> {
        let mut streams = self.copy_streams.lock();
        if let Some(stream) = streams.pop() {
            Ok(stream)
        } else {
            CudaStream::new(self.device.clone())
        }
    }

    /// Return a copy stream to the pool.
    pub fn put_copy(&self, stream: CudaStream) {
        let mut streams = self.copy_streams.lock();
        if streams.len() < self.max_streams {
            streams.push(stream);
        }
    }

    /// Synchronize all streams in the pool.
    pub fn synchronize_all(&self) -> CudaResult<()> {
        for stream in self.compute_streams.lock().iter() {
            stream.synchronize()?;
        }
        for stream in self.copy_streams.lock().iter() {
            stream.synchronize()?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_device::init_cuda;

    #[test]
    fn test_stream_creation() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let stream = CudaStream::new(device).unwrap();
                assert!(stream.handle() > 0);
            }
        }
    }

    #[test]
    fn test_stream_synchronize() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let stream = CudaStream::new(device).unwrap();
                assert!(stream.synchronize().is_ok());
            }
        }
    }

    #[test]
    fn test_event_creation() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let event = CudaEvent::new(device).unwrap();
                assert!(event.handle() > 0);
            }
        }
    }

    #[test]
    fn test_event_record_and_query() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let stream = CudaStream::new(device.clone()).unwrap();
                let event = CudaEvent::new(device).unwrap();

                event.record(&stream).unwrap();
                stream.synchronize().unwrap();
                assert!(event.query().unwrap());
            }
        }
    }

    #[test]
    fn test_stream_pool() {
        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let pool = StreamPool::new(device, 4);

                let compute = pool.get_compute().unwrap();
                pool.put_compute(compute);

                let copy = pool.get_copy().unwrap();
                pool.put_copy(copy);

                assert!(pool.synchronize_all().is_ok());
            }
        }
    }
}
