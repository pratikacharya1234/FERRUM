//! CUDA Graphs for Static Inference.
//!
//! CUDA Graphs capture a sequence of operations and replay them
//! with minimal CPU overhead. This is ideal for inference where
//! the computation graph is static.
//!
//! # Usage
//!
//! ```text
//! // One-time capture:
//! let graph = CudaGraph::capture(&stream, || {
//!     model.forward(&input);
//! });
//!
//! // Every inference:
//! graph.launch(&stream);
//! ```

use std::ffi::c_void;
use std::sync::Arc;

use crate::cuda_device::CudaDevice;
use crate::error::{CudaError, CudaResult};
use crate::stream::CudaStream;

// ── CUDA Graph Function Types ─────────────────────────────────────────

type CuStreamBeginCaptureFn = unsafe extern "C" fn(u64, u32) -> i32;
type CuStreamEndCaptureFn = unsafe extern "C" fn(u64, *mut *mut c_void) -> i32;
type CuGraphInstantiateFn =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *mut c_void, *mut c_void, u64) -> i32;
type CuGraphLaunchFn = unsafe extern "C" fn(*mut c_void, u64) -> i32;
type CuGraphDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
type CuGraphExecDestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;

// ── Constants ─────────────────────────────────────────────────────────

/// Capture mode: global capture of all operations.
const CU_STREAM_CAPTURE_MODE_GLOBAL: u32 = 0;

// ── CUDA Graph ────────────────────────────────────────────────────────

/// A captured CUDA graph ready for replay.
#[derive(Debug)]
pub struct CudaGraph {
    /// The captured graph.
    graph: *mut c_void,
    /// The instantiated executable graph.
    exec: *mut c_void,
    /// Device this graph was captured on.
    device: Arc<CudaDevice>,
}

// SAFETY: CudaGraph is used within a single CUDA context.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl CudaGraph {
    /// Capture a sequence of operations as a CUDA graph.
    ///
    /// The closure `f` should execute all operations that should
    /// be captured (forward pass of a model, etc.).
    ///
    /// # Example
    /// ```rust,ignore
    /// let graph = CudaGraph::capture(&stream, || {
    ///     let output = model.forward(&input);
    /// })?;
    /// ```
    pub fn capture<F: FnOnce()>(
        device: &Arc<CudaDevice>,
        stream: &CudaStream,
        f: F,
    ) -> Result<Self, CudaError> {
        device.set_current()?;

        let begin_fn: CuStreamBeginCaptureFn =
            resolve_graph_symbol("cuStreamBeginCapture_v2")?;
        let end_fn: CuStreamEndCaptureFn =
            resolve_graph_symbol("cuStreamEndCapture_v2")?;
        let instantiate_fn: CuGraphInstantiateFn =
            resolve_graph_symbol("cuGraphInstantiate_v2")?;

        // Begin capture
        let result = unsafe { (begin_fn)(stream.handle(), CU_STREAM_CAPTURE_MODE_GLOBAL) };
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuStreamBeginCapture failed: {result}"),
            });
        }

        // Execute the closure (operations are recorded, not executed)
        f();

        // End capture
        let mut graph: *mut c_void = std::ptr::null_mut();
        let result = unsafe { (end_fn)(stream.handle(), &mut graph) };
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuStreamEndCapture failed: {result}"),
            });
        }

        // Instantiate the graph
        let mut exec: *mut c_void = std::ptr::null_mut();
        let result = unsafe {
            (instantiate_fn)(
                &mut exec,
                graph,
                std::ptr::null_mut(), // error node
                std::ptr::null_mut(), // log buffer
                0,                     // log buffer size
            )
        };
        if result != 0 {
            unsafe {
                let destroy_fn: CuGraphDestroyFn = resolve_graph_symbol("cuGraphDestroy")?;
                (destroy_fn)(graph);
            }
            return Err(CudaError::DriverError {
                message: format!("cuGraphInstantiate failed: {result}"),
            });
        }

        Ok(Self {
            graph,
            exec,
            device: device.clone(),
        })
    }

    /// Launch the captured graph on a stream.
    pub fn launch(&self, stream: &CudaStream) -> CudaResult<()> {
        let launch_fn: CuGraphLaunchFn = resolve_graph_symbol("cuGraphLaunch")?;

        let result = unsafe { (launch_fn)(self.exec, stream.handle()) };

        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuGraphLaunch failed: {result}"),
            });
        }
        Ok(())
    }

    /// Synchronize after launching the graph (blocks until complete).
    pub fn launch_and_sync(&self, stream: &CudaStream) -> CudaResult<()> {
        self.launch(stream)?;
        stream.synchronize()
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.exec.is_null() {
            if self.device.set_current().is_ok() {
                if let Ok(destroy_fn) =
                    resolve_graph_symbol::<CuGraphExecDestroyFn>("cuGraphExecDestroy")
                {
                    unsafe { (destroy_fn)(self.exec); };
                }
            }
        }
        if !self.graph.is_null() {
            if self.device.set_current().is_ok() {
                if let Ok(destroy_fn) =
                    resolve_graph_symbol::<CuGraphDestroyFn>("cuGraphDestroy")
                {
                    unsafe { (destroy_fn)(self.graph); };
                }
            }
        }
    }
}

// ── Graph Runner (High-Level API) ─────────────────────────────────────

/// High-level inference runner using CUDA Graphs.
///
/// Captures a model's forward pass and replays it with minimal
/// CPU overhead.
pub struct GraphRunner {
    graph: CudaGraph,
    stream: CudaStream,
}

impl GraphRunner {
    /// Create a graph runner by capturing a forward pass.
    ///
    /// # Arguments
    /// * `device` - CUDA device
    /// * `capture_fn` - Closure that executes the forward pass
    ///
    /// # Example
    /// ```rust,ignore
    /// let runner = GraphRunner::capture(&device, || {
    ///     let _ = model.forward(&example_input);
    /// })?;
    ///
    /// // Every inference:
    /// runner.launch()?;
    /// ```
    pub fn capture<F: FnOnce()>(
        device: &Arc<CudaDevice>,
        capture_fn: F,
    ) -> Result<Self, CudaError> {
        let stream = CudaStream::new(device.clone())?;
        let graph = CudaGraph::capture(device, &stream, capture_fn)?;

        Ok(Self { graph, stream })
    }

    /// Launch the captured graph.
    pub fn launch(&self) -> CudaResult<()> {
        self.graph.launch(&self.stream)
    }

    /// Launch and synchronize (blocks until complete).
    pub fn launch_and_sync(&self) -> CudaResult<()> {
        self.graph.launch_and_sync(&self.stream)
    }

    /// Get the stream used for execution.
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Resolve a CUDA graph symbol.
fn resolve_graph_symbol<T>(name: &str) -> CudaResult<T> {
    let c_name = std::ffi::CString::new(name).unwrap();
    let c_lib_name = std::ffi::CString::new("libcuda.so.1").unwrap();

    unsafe {
        let lib = libc::dlopen(c_lib_name.as_ptr(), libc::RTLD_NOW);
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

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_capture_and_launch() {
        use crate::cuda_device::init_cuda;
        use crate::cuda_memory::CudaBuffer;

        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);
                let stream = CudaStream::new(device.clone()).unwrap();

                // Capture a simple memory operation
                let mut buf = CudaBuffer::new(device.clone(), 1024).unwrap();
                let data = vec![1.0f32; 256];
                let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

                let graph = CudaGraph::capture(&device, &stream, || {
                    let _ = buf.copy_from_host(&bytes);
                });

                if let Ok(graph) = graph {
                    // Launch should succeed
                    assert!(graph.launch(&stream).is_ok());
                    assert!(stream.synchronize().is_ok());
                }
            }
        }
    }

    #[test]
    fn test_graph_runner() {
        use crate::cuda_device::init_cuda;
        use crate::cuda_memory::CudaBuffer;

        if init_cuda().is_ok() {
            if let Ok(device) = CudaDevice::new(0) {
                let device = Arc::new(device);

                let runner = GraphRunner::capture(&device, || {
                    // Empty capture
                });

                if let Ok(runner) = runner {
                    assert!(runner.launch().is_ok());
                }
            }
        }
    }
}
