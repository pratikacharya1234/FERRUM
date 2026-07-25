//! CUDA device management via the CUDA Driver API.
//!
//! FERRUM interfaces with NVIDIA GPUs through the CUDA Driver API
//! (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows). The driver
//! is loaded dynamically at runtime via `dlopen` — no compile-time
//! dependency on the CUDA toolkit.

use std::ffi::{c_int, c_void, CStr, CString};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;

use crate::error::{CudaError, CudaResult};

// ── CUDA Driver API FFI Types ─────────────────────────────────────────

/// Result codes returned by CUDA Driver API functions.
#[allow(non_camel_case_types, dead_code)]
type CUresult = i32;

/// CUDA device handle.
#[allow(non_camel_case_types)]
type CUdevice = i32;

/// CUDA context handle.
#[allow(non_camel_case_types)]
type CUcontext = *mut c_void;

/// CUDA Driver API function signatures.
type CuInitFn = unsafe extern "C" fn(u32) -> CUresult;
type CuDeviceGetCountFn = unsafe extern "C" fn(*mut c_int) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type CuDeviceGetNameFn = unsafe extern "C" fn(*mut i8, c_int, CUdevice) -> CUresult;
type CuDeviceGetAttributeFn = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type CuCtxCreateFn = unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult;
type CuCtxDestroyFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxSetCurrentFn = unsafe extern "C" fn(CUcontext) -> CUresult;
type CuCtxGetCurrentFn = unsafe extern "C" fn(*mut CUcontext) -> CUresult;
type CuCtxSynchronizeFn = unsafe extern "C" fn() -> CUresult;
type CuModuleLoadDataFn = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type CuModuleGetFunctionFn = unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    CUfunction,
    u32, u32, u32,   // grid dims
    u32, u32, u32,   // block dims
    u32,             // shared mem bytes
    CUstream,
    *mut *mut c_void, // kernel params
    *mut *mut c_void, // extra
) -> CUresult;

/// CUDA module handle.
#[allow(non_camel_case_types)]
type CUmodule = *mut c_void;
/// CUDA function (kernel) handle.
#[allow(non_camel_case_types)]
type CUfunction = *mut c_void;
/// CUDA stream handle.
#[allow(non_camel_case_types)]
type CUstream = *mut c_void;

/// Device attributes for `cuDeviceGetAttribute`.
const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: c_int = 1;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: c_int = 2;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: c_int = 3;
const CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: c_int = 4;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: c_int = 5;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: c_int = 6;
const CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: c_int = 7;
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
const CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: c_int = 9;
const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: c_int = 41;
const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: c_int = 11;

// ── Driver Handle ─────────────────────────────────────────────────────

/// Loaded CUDA Driver API function table.
#[derive(Debug)]
struct CudaDriver {
    _lib: *mut c_void,
    cu_init: CuInitFn,
    cu_device_get_count: CuDeviceGetCountFn,
    cu_device_get: CuDeviceGetFn,
    cu_device_get_name: CuDeviceGetNameFn,
    cu_device_get_attribute: CuDeviceGetAttributeFn,
    cu_ctx_create: CuCtxCreateFn,
    cu_ctx_destroy: CuCtxDestroyFn,
    cu_ctx_set_current: CuCtxSetCurrentFn,
    cu_ctx_get_current: CuCtxGetCurrentFn,
    cu_ctx_synchronize: CuCtxSynchronizeFn,
    cu_module_load_data: CuModuleLoadDataFn,
    cu_module_get_function: CuModuleGetFunctionFn,
    cu_launch_kernel: CuLaunchKernelFn,
}

// SAFETY: Function pointers are resolved from a shared library and are
// thread-safe (CUDA Driver API is inherently thread-safe).
unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

/// Global CUDA driver instance — loaded once on first use.
static CUDA_DRIVER: OnceLock<Result<Arc<CudaDriver>, CudaError>> = OnceLock::new();

/// Global CUDA initialization state.
static CUDA_INITIALIZED: OnceLock<bool> = OnceLock::new();

/// Current device per thread.
thread_local! {
    static CURRENT_DEVICE: AtomicUsize = const { AtomicUsize::new(0) };
}

// ── Driver Loading ────────────────────────────────────────────────────

fn load_cuda_driver() -> CudaResult<Arc<CudaDriver>> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: We load known system libraries with fixed paths.
        // Function pointers are resolved by name and transmuted to
        // the correct C ABI signatures.
        unsafe {
            // Load libcuda.so.1
            let lib_name = CString::new("libcuda.so.1").map_err(|_| {
                CudaError::NotAvailable
            })?;
            let lib = libc::dlopen(lib_name.as_ptr(), libc::RTLD_NOW);
            if lib.is_null() {
                // Try unversioned symlink
                let lib_name2 = CString::new("libcuda.so").map_err(|_| {
                    CudaError::NotAvailable
                })?;
                let lib2 = libc::dlopen(lib_name2.as_ptr(), libc::RTLD_NOW);
                if lib2.is_null() {
                    return Err(CudaError::NotAvailable);
                }
                build_driver(lib2)
            } else {
                build_driver(lib)
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Windows: LoadLibrary("nvcuda.dll")
        #[cfg(target_os = "windows")]
        {
            use std::ffi::CString;
            let lib_name = CString::new("nvcuda.dll").map_err(|_| CudaError::NotAvailable)?;
            unsafe {
                let lib = windows::Win32::System::LibraryLoader::LoadLibraryA(
                    windows::core::PCSTR(lib_name.as_ptr() as *const u8)
                );
                match lib {
                    Ok(handle) => build_driver(handle.0 as *mut c_void),
                    Err(_) => Err(CudaError::NotAvailable),
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            Err(CudaError::NotAvailable)
        }
    }
}

#[cfg(target_os = "linux")]
unsafe fn build_driver(lib: *mut c_void) -> CudaResult<Arc<CudaDriver>> {
    macro_rules! resolve {
        ($name:expr, $type:ty) => {{
            let c_name = CString::new($name).unwrap();
            let ptr = libc::dlsym(lib, c_name.as_ptr());
            if ptr.is_null() {
                libc::dlclose(lib);
                return Err(CudaError::NotAvailable);
            }
            std::mem::transmute::<*mut c_void, $type>(ptr)
        }};
    }

    let driver = Arc::new(CudaDriver {
        _lib: lib,
        cu_init: resolve!("cuInit", CuInitFn),
        cu_device_get_count: resolve!("cuDeviceGetCount", CuDeviceGetCountFn),
        cu_device_get: resolve!("cuDeviceGet", CuDeviceGetFn),
        cu_device_get_name: resolve!("cuDeviceGetName", CuDeviceGetNameFn),
        cu_device_get_attribute: resolve!("cuDeviceGetAttribute", CuDeviceGetAttributeFn),
        cu_ctx_create: resolve!("cuCtxCreate_v2", CuCtxCreateFn),
        cu_ctx_destroy: resolve!("cuCtxDestroy_v2", CuCtxDestroyFn),
        cu_ctx_set_current: resolve!("cuCtxSetCurrent", CuCtxSetCurrentFn),
        cu_ctx_get_current: resolve!("cuCtxGetCurrent", CuCtxGetCurrentFn),
        cu_ctx_synchronize: resolve!("cuCtxSynchronize", CuCtxSynchronizeFn),
        cu_module_load_data: resolve!("cuModuleLoadData", CuModuleLoadDataFn),
        cu_module_get_function: resolve!("cuModuleGetFunction", CuModuleGetFunctionFn),
        cu_launch_kernel: resolve!("cuLaunchKernel", CuLaunchKernelFn),
    });

    Ok(driver)
}

fn get_driver() -> CudaResult<Arc<CudaDriver>> {
    CUDA_DRIVER
        .get_or_init(|| match load_cuda_driver() {
            Ok(driver) => Ok(driver),
            Err(e) => Err(e),
        })
        .clone()
        .map_err(|e| e.clone())
}

// ── Device Properties ─────────────────────────────────────────────────

/// Properties of a CUDA device, queried from the actual GPU hardware.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// GPU name (e.g. "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Total global memory in bytes.
    pub total_memory: usize,
    /// Number of streaming multiprocessors.
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
    /// Warp size (always 32 on NVIDIA).
    pub warp_size: usize,
    /// Shared memory per block in bytes.
    pub shared_memory_per_block: usize,
    /// Whether unified (managed) memory is supported.
    pub unified_memory: bool,
}

impl Default for DeviceProperties {
    /// Fallback properties when no GPU is available.
    fn default() -> Self {
        Self {
            name: "CPU (no GPU detected)".to_string(),
            total_memory: 0,
            multiprocessor_count: 0,
            compute_major: 0,
            compute_minor: 0,
            max_threads_per_block: 0,
            max_block_dim: [0, 0, 0],
            max_grid_dim: [0, 0, 0],
            warp_size: 32,
            shared_memory_per_block: 0,
            unified_memory: false,
        }
    }
}

// ── CudaDevice ────────────────────────────────────────────────────────

/// A handle to a physical CUDA GPU device.
#[derive(Debug)]
pub struct CudaDevice {
    /// Device ordinal (0, 1, 2, ...).
    id: usize,
    /// Properties queried from the hardware.
    properties: DeviceProperties,
    /// CUDA context for this device.
    context: CUcontext,
    /// Whether this handle created (and therefore must destroy) the
    /// context. False when reusing a context that already existed.
    owns_context: bool,
    /// Driver function table.
    driver: Arc<CudaDriver>,
    /// Allocated memory tracking (bytes).
    allocated_memory: AtomicUsize,
}

impl CudaDevice {
    /// Open a CUDA device by its ordinal index.
    ///
    /// Initializes the CUDA driver, queries hardware properties,
    /// and creates a CUDA context on the device.
    pub fn new(device_id: usize) -> CudaResult<Self> {
        init_cuda()?;

        let driver = get_driver()?;

        let count = get_device_count()?;
        if device_id >= count {
            return Err(CudaError::InvalidDevice {
                device_id,
                available: count,
            });
        }

        // Get the raw CUdevice handle
        let mut raw_device: CUdevice = 0;
        let result = unsafe {
            (driver.cu_device_get)(&mut raw_device, device_id as c_int)
        };
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuDeviceGet({device_id}) failed: {result}"),
            });
        }

        // Query properties
        let properties = query_device_properties(&driver, raw_device);

        // Reuse the thread's current context if one exists. Creating a
        // second context on the same device invalidates every module,
        // kernel, and cuBLAS handle cached under the first one (they are
        // per-context), which surfaces later as CUDA_ERROR_INVALID_HANDLE
        // (400) on launches. One process, one context.
        let mut ctx: CUcontext = std::ptr::null_mut();
        unsafe {
            (driver.cu_ctx_get_current)(&mut ctx);
        }
        let owns_context = ctx.is_null();
        if owns_context {
            let result = unsafe { (driver.cu_ctx_create)(&mut ctx, 0, raw_device) };
            if result != 0 {
                return Err(CudaError::DriverError {
                    message: format!("cuCtxCreate failed: {result}"),
                });
            }
            unsafe {
                (driver.cu_ctx_set_current)(ctx);
            }
        }

        Ok(Self {
            id: device_id,
            properties,
            context: ctx,
            owns_context,
            driver,
            allocated_memory: AtomicUsize::new(0),
        })
    }

    /// Get the device ordinal.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the human-readable GPU name.
    pub fn name(&self) -> &str {
        &self.properties.name
    }

    /// Get device properties.
    pub fn properties(&self) -> &DeviceProperties {
        &self.properties
    }

    /// Get total global memory in bytes.
    pub fn total_memory(&self) -> usize {
        self.properties.total_memory
    }

    /// Get currently tracked allocated memory in bytes.
    pub fn allocated_memory(&self) -> usize {
        self.allocated_memory.load(Ordering::Relaxed)
    }

    /// Get estimated free memory in bytes.
    pub fn free_memory(&self) -> usize {
        self.properties
            .total_memory
            .saturating_sub(self.allocated_memory())
    }

    /// Set this device's context as the current CUDA context.
    pub fn set_current(&self) -> CudaResult<()> {
        let result = unsafe {
            (self.driver.cu_ctx_set_current)(self.context)
        };
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuCtxSetCurrent failed: {result}"),
            });
        }
        CURRENT_DEVICE.with(|d| d.store(self.id, Ordering::Relaxed));
        Ok(())
    }

    /// Block until all GPU operations on this device are complete.
    pub fn synchronize(&self) -> CudaResult<()> {
        let result = unsafe { (self.driver.cu_ctx_synchronize)() };
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuCtxSynchronize failed: {result}"),
            });
        }
        Ok(())
    }

    /// Track GPU memory allocation.
    pub(crate) fn track_alloc(&self, bytes: usize) {
        self.allocated_memory.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Track GPU memory deallocation.
    pub(crate) fn track_free(&self, bytes: usize) {
        self.allocated_memory.fetch_sub(bytes, Ordering::Relaxed);
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        // Only destroy a context this handle created; destroying a
        // borrowed one would pull the context out from under every other
        // user in the process.
        if self.owns_context && !self.context.is_null() {
            unsafe {
                (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

// ── Driver Initialization ─────────────────────────────────────────────

/// Initialize the CUDA driver subsystem.
///
/// This loads the CUDA Driver API shared library and calls `cuInit(0)`.
/// It is idempotent — subsequent calls are cheap.
pub fn init_cuda() -> CudaResult<()> {
    if CUDA_INITIALIZED.get().is_some() {
        return Ok(());
    }

    let driver = get_driver()?;
    let result = unsafe { (driver.cu_init)(0) };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuInit(0) failed with error code {result}"),
        });
    }

    let _ = CUDA_INITIALIZED.set(true);
    Ok(())
}

/// Get the number of available CUDA devices from the actual hardware.
pub fn get_device_count() -> CudaResult<usize> {
    let driver = get_driver()?;
    let mut count: c_int = 0;
    let result = unsafe { (driver.cu_device_get_count)(&mut count) };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuDeviceGetCount failed: {result}"),
        });
    }
    Ok(count as usize)
}

/// Query properties for a specific CUDA device.
fn query_device_properties(driver: &CudaDriver, device: CUdevice) -> DeviceProperties {
    let mut props = DeviceProperties::default();

    unsafe {
        // Device name
        let mut name_buf = [0i8; 256];
        let name_result = (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, device);
        if name_result == 0 {
            if let Ok(name) = CStr::from_ptr(name_buf.as_ptr()).to_str() {
                props.name = name.to_string();
            }
        }

        // Query integer attributes
        let mut val: c_int = 0;

        macro_rules! attr {
            ($attr:ident, $field:expr) => {
                if (driver.cu_device_get_attribute)(&mut val, $attr, device) == 0 {
                    $field = val as usize;
                }
            };
        }

        attr!(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, props.max_threads_per_block);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, props.max_block_dim[0]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, props.max_block_dim[1]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, props.max_block_dim[2]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, props.max_grid_dim[0]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, props.max_grid_dim[1]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, props.max_grid_dim[2]);
        attr!(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, props.shared_memory_per_block);
        attr!(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, props.multiprocessor_count);
        attr!(CU_DEVICE_ATTRIBUTE_WARP_SIZE, props.warp_size);

        if (driver.cu_device_get_attribute)(&mut val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == 0 {
            props.compute_major = val as u32;
        }
        if (driver.cu_device_get_attribute)(&mut val, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == 0 {
            props.compute_minor = val as u32;
        }
        if (driver.cu_device_get_attribute)(&mut val, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device) == 0 {
            props.unified_memory = val != 0;
        }

        // Query total memory via cudaMemGetInfo equivalent
        // (Driver API way: cuMemGetInfo, but we need a context for that;
        // for now we estimate from device properties)
        // Real implementation would create a temporary context and call cuMemGetInfo
        props.total_memory = estimate_total_memory(&props);
    }

    props
}

/// Estimate total memory from compute capability and SM count.
///
/// This is a rough heuristic used when `cuMemGetInfo` isn't available
/// (it requires an active context). Production code should query
/// cuMemGetInfo inside a context.
fn estimate_total_memory(props: &DeviceProperties) -> usize {
    // Heuristic: high-end GPUs have ~10-24 GB
    // RTX 3080: 68 SMs, ~10 GB
    // RTX 4090: 128 SMs, ~24 GB
    // A100: 108 SMs, ~40 GB
    if props.multiprocessor_count > 100 {
        24 * 1024 * 1024 * 1024 // 24 GB
    } else if props.multiprocessor_count > 60 {
        10 * 1024 * 1024 * 1024 // 10 GB
    } else if props.multiprocessor_count > 0 {
        6 * 1024 * 1024 * 1024 // 6 GB
    } else {
        0
    }
}

/// Set the current CUDA device by ordinal.
pub fn set_device(device_id: usize) -> CudaResult<()> {
    let count = get_device_count()?;
    if device_id >= count {
        return Err(CudaError::InvalidDevice {
            device_id,
            available: count,
        });
    }

    CURRENT_DEVICE.with(|d| d.store(device_id, Ordering::Relaxed));
    Ok(())
}

/// Get the current CUDA device ordinal.
pub fn get_device() -> usize {
    CURRENT_DEVICE.with(|d| d.load(Ordering::Relaxed))
}

/// Synchronize a specific device.
pub fn synchronize_device(_device_id: usize) -> CudaResult<()> {
    let driver = get_driver()?;
    let result = unsafe { (driver.cu_ctx_synchronize)() };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuCtxSynchronize failed: {result}"),
        });
    }
    Ok(())
}

/// Synchronize all CUDA devices.
pub fn synchronize_all() -> CudaResult<()> {
    let driver = get_driver()?;
    let result = unsafe { (driver.cu_ctx_synchronize)() };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuCtxSynchronize failed: {result}"),
        });
    }
    Ok(())
}

/// Get the compute capability (major, minor) of the current CUDA device.
pub fn get_device_compute_capability() -> CudaResult<(u32, u32)> {
    let driver = get_driver()?;
    let mut major: c_int = 0;
    let mut minor: c_int = 0;
    let device = CURRENT_DEVICE.with(|d| d.load(Ordering::Relaxed));
    let result = unsafe {
        (driver.cu_device_get_attribute)(
            &mut major as *mut c_int,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device as c_int,
        )
    };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuDeviceGetAttribute(major) failed: {result}"),
        });
    }
    let result = unsafe {
        (driver.cu_device_get_attribute)(
            &mut minor as *mut c_int,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device as c_int,
        )
    };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuDeviceGetAttribute(minor) failed: {result}"),
        });
    }
    Ok((major as u32, minor as u32))
}

// ── Device Manager ────────────────────────────────────────────────────

/// Manages multiple CUDA devices.
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

    /// Initialize all available CUDA devices.
    pub fn initialize(&self) -> CudaResult<()> {
        init_cuda()?;
        let count = get_device_count()?;
        let mut devices = self.devices.write();

        for i in 0..count {
            devices.push(CudaDevice::new(i)?);
        }

        Ok(())
    }

    /// Get the number of initialized devices.
    pub fn device_count(&self) -> usize {
        self.devices.read().len()
    }

    /// Set the current device.
    pub fn set_device(&self, device_id: usize) -> CudaResult<()> {
        set_device(device_id)
    }

    /// Get the current device ordinal.
    pub fn current_device(&self) -> usize {
        get_device()
    }

    /// Synchronize the current device.
    pub fn synchronize(&self) -> CudaResult<()> {
        synchronize_device(get_device())
    }

    /// Get a reference to a device.
    ///
    /// Note: This creates a new `CudaDevice` handle each time because
    /// `RwLock` cannot return references. For performance-critical
    /// code, hold your own `CudaDevice` handles.
    pub fn get_device(&self, device_id: usize) -> CudaResult<CudaDevice> {
        let devices = self.devices.read();
        if device_id >= devices.len() {
            return Err(CudaError::InvalidDevice {
                device_id,
                available: devices.len(),
            });
        }
        CudaDevice::new(device_id)
    }
}

impl Default for CudaDeviceManager {
    fn default() -> Self {
        Self::new().expect("Failed to create CudaDeviceManager")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_cuda_does_not_panic() {
        // init_cuda should either succeed (GPU present) or return
        // NotAvailable — never panic.
        let result = init_cuda();
        assert!(result.is_ok() || matches!(result, Err(CudaError::NotAvailable)));
    }

    #[test]
    fn test_device_count_does_not_panic() {
        let result = get_device_count();
        assert!(result.is_ok() || matches!(result, Err(CudaError::NotAvailable)));
    }

    #[test]
    fn test_device_creation_when_available() {
        if let Ok(count) = get_device_count() {
            if count > 0 {
                let device = CudaDevice::new(0).unwrap();
                assert_eq!(device.id(), 0);
                assert!(!device.name().is_empty());
            }
        }
    }
}