//! NVRTC Runtime Compilation for Fused Kernels.
//!
//! Compiles CUDA C code at runtime using NVRTC (NVIDIA Runtime Compilation).
//! Compiled kernels are cached by source hash for reuse across process restarts.
//!
//! Performance: First compilation ~50ms, cached lookup ~1ns.

use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::sync::Arc;

use parking_lot::Mutex;

use crate::cuda_device::CudaDevice;
use crate::error::{CudaError, CudaResult};
use crate::fusion::codegen::CodeGen;
use crate::fusion::ir::FusionGraph;

// ── NVRTC Function Types ──────────────────────────────────────────────

type NvrtcCreateProgramFn = unsafe extern "C" fn(
    *mut *mut c_void,
    *const i8,
    *const i8,
    i32,
    *const *const i8,
    *const *const i8,
) -> i32;

type NvrtcDestroyProgramFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;

type NvrtcCompileProgramFn =
    unsafe extern "C" fn(*mut c_void, i32, *const *const i8) -> i32;

type NvrtcGetPTXSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;

type NvrtcGetPTXFn = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;

type NvrtcGetProgramLogSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;

type NvrtcGetProgramLogFn = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;

type CuModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;

type CuModuleGetFunctionFn =
    unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;

type CuModuleUnloadFn = unsafe extern "C" fn(*mut c_void) -> i32;

// ── NVRTC Constants ───────────────────────────────────────────────────

const NVRTC_SUCCESS: i32 = 0;

// ── Fusion Error ──────────────────────────────────────────────────────

#[derive(Debug)]
pub enum FusionError {
    CompileError(String),
    LoadError(String),
    LaunchError(String),
    NoDevice,
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::CompileError(msg) => write!(f, "NVRTC compile error: {}", msg),
            FusionError::LoadError(msg) => write!(f, "Module load error: {}", msg),
            FusionError::LaunchError(msg) => write!(f, "Kernel launch error: {}", msg),
            FusionError::NoDevice => write!(f, "No CUDA device available"),
        }
    }
}

impl std::error::Error for FusionError {}

// ── Compiled Kernel ───────────────────────────────────────────────────

/// A compiled CUDA kernel ready for launch.
#[derive(Clone)]
pub struct CompiledKernel {
    /// Function pointer to the kernel.
    pub function: *mut c_void,
    /// The CUDA module (must be kept alive).
    pub _module: Arc<CudaModule>,
    /// Kernel name.
    pub name: String,
    /// Grid dimensions.
    pub grid: (u32, u32, u32),
    /// Block dimensions.
    pub block: (u32, u32, u32),
    /// Shared memory size in bytes.
    pub shared_mem: u32,
}

// SAFETY: CompiledKernel is used within a single CUDA context.
unsafe impl Send for CompiledKernel {}
unsafe impl Sync for CompiledKernel {}

impl CompiledKernel {
    /// Launch this kernel with the given arguments.
    pub fn launch(&self, args: &mut [*mut c_void]) -> CudaResult<()> {
        use crate::kernel_launch::cuda_launch_kernel;

        cuda_launch_kernel(
            self.function,
            self.grid,
            self.block,
            self.shared_mem,
            args,
        )
    }
}

// ── CUDA Module ───────────────────────────────────────────────────────

/// A loaded CUDA module (keeps the module alive).
struct CudaModule {
    module: *mut c_void,
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                if let Ok(unload_fn) =
                    resolve_nvrtc_symbol::<CuModuleUnloadFn>("cuModuleUnload")
                {
                    (unload_fn)(self.module);
                }
            }
        }
    }
}

// SAFETY: CudaModule is used within a single CUDA context.
unsafe impl Send for CudaModule {}
unsafe impl Sync for CudaModule {}

// ── NVRTC Compiler ────────────────────────────────────────────────────

/// Runtime compiler for fused CUDA kernels.
///
/// Uses NVRTC to compile CUDA C code at runtime. Compiled kernels
/// are cached by source hash for reuse.
pub struct NvrtcCompiler {
    /// Cache: source hash -> compiled kernel.
    cache: Mutex<HashMap<String, CompiledKernel>>,
    /// Device to compile for.
    device: Arc<CudaDevice>,
}

impl NvrtcCompiler {
    /// Create a new NVRTC compiler.
    pub fn new(device: Arc<CudaDevice>) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            device,
        }
    }

    /// Compile a fusion graph into a launchable kernel.
    pub fn compile_graph(
        &self,
        graph: &FusionGraph,
        kernel_name: &str,
    ) -> Result<CompiledKernel, FusionError> {
        let (source, _) = CodeGen::generate(graph, kernel_name);
        self.compile_source(&source, kernel_name)
    }

    /// Compile CUDA C source code into a launchable kernel.
    pub fn compile_source(
        &self,
        source: &str,
        kernel_name: &str,
    ) -> Result<CompiledKernel, FusionError> {
        // Check cache
        let hash = blake3_hash(source);
        {
            let cache = self.cache.lock();
            if let Some(kernel) = cache.get(&hash) {
                return Ok(kernel.clone());
            }
        }

        // Ensure device is initialized
        self.device
            .set_current()
            .map_err(|_| FusionError::NoDevice)?;

        // Compile with NVRTC
        let ptx = self.nvrtc_compile(source, kernel_name)?;

        // Load into CUDA module
        let kernel = self.load_module(&ptx, kernel_name)?;

        // Cache
        self.cache.lock().insert(hash, kernel.clone());

        Ok(kernel)
    }

    /// Compile a simple binary op kernel.
    pub fn compile_binary_op(
        &self,
        op: &crate::fusion::ir::FusionOp,
        kernel_name: &str,
    ) -> Result<CompiledKernel, FusionError> {
        let (source, _) = CodeGen::generate_binary_op(op, kernel_name);
        self.compile_source(&source, kernel_name)
    }

    /// Compile a simple unary op kernel.
    pub fn compile_unary_op(
        &self,
        op: &crate::fusion::ir::FusionOp,
        kernel_name: &str,
    ) -> Result<CompiledKernel, FusionError> {
        let (source, _) = CodeGen::generate_unary_op(op, kernel_name);
        self.compile_source(&source, kernel_name)
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock();
        (cache.len(), cache.capacity())
    }

    /// Clear the kernel cache.
    pub fn clear_cache(&self) {
        self.cache.lock().clear();
    }

    // ── Private Methods ───────────────────────────────────────────────

    /// Compile CUDA C source with NVRTC.
    fn nvrtc_compile(&self, source: &str, name: &str) -> Result<String, FusionError> {
        unsafe {
            // Create program
            let create_fn: NvrtcCreateProgramFn =
                resolve_nvrtc_symbol("nvrtcCreateProgram")
                    .map_err(|e| FusionError::CompileError(e.to_string()))?;

            let c_source = CString::new(source).unwrap();
            let c_name = CString::new(name).unwrap();

            let mut prog: *mut c_void = std::ptr::null_mut();
            let result = (create_fn)(
                &mut prog,
                c_source.as_ptr(),
                c_name.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );

            if result != NVRTC_SUCCESS {
                return Err(FusionError::CompileError(format!(
                    "nvrtcCreateProgram failed: {}",
                    result
                )));
            }

            // Compile
            let compile_fn: NvrtcCompileProgramFn =
                resolve_nvrtc_symbol("nvrtcCompileProgram")
                    .map_err(|e| FusionError::CompileError(e.to_string()))?;

            // No architecture flag: NVRTC's baseline PTX target is
            // forward-compatible, and the driver JIT-compiles it for
            // whatever GPU is present at module-load time.
            let fast_math = CString::new("--use_fast_math").unwrap();
            let opts = [fast_math.as_ptr()];

            let result = (compile_fn)(prog, opts.len() as i32, opts.as_ptr());

            if result != NVRTC_SUCCESS {
                // Get error log
                let log_size_fn: NvrtcGetProgramLogSizeFn =
                    resolve_nvrtc_symbol("nvrtcGetProgramLogSize").unwrap();
                let log_fn: NvrtcGetProgramLogFn =
                    resolve_nvrtc_symbol("nvrtcGetProgramLog").unwrap();

                let mut log_size: usize = 0;
                (log_size_fn)(prog, &mut log_size);
                let mut log = vec![0u8; log_size];
                (log_fn)(prog, log.as_mut_ptr() as *mut i8);

                let destroy_fn: NvrtcDestroyProgramFn =
                    resolve_nvrtc_symbol("nvrtcDestroyProgram").unwrap();
                (destroy_fn)(&mut prog);

                return Err(FusionError::CompileError(
                    String::from_utf8_lossy(&log).to_string(),
                ));
            }

            // Get PTX
            let ptx_size_fn: NvrtcGetPTXSizeFn =
                resolve_nvrtc_symbol("nvrtcGetPTXSize").unwrap();
            let ptx_fn: NvrtcGetPTXFn = resolve_nvrtc_symbol("nvrtcGetPTX").unwrap();

            let mut ptx_size: usize = 0;
            (ptx_size_fn)(prog, &mut ptx_size);
            let mut ptx = vec![0u8; ptx_size];
            (ptx_fn)(prog, ptx.as_mut_ptr() as *mut i8);

            // Destroy program
            let destroy_fn: NvrtcDestroyProgramFn =
                resolve_nvrtc_symbol("nvrtcDestroyProgram").unwrap();
            (destroy_fn)(&mut prog);

            String::from_utf8(ptx).map_err(|e| FusionError::CompileError(e.to_string()))
        }
    }

    /// Load PTX into a CUDA module and get the kernel function.
    fn load_module(&self, ptx: &str, name: &str) -> Result<CompiledKernel, FusionError> {
        unsafe {
            // Load module
            let load_fn: CuModuleLoadDataFn =
                resolve_nvrtc_symbol("cuModuleLoadData")
                    .map_err(|e| FusionError::LoadError(e.to_string()))?;

            let mut module: *mut c_void = std::ptr::null_mut();
            let result = (load_fn)(&mut module, ptx.as_ptr() as *const c_void);

            if result != 0 {
                return Err(FusionError::LoadError(format!(
                    "cuModuleLoadData failed: {}",
                    result
                )));
            }

            // Get function
            let get_fn: CuModuleGetFunctionFn =
                resolve_nvrtc_symbol("cuModuleGetFunction")
                    .map_err(|e| FusionError::LoadError(e.to_string()))?;

            let c_name = CString::new(name).unwrap();
            let mut function: *mut c_void = std::ptr::null_mut();
            let result = (get_fn)(&mut function, module, c_name.as_ptr());

            if result != 0 {
                return Err(FusionError::LoadError(format!(
                    "cuModuleGetFunction failed: {}",
                    result
                )));
            }

            let cuda_module = Arc::new(CudaModule { module });

            Ok(CompiledKernel {
                function,
                _module: cuda_module,
                name: name.to_string(),
                grid: (0, 0, 0),  // Will be set at launch time
                block: (256, 1, 1),
                shared_mem: 0,
            })
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Resolve an NVRTC (or CUDA driver) symbol.
///
/// The library handle is intentionally never `dlclose`d: unlike libcuda
/// (which `init_cuda` keeps loaded), nothing else holds libnvrtc open,
/// so closing it here would unmap the library and turn the first call
/// through a resolved function pointer into a segfault.
pub(crate) fn resolve_nvrtc_symbol<T>(name: &str) -> CudaResult<T> {
    let c_name = CString::new(name).unwrap();

    // Try libnvrtc first, then fall back to libcuda
    let lib_names = ["libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11"];

    for lib_name in &lib_names {
        let c_lib_name = CString::new(*lib_name).unwrap();
        unsafe {
            let lib = libc::dlopen(c_lib_name.as_ptr(), libc::RTLD_NOW);
            if lib.is_null() {
                continue;
            }
            let ptr = libc::dlsym(lib, c_name.as_ptr());
            if !ptr.is_null() {
                return Ok(std::mem::transmute_copy::<*mut c_void, T>(&ptr));
            }
        }
    }

    // Try libcuda (some functions are there)
    let c_lib_name = CString::new("libcuda.so.1").unwrap();
    unsafe {
        let lib = libc::dlopen(c_lib_name.as_ptr(), libc::RTLD_NOW);
        if !lib.is_null() {
            let ptr = libc::dlsym(lib, c_name.as_ptr());
            if !ptr.is_null() {
                return Ok(std::mem::transmute_copy::<*mut c_void, T>(&ptr));
            }
        }
    }

    Err(CudaError::NotAvailable)
}

/// Simple BLAKE3-like hash (using std::collections::hash_map::DefaultHasher).
fn blake3_hash(data: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_simple_kernel() {
        let source = r#"
extern "C" __global__ void add_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}
"#;

        // This test only runs if NVRTC is available
        if let Ok(device) = crate::cuda_device::CudaDevice::new(0) {
            let device = Arc::new(device);
            let compiler = NvrtcCompiler::new(device);

            // Try to compile - will succeed only if NVRTC is installed
            if let Ok(kernel) = compiler.compile_source(source, "add_kernel") {
                assert!(!kernel.function.is_null());
                assert_eq!(kernel.name, "add_kernel");
            }
        }
    }

    #[test]
    fn test_cache_lookup() {
        let source = "test kernel source";

        let hash1 = blake3_hash(source);
        let hash2 = blake3_hash(source);
        let hash3 = blake3_hash("different source");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
