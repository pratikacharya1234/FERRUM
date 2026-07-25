//! CUDA kernel launch infrastructure.
//!
//! Element-wise and reduction kernels are written as CUDA C and compiled
//! at runtime with NVRTC (the same approach as `fusion::compiler`).
//! NVRTC emits PTX matched to the installed toolkit and the driver
//! JIT-compiles it for whatever GPU is present, so no hardcoded
//! `.target` / `.version` directives are needed and new architectures
//! (A100, H100, ...) work without recompiling FERRUM.
//!
//! Compiled `CUfunction` handles are cached per kernel name for the
//! lifetime of the process.

use std::collections::HashMap;
use std::ffi::{c_void, CString};

use lazy_static::lazy_static;
use parking_lot::Mutex;

use crate::cuda_device::init_cuda;
use crate::error::{CudaError, CudaResult};
use crate::fusion::compiler::resolve_nvrtc_symbol;

// ── CUDA C Kernel Sources ─────────────────────────────────────────────

const SCALAR_KERNEL: &str = "scalar_kernel";
const BINARY_KERNEL: &str = "binary_kernel";
const UNARY_KERNEL: &str = "unary_kernel";
const REDUCE_SUM_KERNEL: &str = "reduce_sum_kernel";
const STRIDED_BINARY_KERNEL: &str = "strided_binary_kernel";
const STRIDED_COPY_KERNEL: &str = "strided_copy_kernel";
const REDUCE_DIM_KERNEL: &str = "reduce_dim_kernel";

/// Maximum tensor rank supported by the strided kernels. Matches
/// `MAX_INLINE_DIMS` in ferrum-core.
pub const MAX_STRIDED_DIMS: usize = 6;

/// Element-wise scalar broadcast operation: `out[i] = a[i] OP scalar`
/// Op codes match `kernels::ScalarOp`.
const SRC_SCALAR: &str = r#"
extern "C" __global__ void scalar_kernel(
    const float* __restrict__ a,
    float* __restrict__ out,
    unsigned int op_code,
    float scalar,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float r;
    switch (op_code) {
        case 0:  r = x + scalar; break;
        case 1:  r = x - scalar; break;
        case 2:  r = x * scalar; break;
        case 3:  r = x / scalar; break;
        // Special-case the exponents training actually uses: x*x must
        // work for negative x (Adam squares gradients), and sqrt is the
        // other optimizer hot path.
        case 4:  r = (scalar == 2.0f) ? x * x
                   : (scalar == 0.5f) ? sqrtf(x)
                   : (scalar == 1.0f) ? x
                   : powf(x, scalar); break;
        default: r = x; break;
    }
    out[i] = r;
}
"#;

/// Element-wise binary operation: `out[i] = a[i] OP b[i]`
/// Op codes match `kernels::BinaryOp`.
const SRC_BINARY: &str = r#"
extern "C" __global__ void binary_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int op_code,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = a[i];
    float y = b[i];
    float r;
    switch (op_code) {
        case 0:  r = x + y; break;
        case 1:  r = x - y; break;
        case 2:  r = x * y; break;
        case 3:  r = x / y; break;
        case 4:  r = powf(x, y); break;
        case 5:  r = fmaxf(x, y); break;
        case 6:  r = fminf(x, y); break;
        default: r = x; break;
    }
    out[i] = r;
}
"#;

/// Element-wise unary operation: `out[i] = OP(in[i])`
/// Op codes match `kernels::UnaryOp`.
const SRC_UNARY: &str = r#"
extern "C" __global__ void unary_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int op_code,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float r;
    switch (op_code) {
        case 0:  r = -x; break;
        case 1:  r = expf(x); break;
        case 2:  r = logf(x); break;
        case 3:  r = sqrtf(x); break;
        case 4:  r = fabsf(x); break;
        case 5:  r = fmaxf(x, 0.0f); break;
        case 6:  r = 1.0f / (1.0f + expf(-x)); break;
        case 7:  r = tanhf(x); break;
        case 8:  r = x > 0.0f ? x : 0.01f * x; break;
        case 9:  r = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); break;
        case 10: r = x / (1.0f + expf(-x)); break;
        case 11: r = sinf(x); break;
        case 12: r = cosf(x); break;
        default: r = x; break;
    }
    out[i] = r;
}
"#;

/// Parallel sum-reduction. Launched as a single block of 256 threads;
/// each thread accumulates a grid-stride partial sum, then the block
/// reduces in shared memory, so any `n` is handled correctly.
const SRC_REDUCE_SUM: &str = r#"
extern "C" __global__ void reduce_sum_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int n
) {
    __shared__ float smem[256];
    unsigned int tid = threadIdx.x;

    float acc = 0.0f;
    for (unsigned int i = tid; i < n; i += blockDim.x) {
        acc += in[i];
    }
    smem[tid] = acc;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = smem[0];
    }
}
"#;

/// Element-wise binary op over strided (possibly broadcast) inputs into a
/// contiguous output. `dims`/strides are passed as fixed-arity scalars
/// (rank <= MAX_STRIDED_DIMS, padded with dim=1/stride=0). A stride of 0
/// broadcasts that dimension. Op codes match `kernels::BinaryOp`.
const SRC_STRIDED_BINARY: &str = r#"
extern "C" __global__ void strided_binary_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    unsigned int op_code,
    unsigned int n,
    unsigned int ndim,
    unsigned int d0, unsigned int d1, unsigned int d2,
    unsigned int d3, unsigned int d4, unsigned int d5,
    unsigned int as0, unsigned int as1, unsigned int as2,
    unsigned int as3, unsigned int as4, unsigned int as5,
    unsigned int bs0, unsigned int bs1, unsigned int bs2,
    unsigned int bs3, unsigned int bs4, unsigned int bs5
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int dims[6] = {d0, d1, d2, d3, d4, d5};
    unsigned int astr[6] = {as0, as1, as2, as3, as4, as5};
    unsigned int bstr[6] = {bs0, bs1, bs2, bs3, bs4, bs5};

    unsigned int rem = idx;
    unsigned int a_off = 0;
    unsigned int b_off = 0;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        unsigned int c = rem % dims[i];
        rem /= dims[i];
        a_off += c * astr[i];
        b_off += c * bstr[i];
    }

    float x = a[a_off];
    float y = b[b_off];
    float r;
    switch (op_code) {
        case 0:  r = x + y; break;
        case 1:  r = x - y; break;
        case 2:  r = x * y; break;
        case 3:  r = x / y; break;
        default: r = x; break;
    }
    out[idx] = r;
}
"#;

/// Materialize a strided view into a contiguous buffer.
const SRC_STRIDED_COPY: &str = r#"
extern "C" __global__ void strided_copy_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    unsigned int n,
    unsigned int ndim,
    unsigned int d0, unsigned int d1, unsigned int d2,
    unsigned int d3, unsigned int d4, unsigned int d5,
    unsigned int s0, unsigned int s1, unsigned int s2,
    unsigned int s3, unsigned int s4, unsigned int s5
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int dims[6] = {d0, d1, d2, d3, d4, d5};
    unsigned int str[6] = {s0, s1, s2, s3, s4, s5};

    unsigned int rem = idx;
    unsigned int off = 0;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        unsigned int c = rem % dims[i];
        rem /= dims[i];
        off += c * str[i];
    }
    dst[idx] = src[off];
}
"#;

/// Sum a contiguous `[pre, axis, post]` tensor along the middle axis:
/// `out[p, q] = sum_a in[p, a, q]`. One thread per output element.
const SRC_REDUCE_DIM: &str = r#"
extern "C" __global__ void reduce_dim_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    unsigned int pre,
    unsigned int axis,
    unsigned int post
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = pre * post;
    if (idx >= total) return;

    unsigned int p = idx / post;
    unsigned int q = idx % post;

    float acc = 0.0f;
    for (unsigned int a = 0; a < axis; ++a) {
        acc += in[(p * axis + a) * post + q];
    }
    out[idx] = acc;
}
"#;

fn kernel_source(name: &str) -> &'static str {
    match name {
        SCALAR_KERNEL => SRC_SCALAR,
        BINARY_KERNEL => SRC_BINARY,
        UNARY_KERNEL => SRC_UNARY,
        REDUCE_SUM_KERNEL => SRC_REDUCE_SUM,
        STRIDED_BINARY_KERNEL => SRC_STRIDED_BINARY,
        STRIDED_COPY_KERNEL => SRC_STRIDED_COPY,
        REDUCE_DIM_KERNEL => SRC_REDUCE_DIM,
        _ => unreachable!("unknown kernel: {name}"),
    }
}

// ── NVRTC / Driver FFI ────────────────────────────────────────────────

type NvrtcCreateProgramFn = unsafe extern "C" fn(
    *mut *mut c_void,
    *const i8,
    *const i8,
    i32,
    *const *const i8,
    *const *const i8,
) -> i32;
type NvrtcDestroyProgramFn = unsafe extern "C" fn(*mut *mut c_void) -> i32;
type NvrtcCompileProgramFn = unsafe extern "C" fn(*mut c_void, i32, *const *const i8) -> i32;
type NvrtcGetPTXSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;
type NvrtcGetPTXFn = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;
type NvrtcGetProgramLogSizeFn = unsafe extern "C" fn(*mut c_void, *mut usize) -> i32;
type NvrtcGetProgramLogFn = unsafe extern "C" fn(*mut c_void, *mut i8) -> i32;
type CuModuleLoadDataFn = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32;
type CuModuleGetFunctionFn = unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const i8) -> i32;

const NVRTC_SUCCESS: i32 = 0;

/// Compile CUDA C source to NUL-terminated PTX with NVRTC.
///
/// No architecture flag is passed: NVRTC's baseline PTX target is
/// forward-compatible, and the driver JIT-compiles it for the GPU
/// actually present at `cuModuleLoadData` time.
fn nvrtc_compile_source(source: &str, name: &str) -> CudaResult<Vec<u8>> {
    unsafe {
        let create_fn: NvrtcCreateProgramFn = resolve_nvrtc_symbol("nvrtcCreateProgram")?;
        let compile_fn: NvrtcCompileProgramFn = resolve_nvrtc_symbol("nvrtcCompileProgram")?;
        let destroy_fn: NvrtcDestroyProgramFn = resolve_nvrtc_symbol("nvrtcDestroyProgram")?;

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
            return Err(CudaError::DriverError {
                message: format!("nvrtcCreateProgram ({name}) failed: {result}"),
            });
        }

        // No --use_fast_math: fast-math powf(x, y) is exp2(y*log2(x)),
        // which is NaN for negative x — and Adam squares gradients
        // (pow(x, 2.0)) that are routinely negative. These element-wise
        // kernels are bandwidth-bound, so precise math costs ~nothing.
        let result = (compile_fn)(prog, 0, std::ptr::null());
        if result != NVRTC_SUCCESS {
            let mut log = String::new();
            if let (Ok(log_size_fn), Ok(log_fn)) = (
                resolve_nvrtc_symbol::<NvrtcGetProgramLogSizeFn>("nvrtcGetProgramLogSize"),
                resolve_nvrtc_symbol::<NvrtcGetProgramLogFn>("nvrtcGetProgramLog"),
            ) {
                let mut log_size: usize = 0;
                (log_size_fn)(prog, &mut log_size);
                let mut buf = vec![0u8; log_size];
                (log_fn)(prog, buf.as_mut_ptr() as *mut i8);
                log = String::from_utf8_lossy(&buf).to_string();
            }
            (destroy_fn)(&mut prog);
            return Err(CudaError::DriverError {
                message: format!("nvrtcCompileProgram ({name}) failed: {log}"),
            });
        }

        let ptx_size_fn: NvrtcGetPTXSizeFn = resolve_nvrtc_symbol("nvrtcGetPTXSize")?;
        let ptx_fn: NvrtcGetPTXFn = resolve_nvrtc_symbol("nvrtcGetPTX")?;

        let mut ptx_size: usize = 0;
        (ptx_size_fn)(prog, &mut ptx_size);
        let mut ptx = vec![0u8; ptx_size];
        (ptx_fn)(prog, ptx.as_mut_ptr() as *mut i8);
        (destroy_fn)(&mut prog);

        // cuModuleLoadData requires a NUL terminator; nvrtcGetPTXSize
        // includes one, but guarantee it regardless.
        if ptx.last() != Some(&0) {
            ptx.push(0);
        }
        Ok(ptx)
    }
}

/// Load PTX into the current CUDA context and resolve the kernel function.
fn load_kernel_function(ptx: &[u8], name: &str) -> CudaResult<(*mut c_void, *mut c_void)> {
    unsafe {
        let load_fn: CuModuleLoadDataFn = resolve_nvrtc_symbol("cuModuleLoadData")?;
        let get_fn: CuModuleGetFunctionFn = resolve_nvrtc_symbol("cuModuleGetFunction")?;

        let mut module: *mut c_void = std::ptr::null_mut();
        let result = (load_fn)(&mut module, ptx.as_ptr() as *const c_void);
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuModuleLoadData ({name}) failed: {result}"),
            });
        }

        let c_name = CString::new(name).unwrap();
        let mut function: *mut c_void = std::ptr::null_mut();
        let result = (get_fn)(&mut function, module, c_name.as_ptr());
        if result != 0 {
            return Err(CudaError::DriverError {
                message: format!("cuModuleGetFunction ({name}) failed: {result}"),
            });
        }

        Ok((module, function))
    }
}

// ── Kernel Cache ──────────────────────────────────────────────────────

/// A loaded kernel function. The module is kept alive for the lifetime
/// of the process (never unloaded), so the function pointer stays valid.
#[derive(Clone, Copy)]
struct CachedKernel {
    function: *mut c_void,
    _module: *mut c_void,
}

// SAFETY: kernel handles are valid within the CUDA context and are only
// dereferenced by the driver.
unsafe impl Send for CachedKernel {}
unsafe impl Sync for CachedKernel {}

lazy_static! {
    /// NVRTC compilation results per kernel (compile once per process;
    /// failures are cached too — NVRTC availability doesn't change).
    static ref PTX_CACHE: Mutex<HashMap<&'static str, CudaResult<Vec<u8>>>> =
        Mutex::new(HashMap::new());
    /// Loaded `CUfunction` handles per kernel name. Only successes are
    /// cached: a module load can fail transiently (e.g. no current
    /// context yet) and is retried on the next call.
    static ref KERNEL_CACHE: Mutex<HashMap<&'static str, CachedKernel>> =
        Mutex::new(HashMap::new());
}

/// Get (compiling and loading on first use) the function handle for a kernel.
fn get_kernel(name: &'static str) -> CudaResult<*mut c_void> {
    init_cuda()?;

    if let Some(k) = KERNEL_CACHE.lock().get(name) {
        return Ok(k.function);
    }

    let ptx = {
        let mut cache = PTX_CACHE.lock();
        cache
            .entry(name)
            .or_insert_with(|| nvrtc_compile_source(kernel_source(name), name))
            .clone()
    }?;

    let (module, function) = load_kernel_function(&ptx, name)?;
    KERNEL_CACHE.lock().insert(
        name,
        CachedKernel {
            function,
            _module: module,
        },
    );
    Ok(function)
}

// ── Public Launch API ─────────────────────────────────────────────────

/// Resolution function for `cuLaunchKernel`.
pub fn cuda_launch_kernel(
    f: *mut c_void,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
    args: &mut [*mut c_void],
) -> CudaResult<()> {
    let resolve = |name: &str| -> CudaResult<unsafe extern "C" fn(
        *mut c_void,
        u32, u32, u32,
        u32, u32, u32,
        u32,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> i32> {
        let c_name = CString::new(name).unwrap();
        let lib_name = CString::new("libcuda.so.1").unwrap();
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
            Ok(std::mem::transmute(ptr))
        }
    };

    let launch = resolve("cuLaunchKernel")?;
    let result = unsafe {
        launch(
            f,
            grid.0, grid.1, grid.2,
            block.0, block.1, block.2,
            shared_mem,
            std::ptr::null_mut(), // stream (null = default stream)
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuLaunchKernel failed: {result}"),
        });
    }
    Ok(())
}

/// Launch the binary element-wise kernel on GPU.
pub fn launch_binary(
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    op_code: u32,
    n: u32,
) -> CudaResult<()> {
    assert!(a_ptr != 0, "null input pointer (a) in binary_kernel");
    assert!(b_ptr != 0, "null input pointer (b) in binary_kernel");
    assert!(out_ptr != 0, "null output pointer in binary_kernel");
    assert!(n > 0, "zero elements in binary_kernel");

    let f = get_kernel(BINARY_KERNEL)?;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    assert!(grid > 0, "zero grid in binary_kernel");

    let mut a = a_ptr as *mut c_void;
    let mut b = b_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut op = op_code;
    let mut count = n;

    let mut args: [*mut c_void; 5] = [
        &mut a as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
    ];

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Launch the unary element-wise kernel on GPU.
pub fn launch_unary(in_ptr: u64, out_ptr: u64, op_code: u32, n: u32) -> CudaResult<()> {
    assert!(in_ptr != 0, "null input pointer in unary_kernel");
    assert!(out_ptr != 0, "null output pointer in unary_kernel");
    assert!(n > 0, "zero elements in unary_kernel");

    let f = get_kernel(UNARY_KERNEL)?;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    assert!(grid > 0, "zero grid in unary_kernel");

    let mut inp = in_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut op = op_code;
    let mut count = n;

    let mut args: [*mut c_void; 4] = [
        &mut inp as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
    ];

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Launch the reduction kernel (sum) on GPU.
pub fn launch_reduce_sum(in_ptr: u64, out_ptr: u64, n: u32) -> CudaResult<()> {
    assert!(in_ptr != 0, "null input pointer in reduce_sum_kernel");
    assert!(out_ptr != 0, "null output pointer in reduce_sum_kernel");
    assert!(n > 0, "zero elements in reduce_sum_kernel");

    let f = get_kernel(REDUCE_SUM_KERNEL)?;

    let mut inp = in_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut count = n;

    let mut args: [*mut c_void; 3] = [
        &mut inp as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
    ];

    // Single block; the kernel grid-strides over the input and uses a
    // static __shared__ array, so no dynamic shared memory is needed.
    cuda_launch_kernel(f, (1, 1, 1), (256, 1, 1), 0, &mut args)
}

/// Launch the scalar broadcast kernel on GPU.
pub fn launch_scalar(a_ptr: u64, out_ptr: u64, op_code: u32, scalar: f32, n: u32) -> CudaResult<()> {
    assert!(a_ptr != 0, "null input pointer in scalar_kernel");
    assert!(out_ptr != 0, "null output pointer in scalar_kernel");
    assert!(n > 0, "zero elements in scalar_kernel");

    let f = get_kernel(SCALAR_KERNEL)?;
    let block = 256u32;
    let grid = (n + block - 1) / block;
    assert!(grid > 0, "zero grid in scalar_kernel");

    let mut a = a_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut op = op_code;
    let mut s = scalar;
    let mut count = n;

    let mut args: [*mut c_void; 5] = [
        &mut a as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut s as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
    ];

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Pad dims (with 1) and strides (with 0) to `MAX_STRIDED_DIMS`.
fn pad_dims_strides(dims: &[usize], strides: &[usize]) -> CudaResult<([u32; 6], [u32; 6], u32)> {
    if dims.len() > MAX_STRIDED_DIMS || strides.len() != dims.len() {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "strided kernel supports rank <= {MAX_STRIDED_DIMS}, got dims {dims:?} / strides {strides:?}"
            ),
        });
    }
    let mut d = [1u32; 6];
    let mut s = [0u32; 6];
    for i in 0..dims.len() {
        d[i] = dims[i] as u32;
        s[i] = strides[i] as u32;
    }
    Ok((d, s, dims.len().max(1) as u32))
}

/// Launch the strided/broadcast binary kernel: contiguous `out[dims]` from
/// strided `a` and `b` (stride 0 = broadcast dimension).
pub fn launch_strided_binary(
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    op_code: u32,
    dims: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
) -> CudaResult<()> {
    let n: usize = dims.iter().product::<usize>().max(1);
    let (d, astr, ndim) = pad_dims_strides(dims, a_strides)?;
    let (_, bstr, _) = pad_dims_strides(dims, b_strides)?;

    let f = get_kernel(STRIDED_BINARY_KERNEL)?;
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;

    let mut a = a_ptr as *mut c_void;
    let mut b = b_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut op = op_code;
    let mut count = n as u32;
    let mut nd = ndim;
    let mut dv = d;
    let mut av = astr;
    let mut bv = bstr;

    let mut args: Vec<*mut c_void> = vec![
        &mut a as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut op as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
        &mut nd as *mut _ as *mut c_void,
    ];
    for i in 0..6 {
        args.push(&mut dv[i] as *mut _ as *mut c_void);
    }
    for i in 0..6 {
        args.push(&mut av[i] as *mut _ as *mut c_void);
    }
    for i in 0..6 {
        args.push(&mut bv[i] as *mut _ as *mut c_void);
    }

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Launch the strided copy kernel: contiguous `dst[dims]` from a strided view.
pub fn launch_strided_copy(
    src_ptr: u64,
    dst_ptr: u64,
    dims: &[usize],
    src_strides: &[usize],
) -> CudaResult<()> {
    let n: usize = dims.iter().product::<usize>().max(1);
    let (d, s, ndim) = pad_dims_strides(dims, src_strides)?;

    let f = get_kernel(STRIDED_COPY_KERNEL)?;
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;

    let mut src = src_ptr as *mut c_void;
    let mut dst = dst_ptr as *mut c_void;
    let mut count = n as u32;
    let mut nd = ndim;
    let mut dv = d;
    let mut sv = s;

    let mut args: Vec<*mut c_void> = vec![
        &mut src as *mut _ as *mut c_void,
        &mut dst as *mut _ as *mut c_void,
        &mut count as *mut _ as *mut c_void,
        &mut nd as *mut _ as *mut c_void,
    ];
    for i in 0..6 {
        args.push(&mut dv[i] as *mut _ as *mut c_void);
    }
    for i in 0..6 {
        args.push(&mut sv[i] as *mut _ as *mut c_void);
    }

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Launch the axis reduction kernel over a contiguous `[pre, axis, post]` input.
pub fn launch_reduce_dim(
    in_ptr: u64,
    out_ptr: u64,
    pre: u32,
    axis: u32,
    post: u32,
) -> CudaResult<()> {
    let f = get_kernel(REDUCE_DIM_KERNEL)?;
    let total = pre * post;
    let block = 256u32;
    let grid = (total + block - 1) / block;

    let mut inp = in_ptr as *mut c_void;
    let mut out = out_ptr as *mut c_void;
    let mut p = pre;
    let mut a = axis;
    let mut q = post;

    let mut args: [*mut c_void; 5] = [
        &mut inp as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut p as *mut _ as *mut c_void,
        &mut a as *mut _ as *mut c_void,
        &mut q as *mut _ as *mut c_void,
    ];

    cuda_launch_kernel(f, (grid, 1, 1), (block, 1, 1), 0, &mut args)
}

/// Check whether GPU kernel launch is available.
pub fn kernel_launch_available() -> bool {
    get_kernel(BINARY_KERNEL).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::cuda_device::CudaDevice;
    use crate::cuda_memory::CudaBuffer;

    #[test]
    fn test_binary_kernel_when_gpu_available() {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            if !kernel_launch_available() {
                return;
            }
            let n = 16u32;
            let size = (n * 4) as usize;

            let mut a_buf = CudaBuffer::new(device.clone(), size).unwrap();
            let mut b_buf = CudaBuffer::new(device.clone(), size).unwrap();
            let out_buf = CudaBuffer::new(device.clone(), size).unwrap();

            let a_data: Vec<f32> = (0..n).map(|i| i as f32 * 2.0).collect();
            let b_data: Vec<f32> = (0..n).map(|i| i as f32 * 3.0).collect();
            let a_bytes = unsafe {
                std::slice::from_raw_parts(a_data.as_ptr() as *const u8, size)
            };
            let b_bytes = unsafe {
                std::slice::from_raw_parts(b_data.as_ptr() as *const u8, size)
            };
            a_buf.copy_from_host(a_bytes).unwrap();
            b_buf.copy_from_host(b_bytes).unwrap();

            launch_binary(a_buf.ptr(), b_buf.ptr(), out_buf.ptr(), 0, n).unwrap();
            device.synchronize().unwrap();

            let mut result = vec![0u8; size];
            out_buf.copy_to_host(&mut result).unwrap();
            let result_f32 = unsafe {
                std::slice::from_raw_parts(result.as_ptr() as *const f32, n as usize)
            };

            for i in 0..n as usize {
                let expected = a_data[i] + b_data[i];
                assert!((result_f32[i] - expected).abs() < 0.01,
                    "Element {i}: expected {expected}, got {}", result_f32[i]);
            }
        }
    }

    #[test]
    fn test_reduce_sum_kernel_when_gpu_available() {
        if let Ok(device) = CudaDevice::new(0) {
            let device = Arc::new(device);
            if !kernel_launch_available() {
                return;
            }
            // Deliberately larger than one block (256) to exercise the
            // grid-stride accumulation.
            let n = 1000u32;
            let size = (n * 4) as usize;

            let mut in_buf = CudaBuffer::new(device.clone(), size).unwrap();
            let out_buf = CudaBuffer::new(device.clone(), 4).unwrap();

            let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let bytes = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, size)
            };
            in_buf.copy_from_host(bytes).unwrap();

            launch_reduce_sum(in_buf.ptr(), out_buf.ptr(), n).unwrap();
            device.synchronize().unwrap();

            let mut result = vec![0u8; 4];
            out_buf.copy_to_host(&mut result).unwrap();
            let sum = f32::from_le_bytes([result[0], result[1], result[2], result[3]]);

            let expected: f32 = data.iter().sum();
            assert!((sum - expected).abs() / expected < 1e-4,
                "expected {expected}, got {sum}");
        }
    }
}
