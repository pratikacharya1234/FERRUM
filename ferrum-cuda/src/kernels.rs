//! CUDA kernels — GPU-accelerated tensor operations.
//!
//! ## Execution Strategy
//!
//! FERRUM attempts GPU kernel launch first (NVRTC-compiled CUDA C via
//! `cuLaunchKernel`). If the GPU driver or NVRTC isn't available,
//! operations fall back to CPU execution transparently.
//!
//! ## Operation Codes
//!
//! Kernel op-codes match the enum discriminants of `BinaryOp`, `UnaryOp`,
//! `ScalarOp`, and `ReduceOp` for direct dispatch in the kernels.

use crate::cuda_device::init_cuda;
use crate::error::{CudaError, CudaResult};
use crate::kernel_launch;
use crate::tensor::CudaTensor;

/// Binary operations.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
    Max = 5,
    Min = 6,
}

/// Unary operations.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum UnaryOp {
    Neg = 0,
    Exp = 1,
    Log = 2,
    Sqrt = 3,
    Abs = 4,
    Relu = 5,
    Sigmoid = 6,
    Tanh = 7,
    LeakyRelu = 8,
    Gelu = 9,
    Silu = 10,
    Sin = 11,
    Cos = 12,
}

/// Scalar operations.
#[derive(Debug, Clone, Copy)]
pub enum ScalarOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Reduction operations.
#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

// ═══════════════════════════════════════════════════════════════════════
// Binary operations
// ═══════════════════════════════════════════════════════════════════════

/// Execute binary operation on GPU.
///
/// ## GPU Path
/// Launches the NVRTC-compiled `binary_kernel` via `cuLaunchKernel`.
/// Op-code is dispatched inline in the kernel.
///
/// ## CPU Fallback
/// Downloads tensors to host, computes element-wise, uploads result.
pub fn binary_op(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
    op: BinaryOp,
) -> CudaResult<()> {
    // Try GPU kernel launch first
    if kernel_launch::kernel_launch_available() {
        let n = output.numel() as u32;
        return kernel_launch::launch_binary(
            a.ptr(),
            b.ptr(),
            output.ptr(),
            op as u32,
            n,
        );
    }

    // CPU fallback
    cpu_binary_op(a, b, output, op)
}

/// CPU fallback for binary operations.
fn cpu_binary_op(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
    op: BinaryOp,
) -> CudaResult<()> {
    let a_data = a.to_f32()?;
    let b_data = b.to_f32()?;

    let result: Vec<f32> = a_data
        .iter()
        .zip(b_data.iter().cycle())
        .map(|(x, y)| match op {
            BinaryOp::Add => x + y,
            BinaryOp::Sub => x - y,
            BinaryOp::Mul => x * y,
            BinaryOp::Div => x / y,
            BinaryOp::Pow => x.powf(*y),
            BinaryOp::Max => x.max(*y),
            BinaryOp::Min => x.min(*y),
        })
        .collect();

    upload_f32_result(output, &result)
}

// ═══════════════════════════════════════════════════════════════════════
// Unary operations
// ═══════════════════════════════════════════════════════════════════════

/// Execute unary operation on GPU.
pub fn unary_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: UnaryOp,
) -> CudaResult<()> {
    if kernel_launch::kernel_launch_available() {
        let n = output.numel() as u32;
        return kernel_launch::launch_unary(input.ptr(), output.ptr(), op as u32, n);
    }

    cpu_unary_op(input, output, op)
}

fn cpu_unary_op(input: &CudaTensor, output: &CudaTensor, op: UnaryOp) -> CudaResult<()> {
    let data = input.to_f32()?;
    let result: Vec<f32> = data
        .iter()
        .map(|&x| match op {
            UnaryOp::Neg => -x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Abs => x.abs(),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::LeakyRelu => if x > 0.0 { x } else { 0.01 * x },
            UnaryOp::Gelu => {
                let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
                0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
            }
            UnaryOp::Silu => x * (1.0 / (1.0 + (-x).exp())),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
        })
        .collect();

    upload_f32_result(output, &result)
}

// ═══════════════════════════════════════════════════════════════════════
// Scalar operations
// ═══════════════════════════════════════════════════════════════════════

/// Execute scalar operation on GPU.
pub fn scalar_op(
    input: &CudaTensor,
    output: &CudaTensor,
    scalar: f64,
    op: ScalarOp,
) -> CudaResult<()> {
    // GPU path: use scalar broadcast kernel
    if kernel_launch::kernel_launch_available() {
        let n = input.numel() as u32;
        let op_code = match op {
            ScalarOp::Add => 0u32,
            ScalarOp::Sub => 1,
            ScalarOp::Mul => 2,
            ScalarOp::Div => 3,
            ScalarOp::Pow => 4,
        };
        return kernel_launch::launch_scalar(
            input.ptr(),
            output.ptr(),
            op_code,
            scalar as f32,
            n,
        );
    }

    // CPU fallback
    let data = input.to_f32()?;
    let s = scalar as f32;
    let result: Vec<f32> = data
        .iter()
        .map(|&x| match op {
            ScalarOp::Add => x + s,
            ScalarOp::Sub => x - s,
            ScalarOp::Mul => x * s,
            ScalarOp::Div => x / s,
            ScalarOp::Pow => x.powf(s),
        })
        .collect();

    upload_f32_result(output, &result)
}

// ═══════════════════════════════════════════════════════════════════════
// Reduction operations
// ═══════════════════════════════════════════════════════════════════════

/// Execute reduction operation on GPU.
pub fn reduce_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: ReduceOp,
    _axis: Option<usize>,
) -> CudaResult<()> {
    // GPU path: use reduce_sum kernel for Sum/Mean
    if kernel_launch::kernel_launch_available() && matches!(op, ReduceOp::Sum | ReduceOp::Mean)
    {
        let n = input.numel() as u32;
        kernel_launch::launch_reduce_sum(input.ptr(), output.ptr(), n)?;
        if matches!(op, ReduceOp::Mean) {
            // Post-process: divide by N
            let data = output.to_f32()?;
            let n = input.numel() as f32;
            let result: Vec<f32> = data.iter().map(|&x| x / n).collect();
            upload_f32_result(output, &result)?;
        }
        return Ok(());
    }

    // CPU fallback
    cpu_reduce_op(input, output, op)
}

fn cpu_reduce_op(input: &CudaTensor, output: &CudaTensor, op: ReduceOp) -> CudaResult<()> {
    let data = input.to_f32()?;
    let result = match op {
        ReduceOp::Sum => vec![data.iter().sum::<f32>()],
        ReduceOp::Mean => vec![data.iter().sum::<f32>() / data.len() as f32],
        ReduceOp::Max => vec![data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)],
        ReduceOp::Min => vec![data.iter().cloned().fold(f32::INFINITY, f32::min)],
        ReduceOp::Prod => vec![data.iter().fold(1.0f32, |acc, x| acc * x)],
    };

    upload_f32_result(output, &result)
}

// ═══════════════════════════════════════════════════════════════════════
// Matrix operations
// ═══════════════════════════════════════════════════════════════════════

/// Execute matrix multiplication on GPU.
///
/// ## cuBLAS Path (when available)
/// Dynamically loads `libcublas.so` and calls `cublasSgemm` or `cublasGemmEx`.
/// Performance: ~50-100x faster than naive Rust matmul.
///
/// ## CPU Fallback
/// Tiled matrix multiplication (32×32 tiles) for cache efficiency.
pub fn matmul(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    // Try cuBLAS first (50-100x faster)
    if let Ok(()) = cublas_matmul(a, b, output) {
        return Ok(());
    }

    // Fallback: tiled CPU matmul
    cpu_matmul(a, b, output)
}

/// Attempt cuBLAS SGEMM via the cublas module.
fn cublas_matmul(a: &CudaTensor, b: &CudaTensor, output: &CudaTensor) -> CudaResult<()> {
    use crate::cublas::CublasHandle;
    use crate::cuda_device::init_cuda;

    if !init_cuda().is_ok() {
        return Err(CudaError::NotAvailable);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    let device = a.device();
    let handle = CublasHandle::new(device.clone())?;

    // Enable TF32 for A100+
    let _ = handle.enable_tf32();

    handle.sgemm(
        false, false,
        m, n, k,
        1.0,            // alpha
        a.ptr(), k,     // A is m×k, lda = k
        b.ptr(), n,     // B is k×n, ldb = n
        0.0,            // beta
        output.ptr(), n, // C is m×n, ldc = n
    )
}

fn cpu_matmul(a: &CudaTensor, b: &CudaTensor, output: &CudaTensor) -> CudaResult<()> {
    let a_data = a.to_f32()?;
    let b_data = b.to_f32()?;

    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];

    let mut result = vec![0.0f32; m * n];

    // Tiled matrix multiplication for better cache performance
    const TILE: usize = 32;
    
    for i_tile in (0..m).step_by(TILE) {
        for j_tile in (0..n).step_by(TILE) {
            for k_tile in (0..k).step_by(TILE) {
                let i_end = (i_tile + TILE).min(m);
                let j_end = (j_tile + TILE).min(n);
                let k_end = (k_tile + TILE).min(k);
                
                for i in i_tile..i_end {
                    for j in j_tile..j_end {
                        let mut sum = result[i * n + j];
                        for l in k_tile..k_end {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }

    upload_f32_result(output, &result)
}

/// Execute transpose on GPU (CPU fallback for now).
pub fn transpose(input: &CudaTensor, output: &CudaTensor) -> CudaResult<()> {
    let data = input.to_f32()?;
    let in_shape = input.shape();
    let m = in_shape[in_shape.len() - 2];
    let n = in_shape[in_shape.len() - 1];

    let mut result = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = data[i * n + j];
        }
    }

    upload_f32_result(output, &result)
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

/// Upload an f32 slice to a GPU output tensor.
fn upload_f32_result(output: &CudaTensor, data: &[f32]) -> CudaResult<()> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    // Use a temporary CudaBuffer for the copy
    // The output tensor already has a buffer allocated; we copy via its device pointer
    let out_bytes = output.size_bytes();
    if bytes.len() > out_bytes {
        return Err(CudaError::InvalidArgument {
            message: format!(
                "Result size {} exceeds output size {}",
                bytes.len(),
                out_bytes
            ),
        });
    }
    // Direct device pointer write via cuMemcpyHtoD
    let resolve = |name: &str| -> CudaResult<*mut std::ffi::c_void> {
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
            Ok(ptr)
        }
    };

    type CuMemcpyHtoDFn = unsafe extern "C" fn(u64, *const std::ffi::c_void, usize) -> i32;
    let copy_fn: CuMemcpyHtoDFn = unsafe { std::mem::transmute(resolve("cuMemcpyHtoD_v2")?) };
    let result = unsafe { (copy_fn)(output.ptr(), bytes.as_ptr() as *const std::ffi::c_void, bytes.len()) };
    if result != 0 {
        return Err(CudaError::DriverError {
            message: format!("cuMemcpyHtoD failed: {result}"),
        });
    }
    Ok(())
}