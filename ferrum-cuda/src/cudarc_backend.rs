//! cuBLAS + NVRTC GPU compute backend via `cudarc`.
//!
//! Enabled with `cargo build --features cudarc`.
//! This file is compiled only when the `cudarc` feature is active.
//!
//! Enabled with: `cargo build --features cudarc`
//!
//! When the `cudarc` feature is active, this module provides the real GPU
//! execution path through cuBLAS (matmul) and NVRTC-compiled CUDA kernels
//! (element-wise ops). Without it, the kernel dispatch falls back transparently
//! to CPU execution via the `kernel_launch` and `kernels` modules.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::cublas::result::GemmConfig;
use cudarc::cublas::{CudaBlas, Gemm};
use cudarc::driver::{CudaDevice, CudaDeviceBuilder, CudaFunction, CudaSlice, LaunchConfig};
use cudarc::nvrtc::Ptx;
use ferrum_core::{Device, FerrumError, Result, Tensor};

// ── Error helpers ─────────────────────────────────────────────────────

fn cuda_err(e: impl std::fmt::Debug) -> FerrumError {
    FerrumError::internal(format!("cudarc: {e:?}"))
}

// ── Backend ───────────────────────────────────────────────────────────

struct Backend {
    dev: Arc<CudaDevice>,
    blas: CudaBlas,
    add_kernel: CudaFunction,
    mul_kernel: CudaFunction,
    relu_kernel: CudaFunction,
}

static BACKENDS: std::sync::OnceLock<Mutex<HashMap<usize, Arc<Backend>>>> = std::sync::OnceLock::new();

fn backends() -> &'static Mutex<HashMap<usize, Arc<Backend>>> {
    BACKENDS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn get_backend(device_id: usize) -> Result<Arc<Backend>> {
    let mut map = backends().lock().map_err(|e| cuda_err(e))?;
    if let Some(b) = map.get(&device_id) {
        return Ok(b.clone());
    }
    let backend = init_backend(device_id)?;
    map.insert(device_id, backend.clone());
    Ok(backend)
}

fn init_backend(device_id: usize) -> Result<Arc<Backend>> {
    let dev = CudaDeviceBuilder::new(device_id)
        .build()
        .map_err(cuda_err)?;

    let blas = dev.cublas().map_err(cuda_err)?;

    let add_kernel = compile_kernel(&dev, ADD_SRC, "add_kernel")?;
    let mul_kernel = compile_kernel(&dev, MUL_SRC, "mul_kernel")?;
    let relu_kernel = compile_kernel(&dev, RELU_SRC, "relu_kernel")?;

    Ok(Arc::new(Backend {
        dev,
        blas,
        add_kernel,
        mul_kernel,
        relu_kernel,
    }))
}

fn compile_kernel(dev: &Arc<CudaDevice>, src: &str, name: &str) -> Result<CudaFunction> {
    let ptx = Ptx::from_src(src);
    dev.load_ptx(ptx, name, &[name]).map_err(cuda_err)
}

fn launch_config_1d(n: usize) -> LaunchConfig {
    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;
    LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ── Tensor ↔ GPU transfers ───────────────────────────────────────────

fn tensor_to_gpu(dev: &Arc<CudaDevice>, t: &Tensor) -> Result<CudaSlice<f32>> {
    let data = t.to_vec::<f32>()?;
    let mut slice = unsafe { dev.alloc::<f32>(data.len()).map_err(cuda_err)? };
    dev.htod_sync_copy_into(&data, &mut slice).map_err(cuda_err)?;
    Ok(slice)
}

fn gpu_to_tensor(slice: &CudaSlice<f32>, shape: &[usize]) -> Result<Tensor> {
    let dev = slice.device();
    let host: Vec<f32> = dev.dtoh_sync_copy(slice).map_err(cuda_err)?;
    Tensor::from_slice(&host, shape, Device::Cpu)
}

// ── cuBLAS matmul ─────────────────────────────────────────────────────

pub fn cublas_matmul(
    device_id: usize,
    a_slice: &CudaSlice<f32>,
    a_shape: &[usize],
    b_slice: &CudaSlice<f32>,
    b_shape: &[usize],
    c_slice: &mut CudaSlice<f32>,
) -> Result<()> {
    let backend = get_backend(device_id)?;

    let m = a_shape[a_shape.len() - 2] as i32;
    let k = a_shape[a_shape.len() - 1] as i32;
    let n = b_shape[b_shape.len() - 1] as i32;

    let cfg = GemmConfig {
        transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
        m: n,
        n: m,
        k,
        alpha: 1.0f32,
        lda: n,
        ldb: k,
        beta: 0.0f32,
        ldc: n,
    };

    unsafe {
        backend
            .blas
            .gemm(cfg, b_slice, a_slice, c_slice)
            .map_err(cuda_err)?;
    }
    Ok(())
}

// ── Element-wise kernels ──────────────────────────────────────────────

pub fn gpu_add(device_id: usize, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let backend = get_backend(device_id)?;
    let n = out.shape().iter().product::<usize>();
    let a_gpu = tensor_to_gpu(&backend.dev, a)?;
    let b_gpu = tensor_to_gpu(&backend.dev, b)?;
    let mut o_gpu = unsafe { backend.dev.alloc::<f32>(n).map_err(cuda_err)? };

    let cfg = launch_config_1d(n);
    unsafe {
        backend
            .add_kernel
            .clone()
            .launch(cfg, (&a_gpu, &b_gpu, &mut o_gpu, n as i32))
            .map_err(cuda_err)?;
    }
    backend.dev.synchronize().map_err(cuda_err)?;
    let result = gpu_to_tensor(&o_gpu, out.shape())?;
    *out = result;
    Ok(())
}

pub fn gpu_mul(device_id: usize, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let backend = get_backend(device_id)?;
    let n = out.shape().iter().product::<usize>();
    let a_gpu = tensor_to_gpu(&backend.dev, a)?;
    let b_gpu = tensor_to_gpu(&backend.dev, b)?;
    let mut o_gpu = unsafe { backend.dev.alloc::<f32>(n).map_err(cuda_err)? };

    let cfg = launch_config_1d(n);
    unsafe {
        backend
            .mul_kernel
            .clone()
            .launch(cfg, (&a_gpu, &b_gpu, &mut o_gpu, n as i32))
            .map_err(cuda_err)?;
    }
    backend.dev.synchronize().map_err(cuda_err)?;
    let result = gpu_to_tensor(&o_gpu, out.shape())?;
    *out = result;
    Ok(())
}

pub fn gpu_relu(device_id: usize, x: &Tensor, out: &mut Tensor) -> Result<()> {
    let backend = get_backend(device_id)?;
    let n = out.shape().iter().product::<usize>();
    let x_gpu = tensor_to_gpu(&backend.dev, x)?;
    let mut o_gpu = unsafe { backend.dev.alloc::<f32>(n).map_err(cuda_err)? };

    let cfg = launch_config_1d(n);
    unsafe {
        backend
            .relu_kernel
            .clone()
            .launch(cfg, (&x_gpu, &mut o_gpu, n as i32))
            .map_err(cuda_err)?;
    }
    backend.dev.synchronize().map_err(cuda_err)?;
    let result = gpu_to_tensor(&o_gpu, out.shape())?;
    *out = result;
    Ok(())
}

// ── Public dispatch (tries GPU, falls back to CPU) ────────────────────

pub fn dispatch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match a.device() {
        Device::Cuda(id) => {
            let shape_a = a.shape().to_vec();
            let shape_b = b.shape().to_vec();
            let m = shape_a[shape_a.len() - 2];
            let n = shape_b[shape_b.len() - 1];
            let backend = get_backend(id)?;
            let a_gpu = tensor_to_gpu(&backend.dev, a)?;
            let b_gpu = tensor_to_gpu(&backend.dev, b)?;
            let mut c_gpu = unsafe {
                backend.dev.alloc::<f32>(m * n).map_err(cuda_err)?
            };
            cublas_matmul(id, &a_gpu, &shape_a, &b_gpu, &shape_b, &mut c_gpu)?;
            backend.dev.synchronize().map_err(cuda_err)?;
            let mut out_shape = shape_a[..shape_a.len() - 2].to_vec();
            out_shape.push(m);
            out_shape.push(n);
            gpu_to_tensor(&c_gpu, &out_shape)
        }
        Device::Cpu => a.matmul(b),
        _ => a.matmul(b),
    }
}

pub fn dispatch_add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match a.device() {
        Device::Cuda(id) => {
            let result = a.add(b)?; // fallback
            let mut out = result.clone();
            gpu_add(id, a, b, &mut out)?;
            Ok(out)
        }
        _ => a.add(b),
    }
}

pub fn dispatch_mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    match a.device() {
        Device::Cuda(id) => {
            let result = a.mul(b)?;
            let mut out = result.clone();
            gpu_mul(id, a, b, &mut out)?;
            Ok(out)
        }
        _ => a.mul(b),
    }
}

pub fn dispatch_relu(x: &Tensor) -> Result<Tensor> {
    match x.device() {
        Device::Cuda(id) => {
            let result = x.relu()?;
            let mut out = result.clone();
            gpu_relu(id, x, &mut out)?;
            Ok(out)
        }
        _ => x.relu(),
    }
}

// ── NVIDIA CUDA kernel sources ────────────────────────────────────────

const ADD_SRC: &str = r#"
extern "C" __global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}
"#;

const MUL_SRC: &str = r#"
extern "C" __global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}
"#;

const RELU_SRC: &str = r#"
extern "C" __global__ void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = fmaxf(0.0f, x[idx]);
}
"#;

// ── Tests (only compile on GPU machines) ──────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_dispatch() {
        let a = Tensor::randn([2, 3], ferrum_core::DType::F32, Device::Cpu);
        let b = Tensor::randn([3, 2], ferrum_core::DType::F32, Device::Cpu);
        let c = dispatch_matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }
}