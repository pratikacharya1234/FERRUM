//! `ferrum_core::gpu::GpuBackend` implementation.
//!
//! This is the bridge that makes `Tensor` ops on `Device::Cuda` execute on
//! the GPU: element-wise and reduction ops run NVRTC-compiled kernels
//! (`kernel_launch`), matmul runs cuBLAS SGEMM (TF32 enabled). Errors
//! propagate — there are no CPU fallbacks on this path; if a kernel or
//! library is unavailable the tensor op fails loudly.
//!
//! Call [`register_gpu_backend`] once at startup (before any CUDA tensor
//! compute) to install it.

use std::sync::Arc;

use once_cell::sync::OnceCell;

use ferrum_core::gpu::{GpuBackend, GpuBinaryOp, GpuScalarOp, GpuUnaryOp};

use crate::cublas::CublasHandle;
use crate::cuda_device::CudaDevice;
use crate::kernel_launch;

/// cuBLAS handle for device 0, created once. Handle creation costs
/// milliseconds; per-matmul creation would dominate small GEMMs.
static CUBLAS: OnceCell<Result<CublasHandle, String>> = OnceCell::new();

fn cublas() -> Result<&'static CublasHandle, String> {
    CUBLAS
        .get_or_init(|| {
            let device = CudaDevice::new(0).map_err(|e| format!("CudaDevice::new(0): {e}"))?;
            let handle = CublasHandle::new(Arc::new(device))
                .map_err(|e| format!("cublasCreate: {e}"))?;
            // TF32 tensor cores for FP32 GEMM on Ampere+; harmless elsewhere.
            let _ = handle.enable_tf32();
            Ok(handle)
        })
        .as_ref()
        .map_err(|e| e.clone())
}

pub struct CudaComputeBackend;

impl GpuBackend for CudaComputeBackend {
    fn matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String> {
        let handle = cublas()?;
        handle
            .sgemm(
                false,
                false,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a,
                k as i32,
                b,
                n as i32,
                0.0,
                c,
                n as i32,
            )
            .map_err(|e| format!("cublasSgemm: {e}"))
    }

    fn binary_f32(
        &self,
        a: u64,
        b: u64,
        out: u64,
        dims: &[usize],
        a_strides: &[usize],
        b_strides: &[usize],
        op: GpuBinaryOp,
    ) -> Result<(), String> {
        kernel_launch::launch_strided_binary(a, b, out, op as u32, dims, a_strides, b_strides)
            .map_err(|e| format!("strided_binary_kernel: {e}"))
    }

    fn unary_f32(&self, input: u64, out: u64, n: usize, op: GpuUnaryOp) -> Result<(), String> {
        kernel_launch::launch_unary(input, out, op as u32, n as u32)
            .map_err(|e| format!("unary_kernel: {e}"))
    }

    fn scalar_f32(
        &self,
        input: u64,
        out: u64,
        n: usize,
        scalar: f32,
        op: GpuScalarOp,
    ) -> Result<(), String> {
        kernel_launch::launch_scalar(input, out, op as u32, scalar, n as u32)
            .map_err(|e| format!("scalar_kernel: {e}"))
    }

    fn reduce_sum_f32(&self, input: u64, out: u64, n: usize) -> Result<(), String> {
        kernel_launch::launch_reduce_sum(input, out, n as u32)
            .map_err(|e| format!("reduce_sum_kernel: {e}"))
    }

    fn reduce_sum_dim_f32(
        &self,
        input: u64,
        out: u64,
        pre: usize,
        axis: usize,
        post: usize,
    ) -> Result<(), String> {
        kernel_launch::launch_reduce_dim(input, out, pre as u32, axis as u32, post as u32)
            .map_err(|e| format!("reduce_dim_kernel: {e}"))
    }

    fn copy_strided_f32(
        &self,
        src: u64,
        dst: u64,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<(), String> {
        kernel_launch::launch_strided_copy(src, dst, dims, src_strides)
            .map_err(|e| format!("strided_copy_kernel: {e}"))
    }
}

/// Install the CUDA compute backend as the process-wide GPU backend.
///
/// Returns true if this call installed it (false if one was already
/// registered). Safe to call multiple times.
pub fn register_gpu_backend() -> bool {
    ferrum_core::gpu::register_gpu_backend(Box::new(CudaComputeBackend))
}
