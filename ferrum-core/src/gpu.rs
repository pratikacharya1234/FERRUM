//! GPU compute backend interface.
//!
//! `ferrum-core` cannot depend on `ferrum-cuda` (which depends on this
//! crate), so GPU compute is reached through a registered trait object:
//! `ferrum-cuda` implements [`GpuBackend`] and installs it with
//! [`register_gpu_backend`] at startup. Tensor ops on `Device::Cuda`
//! dispatch through [`backend`]; if no backend is registered the op
//! returns a loud error instead of silently computing on the CPU.
//!
//! Contract for all methods:
//! - f32 elements only; `dtype` gating happens in the tensor layer.
//! - Pointers are CUDA device addresses with any element offset already
//!   applied by the caller.
//! - Output buffers are contiguous and sized for the full result.
//! - Strided inputs describe (dims, strides) in *elements*; a stride of 0
//!   broadcasts that dimension.
//! - Errors are strings; the tensor layer wraps them in `FerrumError`.

use std::sync::OnceLock;

/// Element-wise binary op codes (match the CUDA kernel's switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuBinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

/// Element-wise unary op codes (match the CUDA kernel's switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuUnaryOp {
    Neg = 0,
    Exp = 1,
    Log = 2,
    Sqrt = 3,
    Abs = 4,
    Relu = 5,
    Sigmoid = 6,
    Tanh = 7,
}

/// Scalar broadcast op codes (match the CUDA kernel's switch).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GpuScalarOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Pow = 4,
}

/// Compute backend for `Device::Cuda` tensors.
pub trait GpuBackend: Send + Sync {
    /// Row-major matmul: `c[m,n] = a[m,k] @ b[k,n]`, all contiguous.
    fn matmul_f32(
        &self,
        a: u64,
        b: u64,
        c: u64,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<(), String>;

    /// Strided/broadcast element-wise binary op into a contiguous output
    /// of shape `dims`.
    fn binary_f32(
        &self,
        a: u64,
        b: u64,
        out: u64,
        dims: &[usize],
        a_strides: &[usize],
        b_strides: &[usize],
        op: GpuBinaryOp,
    ) -> Result<(), String>;

    /// Element-wise unary op over `n` contiguous elements.
    fn unary_f32(&self, input: u64, out: u64, n: usize, op: GpuUnaryOp) -> Result<(), String>;

    /// Element-wise op with a scalar over `n` contiguous elements.
    fn scalar_f32(
        &self,
        input: u64,
        out: u64,
        n: usize,
        scalar: f32,
        op: GpuScalarOp,
    ) -> Result<(), String>;

    /// Sum of `n` contiguous elements into a single f32 at `out`.
    fn reduce_sum_f32(&self, input: u64, out: u64, n: usize) -> Result<(), String>;

    /// Sum along the middle axis of a contiguous `[pre, axis, post]` view:
    /// `out[pre, post] = sum_axis(input)`.
    fn reduce_sum_dim_f32(
        &self,
        input: u64,
        out: u64,
        pre: usize,
        axis: usize,
        post: usize,
    ) -> Result<(), String>;

    /// Materialize a strided view (`dims` + `src_strides`, in elements)
    /// into a contiguous buffer of shape `dims`.
    fn copy_strided_f32(
        &self,
        src: u64,
        dst: u64,
        dims: &[usize],
        src_strides: &[usize],
    ) -> Result<(), String>;
}

static GPU_BACKEND: OnceLock<Box<dyn GpuBackend>> = OnceLock::new();

/// Install the process-wide GPU backend. The first registration wins;
/// later calls are ignored (returns false).
pub fn register_gpu_backend(backend: Box<dyn GpuBackend>) -> bool {
    GPU_BACKEND.set(backend).is_ok()
}

/// The registered GPU backend, if any.
pub fn backend() -> Option<&'static dyn GpuBackend> {
    GPU_BACKEND.get().map(|b| b.as_ref())
}

/// True once a GPU compute backend has been registered.
pub fn backend_registered() -> bool {
    GPU_BACKEND.get().is_some()
}
