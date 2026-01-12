//! CUDA kernels for tensor operations.
//! 
//! This module provides GPU-accelerated implementations of tensor operations.
//! In simulation mode, these fall back to CPU implementations.

use crate::tensor::CudaTensor;
use crate::error::{CudaError, CudaResult};

/// Binary operations.
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Max,
    Min,
}

/// Unary operations.
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    Relu,
    LeakyRelu,
    Gelu,
    Silu,
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

// ============================================================================
// Binary operations
// ============================================================================

/// Execute binary operation on GPU.
pub fn binary_op(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
    op: BinaryOp,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_binary_kernel(a, b, output, op)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_binary_op(a, b, output, op)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (a, b, output, op);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_binary_op(
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
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// Unary operations
// ============================================================================

/// Execute unary operation on GPU.
pub fn unary_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: UnaryOp,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_unary_kernel(input, output, op)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_unary_op(input, output, op)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output, op);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_unary_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: UnaryOp,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    
    let result: Vec<f32> = data
        .iter()
        .map(|x| match op {
            UnaryOp::Neg => -x,
            UnaryOp::Exp => x.exp(),
            UnaryOp::Log => x.ln(),
            UnaryOp::Sqrt => x.sqrt(),
            UnaryOp::Abs => x.abs(),
            UnaryOp::Sin => x.sin(),
            UnaryOp::Cos => x.cos(),
            UnaryOp::Tanh => x.tanh(),
            UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            UnaryOp::Relu => x.max(0.0),
            UnaryOp::LeakyRelu => if *x > 0.0 { *x } else { 0.01 * x },
            UnaryOp::Gelu => {
                let cdf = 0.5 * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh());
                x * cdf
            }
            UnaryOp::Silu => x / (1.0 + (-x).exp()),
        })
        .collect();
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// Scalar operations
// ============================================================================

/// Execute scalar operation on GPU.
pub fn scalar_op(
    input: &CudaTensor,
    output: &CudaTensor,
    scalar: f64,
    op: ScalarOp,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_scalar_kernel(input, output, scalar, op)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_scalar_op(input, output, scalar, op)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output, scalar, op);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_scalar_op(
    input: &CudaTensor,
    output: &CudaTensor,
    scalar: f64,
    op: ScalarOp,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    let s = scalar as f32;
    
    let result: Vec<f32> = data
        .iter()
        .map(|x| match op {
            ScalarOp::Add => x + s,
            ScalarOp::Sub => x - s,
            ScalarOp::Mul => x * s,
            ScalarOp::Div => x / s,
            ScalarOp::Pow => x.powf(s),
        })
        .collect();
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// Reduction operations
// ============================================================================

/// Execute reduction operation on GPU.
pub fn reduce_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: ReduceOp,
    axis: Option<usize>,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_reduce_kernel(input, output, op, axis)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_reduce_op(input, output, op, axis)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output, op, axis);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_reduce_op(
    input: &CudaTensor,
    output: &CudaTensor,
    op: ReduceOp,
    axis: Option<usize>,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    
    let result = match axis {
        None => {
            // Full reduction
            let val = match op {
                ReduceOp::Sum => data.iter().sum(),
                ReduceOp::Mean => data.iter().sum::<f32>() / data.len() as f32,
                ReduceOp::Max => data.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                ReduceOp::Min => data.iter().cloned().fold(f32::INFINITY, f32::min),
                ReduceOp::Prod => data.iter().product(),
            };
            vec![val]
        }
        Some(ax) => {
            // Axis reduction - simplified implementation
            let shape = input.shape();
            if ax >= shape.len() {
                return Err(CudaError::InvalidArgument {
                    message: format!("Axis {} out of bounds for {} dims", ax, shape.len()),
                });
            }
            
            // Compute output size
            let outer_size: usize = shape[..ax].iter().product();
            let reduce_size = shape[ax];
            let inner_size: usize = shape[ax + 1..].iter().product();
            let out_size = outer_size * inner_size;
            
            let mut result = vec![0.0f32; out_size];
            
            for o in 0..outer_size {
                for i in 0..inner_size {
                    let mut acc = match op {
                        ReduceOp::Sum | ReduceOp::Mean => 0.0,
                        ReduceOp::Max => f32::NEG_INFINITY,
                        ReduceOp::Min => f32::INFINITY,
                        ReduceOp::Prod => 1.0,
                    };
                    
                    for r in 0..reduce_size {
                        let idx = o * reduce_size * inner_size + r * inner_size + i;
                        let val = data[idx];
                        acc = match op {
                            ReduceOp::Sum | ReduceOp::Mean => acc + val,
                            ReduceOp::Max => acc.max(val),
                            ReduceOp::Min => acc.min(val),
                            ReduceOp::Prod => acc * val,
                        };
                    }
                    
                    if matches!(op, ReduceOp::Mean) {
                        acc /= reduce_size as f32;
                    }
                    
                    result[o * inner_size + i] = acc;
                }
            }
            
            result
        }
    };
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// Matrix operations
// ============================================================================

/// Execute matrix multiplication on GPU.
pub fn matmul(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        // Use cuBLAS for real CUDA
        launch_matmul_kernel(a, b, output)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_matmul(a, b, output)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (a, b, output);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_matmul(
    a: &CudaTensor,
    b: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    let a_data = a.to_f32()?;
    let b_data = b.to_f32()?;
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    
    let mut result = vec![0.0f32; m * n];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

/// Execute matrix transpose on GPU.
pub fn transpose(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_transpose_kernel(input, output)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_transpose(input, output)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_transpose(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    let shape = input.shape();
    
    let rows = shape[shape.len() - 2];
    let cols = shape[shape.len() - 1];
    
    let mut result = vec![0.0f32; data.len()];
    
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// Additional operations
// ============================================================================

/// Softmax along last dimension.
pub fn softmax(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_softmax_kernel(input, output)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_softmax(input, output)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_softmax(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    let shape = input.shape();
    let last_dim = *shape.last().unwrap_or(&1);
    let batch_size = data.len() / last_dim;
    
    let mut result = vec![0.0f32; data.len()];
    
    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let slice = &data[start..end];
        
        // Numerically stable softmax
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = slice.iter().map(|x| (x - max_val).exp()).sum();
        
        for (i, x) in slice.iter().enumerate() {
            result[start + i] = (x - max_val).exp() / exp_sum;
        }
    }
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

/// Log softmax along last dimension.
pub fn log_softmax(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    #[cfg(feature = "cuda")]
    {
        launch_log_softmax_kernel(input, output)
    }

    #[cfg(feature = "simulate")]
    {
        simulate_log_softmax(input, output)
    }

    #[cfg(not(any(feature = "cuda", feature = "simulate")))]
    {
        let _ = (input, output);
        Err(CudaError::NotAvailable)
    }
}

#[cfg(feature = "simulate")]
fn simulate_log_softmax(
    input: &CudaTensor,
    output: &CudaTensor,
) -> CudaResult<()> {
    let data = input.to_f32()?;
    let shape = input.shape();
    let last_dim = *shape.last().unwrap_or(&1);
    let batch_size = data.len() / last_dim;
    
    let mut result = vec![0.0f32; data.len()];
    
    for b in 0..batch_size {
        let start = b * last_dim;
        let end = start + last_dim;
        let slice = &data[start..end];
        
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = slice.iter().map(|x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
        
        for (i, x) in slice.iter().enumerate() {
            result[start + i] = x - log_sum_exp;
        }
    }
    
    let bytes = unsafe {
        std::slice::from_raw_parts(
            result.as_ptr() as *const u8,
            result.len() * 4,
        )
    };
    
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), output.ptr(), bytes.len());
    }
    
    Ok(())
}

// ============================================================================
// CUDA kernel launch stubs (for real CUDA implementation)
// ============================================================================

#[cfg(feature = "cuda")]
fn launch_binary_kernel(
    _a: &CudaTensor,
    _b: &CudaTensor,
    _output: &CudaTensor,
    _op: BinaryOp,
) -> CudaResult<()> {
    todo!("Real CUDA binary kernel")
}

#[cfg(feature = "cuda")]
fn launch_unary_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
    _op: UnaryOp,
) -> CudaResult<()> {
    todo!("Real CUDA unary kernel")
}

#[cfg(feature = "cuda")]
fn launch_scalar_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
    _scalar: f64,
    _op: ScalarOp,
) -> CudaResult<()> {
    todo!("Real CUDA scalar kernel")
}

#[cfg(feature = "cuda")]
fn launch_reduce_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
    _op: ReduceOp,
    _axis: Option<usize>,
) -> CudaResult<()> {
    todo!("Real CUDA reduce kernel")
}

#[cfg(feature = "cuda")]
fn launch_matmul_kernel(
    _a: &CudaTensor,
    _b: &CudaTensor,
    _output: &CudaTensor,
) -> CudaResult<()> {
    todo!("Real CUDA matmul kernel (cuBLAS)")
}

#[cfg(feature = "cuda")]
fn launch_transpose_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
) -> CudaResult<()> {
    todo!("Real CUDA transpose kernel")
}

#[cfg(feature = "cuda")]
fn launch_softmax_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
) -> CudaResult<()> {
    todo!("Real CUDA softmax kernel")
}

#[cfg(feature = "cuda")]
fn launch_log_softmax_kernel(
    _input: &CudaTensor,
    _output: &CudaTensor,
) -> CudaResult<()> {
    todo!("Real CUDA log softmax kernel")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_device::CudaDevice;
    use std::sync::Arc;

    #[test]
    #[cfg(feature = "simulate")]
    fn test_binary_add() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_f32(device.clone(), &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = CudaTensor::from_f32(device.clone(), &[4], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let result = a.add(&b).unwrap();
        let data = result.to_f32().unwrap();
        
        assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_unary_relu() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_f32(device, &[4], &[-1.0, 0.0, 1.0, 2.0]).unwrap();
        
        let result = a.relu().unwrap();
        let data = result.to_f32().unwrap();
        
        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_reduce_sum() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_f32(device, &[4], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        
        let result = a.sum().unwrap();
        let data = result.to_f32().unwrap();
        
        assert_eq!(data, vec![10.0]);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_matmul() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let a = CudaTensor::from_f32(device.clone(), &[2, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();
        let b = CudaTensor::from_f32(device, &[3, 2], &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();
        
        let result = a.matmul(&b).unwrap();
        let data = result.to_f32().unwrap();
        
        // [1,2,3] @ [1,2; 3,4; 5,6] = [22, 28]
        // [4,5,6] @ [1,2; 3,4; 5,6] = [49, 64]
        assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
    }
}
