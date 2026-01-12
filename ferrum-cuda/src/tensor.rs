//! GPU tensor implementation.

use std::sync::Arc;

use crate::cuda_device::CudaDevice;
use crate::cuda_memory::CudaBuffer;
use crate::error::{CudaError, CudaResult};
use crate::kernels;

/// A tensor stored on GPU.
#[derive(Debug)]
pub struct CudaTensor {
    /// GPU buffer containing the data.
    buffer: CudaBuffer,
    /// Shape of the tensor.
    shape: Vec<usize>,
    /// Strides for each dimension.
    strides: Vec<usize>,
    /// Number of elements.
    numel: usize,
    /// Data type size in bytes.
    dtype_size: usize,
    /// Device this tensor is on.
    device: Arc<CudaDevice>,
}

impl CudaTensor {
    /// Create a new GPU tensor with uninitialized data.
    pub fn new(device: Arc<CudaDevice>, shape: &[usize], dtype_size: usize) -> CudaResult<Self> {
        let numel: usize = shape.iter().product();
        let size_bytes = numel * dtype_size;
        let buffer = CudaBuffer::new(device.clone(), size_bytes)?;
        let strides = compute_strides(shape);

        Ok(Self {
            buffer,
            shape: shape.to_vec(),
            strides,
            numel,
            dtype_size,
            device,
        })
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(device: Arc<CudaDevice>, shape: &[usize], dtype_size: usize) -> CudaResult<Self> {
        let mut tensor = Self::new(device, shape, dtype_size)?;
        tensor.buffer.zero()?;
        Ok(tensor)
    }

    /// Create a tensor from host data.
    pub fn from_host(device: Arc<CudaDevice>, shape: &[usize], data: &[u8], dtype_size: usize) -> CudaResult<Self> {
        let numel: usize = shape.iter().product();
        let expected_size = numel * dtype_size;
        
        if data.len() != expected_size {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Data size {} doesn't match expected size {}",
                    data.len(),
                    expected_size
                ),
            });
        }

        let mut tensor = Self::new(device, shape, dtype_size)?;
        tensor.buffer.copy_from_host(data)?;
        Ok(tensor)
    }

    /// Create a tensor from f32 slice.
    pub fn from_f32(device: Arc<CudaDevice>, shape: &[usize], data: &[f32]) -> CudaResult<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f32>(),
            )
        };
        Self::from_host(device, shape, bytes, std::mem::size_of::<f32>())
    }

    /// Create a tensor from f64 slice.
    pub fn from_f64(device: Arc<CudaDevice>, shape: &[usize], data: &[f64]) -> CudaResult<Self> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<f64>(),
            )
        };
        Self::from_host(device, shape, bytes, std::mem::size_of::<f64>())
    }

    /// Get the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the number of elements.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the device.
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get raw pointer to GPU data.
    pub fn ptr(&self) -> *mut u8 {
        self.buffer.ptr()
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.numel * self.dtype_size
    }

    /// Copy data to host.
    pub fn to_host(&self) -> CudaResult<Vec<u8>> {
        let mut data = vec![0u8; self.size_bytes()];
        self.buffer.copy_to_host(&mut data)?;
        Ok(data)
    }

    /// Copy to f32 vector.
    pub fn to_f32(&self) -> CudaResult<Vec<f32>> {
        let bytes = self.to_host()?;
        let data = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.numel,
            )
        };
        Ok(data.to_vec())
    }

    /// Copy to f64 vector.
    pub fn to_f64(&self) -> CudaResult<Vec<f64>> {
        let bytes = self.to_host()?;
        let data = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f64,
                self.numel,
            )
        };
        Ok(data.to_vec())
    }

    /// Reshape the tensor (must have same number of elements).
    pub fn reshape(&self, new_shape: &[usize]) -> CudaResult<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Cannot reshape {} elements to {} elements",
                    self.numel, new_numel
                ),
            });
        }

        Ok(Self {
            buffer: self.buffer.clone(),
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
            numel: new_numel,
            dtype_size: self.dtype_size,
            device: self.device.clone(),
        })
    }

    // ========================================================================
    // Element-wise operations
    // ========================================================================

    /// Element-wise addition.
    pub fn add(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(other, kernels::BinaryOp::Add)
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(other, kernels::BinaryOp::Sub)
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(other, kernels::BinaryOp::Mul)
    }

    /// Element-wise division.
    pub fn div(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        self.binary_op(other, kernels::BinaryOp::Div)
    }

    /// Scalar addition.
    pub fn add_scalar(&self, scalar: f64) -> CudaResult<CudaTensor> {
        self.scalar_op(scalar, kernels::ScalarOp::Add)
    }

    /// Scalar multiplication.
    pub fn mul_scalar(&self, scalar: f64) -> CudaResult<CudaTensor> {
        self.scalar_op(scalar, kernels::ScalarOp::Mul)
    }

    /// Element-wise negation.
    pub fn neg(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Neg)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Exp)
    }

    /// Element-wise natural log.
    pub fn log(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Log)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Sqrt)
    }

    /// Element-wise ReLU.
    pub fn relu(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Relu)
    }

    /// Element-wise sigmoid.
    pub fn sigmoid(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Sigmoid)
    }

    /// Element-wise tanh.
    pub fn tanh(&self) -> CudaResult<CudaTensor> {
        self.unary_op(kernels::UnaryOp::Tanh)
    }

    /// Element-wise power.
    pub fn pow(&self, exponent: f64) -> CudaResult<CudaTensor> {
        self.scalar_op(exponent, kernels::ScalarOp::Pow)
    }

    // ========================================================================
    // Reduction operations
    // ========================================================================

    /// Sum all elements.
    pub fn sum(&self) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Sum, None)
    }

    /// Sum along axis.
    pub fn sum_axis(&self, axis: usize) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Sum, Some(axis))
    }

    /// Mean of all elements.
    pub fn mean(&self) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Mean, None)
    }

    /// Mean along axis.
    pub fn mean_axis(&self, axis: usize) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Mean, Some(axis))
    }

    /// Max of all elements.
    pub fn max(&self) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Max, None)
    }

    /// Min of all elements.
    pub fn min(&self) -> CudaResult<CudaTensor> {
        self.reduce_op(kernels::ReduceOp::Min, None)
    }

    // ========================================================================
    // Matrix operations
    // ========================================================================

    /// Matrix multiplication.
    pub fn matmul(&self, other: &CudaTensor) -> CudaResult<CudaTensor> {
        if self.ndim() < 2 || other.ndim() < 2 {
            return Err(CudaError::InvalidArgument {
                message: "matmul requires at least 2D tensors".to_string(),
            });
        }

        let m = self.shape[self.ndim() - 2];
        let k = self.shape[self.ndim() - 1];
        let n = other.shape[other.ndim() - 1];

        if other.shape[other.ndim() - 2] != k {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "matmul shape mismatch: [{}, {}] x [{}, {}]",
                    m, k,
                    other.shape[other.ndim() - 2], n
                ),
            });
        }

        let mut out_shape = self.shape[..self.ndim() - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);

        let output = CudaTensor::zeros(self.device.clone(), &out_shape, self.dtype_size)?;
        kernels::matmul(self, other, &output)?;
        Ok(output)
    }

    /// Transpose (swap last two dimensions).
    pub fn transpose(&self) -> CudaResult<CudaTensor> {
        if self.ndim() < 2 {
            return Err(CudaError::InvalidArgument {
                message: "transpose requires at least 2D tensor".to_string(),
            });
        }

        let mut new_shape = self.shape.clone();
        let n = new_shape.len();
        new_shape.swap(n - 1, n - 2);

        let output = CudaTensor::new(self.device.clone(), &new_shape, self.dtype_size)?;
        kernels::transpose(self, &output)?;
        Ok(output)
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    fn binary_op(&self, other: &CudaTensor, op: kernels::BinaryOp) -> CudaResult<CudaTensor> {
        // Check shapes are compatible (broadcasting)
        let out_shape = broadcast_shapes(&self.shape, &other.shape)?;
        let output = CudaTensor::new(self.device.clone(), &out_shape, self.dtype_size)?;
        kernels::binary_op(self, other, &output, op)?;
        Ok(output)
    }

    fn unary_op(&self, op: kernels::UnaryOp) -> CudaResult<CudaTensor> {
        let output = CudaTensor::new(self.device.clone(), &self.shape, self.dtype_size)?;
        kernels::unary_op(self, &output, op)?;
        Ok(output)
    }

    fn scalar_op(&self, scalar: f64, op: kernels::ScalarOp) -> CudaResult<CudaTensor> {
        let output = CudaTensor::new(self.device.clone(), &self.shape, self.dtype_size)?;
        kernels::scalar_op(self, &output, scalar, op)?;
        Ok(output)
    }

    fn reduce_op(&self, op: kernels::ReduceOp, axis: Option<usize>) -> CudaResult<CudaTensor> {
        let out_shape = match axis {
            Some(ax) => {
                let mut shape = self.shape.clone();
                shape[ax] = 1;
                shape
            }
            None => vec![1],
        };
        let output = CudaTensor::new(self.device.clone(), &out_shape, self.dtype_size)?;
        kernels::reduce_op(self, &output, op, axis)?;
        Ok(output)
    }
}

impl Clone for CudaTensor {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            numel: self.numel,
            dtype_size: self.dtype_size,
            device: self.device.clone(),
        }
    }
}

// ============================================================================
// Helper functions
// ============================================================================

fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

fn broadcast_shapes(a: &[usize], b: &[usize]) -> CudaResult<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0; max_len];

    for i in 0..max_len {
        let a_dim = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let b_dim = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };

        if a_dim == b_dim {
            result[i] = a_dim;
        } else if a_dim == 1 {
            result[i] = b_dim;
        } else if b_dim == 1 {
            result[i] = a_dim;
        } else {
            return Err(CudaError::InvalidArgument {
                message: format!(
                    "Cannot broadcast shapes {:?} and {:?}",
                    a, b
                ),
            });
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "simulate")]
    fn test_tensor_creation() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let tensor = CudaTensor::zeros(device, &[2, 3], 4).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    #[cfg(feature = "simulate")]
    fn test_tensor_from_host() {
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = CudaTensor::from_f32(device, &[2, 3], &data).unwrap();
        
        let result = tensor.to_f32().unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_broadcast_shapes() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[1, 4], &[3, 1]).unwrap(), vec![3, 4]);
        assert_eq!(broadcast_shapes(&[2, 3, 4], &[3, 4]).unwrap(), vec![2, 3, 4]);
    }
}
