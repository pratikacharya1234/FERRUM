//! N-dimensional tensor implementation.
//!
//! The [`Tensor`] type is the central data structure in FERRUM. It provides:
//!
//! - N-dimensional array storage with arbitrary dtypes
//! - Zero-copy views (slicing, reshaping, transposing)
//! - Reference-counted storage for efficient memory sharing
//! - Broadcasting for element-wise operations
//! - Optional gradient tracking for autograd
//!
//! ## Creating Tensors
//!
//! ```rust
//! use ferrum_core::prelude::*;
//!
//! // From data
//! let x = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
//!
//! // Filled tensors
//! let zeros = Tensor::zeros([2, 3], DType::F32, Device::Cpu);
//! let ones = Tensor::ones([2, 3], DType::F32, Device::Cpu);
//!
//! // Random tensors
//! let randn = Tensor::randn([128, 256], DType::F32, Device::Cpu);
//! ```
//!
//! ## Views vs Copies
//!
//! Operations that change the logical layout (reshape, transpose, slice)
//! create views that share the underlying storage:
//!
//! ```rust
//! use ferrum_core::prelude::*;
//!
//! let x = Tensor::randn([6], DType::F32, Device::Cpu);
//! let y = x.reshape([2, 3]).unwrap();  // View, no copy
//! assert_eq!(x.storage_ref_count(), 2);  // Shared storage
//! ```
//!
//! ## Broadcasting
//!
//! Element-wise operations automatically broadcast compatible shapes:
//!
//! ```rust
//! use ferrum_core::prelude::*;
//!
//! let x = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);
//! let y = Tensor::randn([4], DType::F32, Device::Cpu);
//! let z = x.add(&y).unwrap();  // y is broadcast to [2, 3, 4]
//! ```

use std::fmt;
use std::sync::Arc;

use parking_lot::RwLock;
use rand_distr::{Distribution, Normal, StandardNormal, Uniform};
use smallvec::SmallVec;

use crate::device::Device;
use crate::autograd_ops;
use crate::dtype::{DType, Element};
use crate::error::{FerrumError, Result};
use crate::shape::Shape;
use crate::storage::Storage;

/// Maximum dimensions for stack-allocated stride arrays.
const MAX_INLINE_DIMS: usize = 6;

/// N-dimensional tensor.
///
/// See module documentation for usage examples.
#[derive(Clone)]
pub struct Tensor {
    /// Underlying storage (shared via Arc).
    storage: Storage,
    /// Shape of the tensor.
    shape: Shape,
    /// Strides for each dimension (in elements, not bytes).
    strides: SmallVec<[usize; MAX_INLINE_DIMS]>,
    /// Offset into storage (in elements).
    offset: usize,
    /// Data type.
    dtype: DType,
    /// Device where data resides.
    device: Device,
    /// Whether this tensor requires gradient computation.
    requires_grad: bool,
    /// Accumulated gradient (for autograd).
    grad: Option<Arc<RwLock<Option<Tensor>>>>,
    /// Unique ID for autograd tracking.
    tensor_id: u64,
}

impl Tensor {
    // ========================================================================
    // CONSTRUCTORS
    // ========================================================================

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let size_bytes = numel * dtype.size_of();

        let storage = Storage::zeros(size_bytes, device).expect("Failed to allocate storage");

        let strides = shape.strides_contiguous();

        Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
            device,
            requires_grad: false,
            grad: None,
            tensor_id: autograd_ops::new_tensor_id(),
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let mut tensor = Self::zeros(shape, dtype, device);
        tensor.fill_scalar(1.0);
        tensor
    }

    /// Create a tensor filled with a scalar value.
    pub fn full(shape: impl Into<Shape>, value: f64, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let mut tensor = Self::zeros(shape, dtype, device);
        tensor.fill_scalar(value);
        tensor
    }

    /// Create a tensor with uninitialized storage.
    ///
    /// # Safety
    ///
    /// The contents are uninitialized. Reading before writing is undefined behavior.
    pub unsafe fn uninit(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let numel = shape.numel();
        let size_bytes = numel * dtype.size_of();

        let storage = Storage::uninit(size_bytes, device).expect("Failed to allocate storage");

        let strides = shape.strides_contiguous();

        Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype,
            device,
            requires_grad: false,
            grad: None,
            tensor_id: autograd_ops::new_tensor_id(),
        }
    }

    /// Create a tensor from a slice of data.
    pub fn from_slice<T: Element + bytemuck::Pod>(
        data: &[T],
        shape: impl Into<Shape>,
        device: Device,
    ) -> Result<Self> {
        let shape = shape.into();
        let numel = shape.numel();

        if data.len() != numel {
            return Err(FerrumError::InvalidShape {
                message: format!(
                    "Data length {} doesn't match shape {:?} (numel={})",
                    data.len(),
                    shape,
                    numel
                ),
            });
        }

        let storage = Storage::from_slice(data, device)?;
        let strides = shape.strides_contiguous();

        Ok(Self {
            storage,
            shape,
            strides,
            offset: 0,
            dtype: T::DTYPE,
            device,
            requires_grad: false,
            grad: None,
            tensor_id: autograd_ops::new_tensor_id(),
        })
    }

    /// Create a tensor from a Vec (takes ownership).
    pub fn from_vec<T: Element + bytemuck::Pod>(
        data: Vec<T>,
        shape: impl Into<Shape>,
        device: Device,
    ) -> Result<Self> {
        Self::from_slice(&data, shape, device)
    }

    /// Create a tensor with values from a standard normal distribution.
    pub fn randn(shape: impl Into<Shape>, dtype: DType, device: Device) -> Self {
        let shape = shape.into();
        let _numel = shape.numel();

        let tensor = Self::zeros(shape, dtype, device);
        let mut rng = rand::thread_rng();

        match dtype {
            DType::F32 => {
                let dist = StandardNormal;
                let mut guard = tensor.storage.write_as::<f32>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            DType::F64 => {
                let dist = StandardNormal;
                let mut guard = tensor.storage.write_as::<f64>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            _ => {
                panic!("randn only supports floating point dtypes, got {:?}", dtype);
            }
        }

        tensor
    }

    /// Create a tensor with values from a normal distribution.
    pub fn normal(
        shape: impl Into<Shape>,
        mean: f64,
        std: f64,
        dtype: DType,
        device: Device,
    ) -> Self {
        let shape = shape.into();
        let tensor = Self::zeros(shape, dtype, device);
        let mut rng = rand::thread_rng();

        match dtype {
            DType::F32 => {
                let dist = Normal::new(mean as f32, std as f32).unwrap();
                let mut guard = tensor.storage.write_as::<f32>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            DType::F64 => {
                let dist = Normal::new(mean, std).unwrap();
                let mut guard = tensor.storage.write_as::<f64>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            _ => {
                panic!(
                    "normal only supports floating point dtypes, got {:?}",
                    dtype
                );
            }
        }

        tensor
    }

    /// Create a tensor with values uniformly distributed in [low, high).
    pub fn uniform(
        shape: impl Into<Shape>,
        low: f64,
        high: f64,
        dtype: DType,
        device: Device,
    ) -> Self {
        let shape = shape.into();
        let tensor = Self::zeros(shape, dtype, device);
        let mut rng = rand::thread_rng();

        match dtype {
            DType::F32 => {
                let dist = Uniform::new(low as f32, high as f32);
                let mut guard = tensor.storage.write_as::<f32>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            DType::F64 => {
                let dist = Uniform::new(low, high);
                let mut guard = tensor.storage.write_as::<f64>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            DType::I32 => {
                let dist = Uniform::new(low as i32, high as i32);
                let mut guard = tensor.storage.write_as::<i32>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            DType::I64 => {
                let dist = Uniform::new(low as i64, high as i64);
                let mut guard = tensor.storage.write_as::<i64>();
                for x in guard.as_mut_slice().iter_mut() {
                    *x = dist.sample(&mut rng);
                }
            }
            _ => {
                panic!("uniform not supported for {:?}", dtype);
            }
        }

        tensor
    }

    /// Create an identity matrix.
    pub fn eye(n: usize, dtype: DType, device: Device) -> Self {
        let tensor = Self::zeros([n, n], dtype, device);

        match dtype {
            DType::F32 => {
                let mut guard = tensor.storage.write_as::<f32>();
                for i in 0..n {
                    guard[i * n + i] = 1.0;
                }
            }
            DType::F64 => {
                let mut guard = tensor.storage.write_as::<f64>();
                for i in 0..n {
                    guard[i * n + i] = 1.0;
                }
            }
            _ => {
                panic!("eye only supports floating point dtypes");
            }
        }

        tensor
    }

    /// Create a 1D tensor with evenly spaced values.
    pub fn arange(start: f64, end: f64, step: f64, dtype: DType, device: Device) -> Self {
        let n = ((end - start) / step).ceil() as usize;
        let tensor = Self::zeros([n], dtype, device);

        match dtype {
            DType::F32 => {
                let mut guard = tensor.storage.write_as::<f32>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = (start + i as f64 * step) as f32;
                }
            }
            DType::F64 => {
                let mut guard = tensor.storage.write_as::<f64>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = start + i as f64 * step;
                }
            }
            DType::I32 => {
                let mut guard = tensor.storage.write_as::<i32>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = (start + i as f64 * step) as i32;
                }
            }
            DType::I64 => {
                let mut guard = tensor.storage.write_as::<i64>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = (start + i as f64 * step) as i64;
                }
            }
            _ => panic!("arange not supported for {:?}", dtype),
        }

        tensor
    }

    /// Create a 1D tensor with n evenly spaced values between start and end.
    pub fn linspace(start: f64, end: f64, n: usize, dtype: DType, device: Device) -> Self {
        if n == 0 {
            return Self::zeros([0], dtype, device);
        }
        if n == 1 {
            return Self::full([1], start, dtype, device);
        }

        let step = (end - start) / (n - 1) as f64;
        let tensor = Self::zeros([n], dtype, device);

        match dtype {
            DType::F32 => {
                let mut guard = tensor.storage.write_as::<f32>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = (start + i as f64 * step) as f32;
                }
            }
            DType::F64 => {
                let mut guard = tensor.storage.write_as::<f64>();
                for (i, x) in guard.as_mut_slice().iter_mut().enumerate() {
                    *x = start + i as f64 * step;
                }
            }
            _ => panic!("linspace only supports floating point dtypes"),
        }

        tensor
    }

    /// Concatenate tensors along a dimension.
    /// 
    /// All tensors must have the same shape except in the concatenating dimension.
    /// 
    /// # Arguments
    /// * `tensors` - Slice of tensors to concatenate
    /// * `dim` - The dimension along which to concatenate
    /// 
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], [3], Device::Cpu)?;
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0], [3], Device::Cpu)?;
    /// let c = Tensor::cat(&[&a, &b], 0)?;  // Shape: [6]
    /// ```
    pub fn cat(tensors: &[&Tensor], dim: i64) -> Result<Self> {
        if tensors.is_empty() {
            return Err(FerrumError::InvalidShape {
                message: "cat requires at least one tensor".to_string(),
            });
        }
        
        let first = tensors[0];
        let ndim = first.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for {} dimensions", dim, ndim),
            });
        }
        
        // Validate all tensors have same dtype, device, and compatible shapes
        let dtype = first.dtype;
        let device = first.device;
        let base_shape = first.shape();
        
        let mut total_dim_size = base_shape[axis];
        
        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.dtype != dtype {
                return Err(FerrumError::InvalidShape {
                    message: format!("cat: dtype mismatch at tensor {}", i),
                });
            }
            if t.device != device {
                return Err(FerrumError::InvalidShape {
                    message: format!("cat: device mismatch at tensor {}", i),
                });
            }
            let t_shape = t.shape();
            if t_shape.len() != ndim {
                return Err(FerrumError::InvalidShape {
                    message: format!("cat: dimension mismatch at tensor {}", i),
                });
            }
            for (d, (&a, &b)) in base_shape.iter().zip(t_shape.iter()).enumerate() {
                if d != axis && a != b {
                    return Err(FerrumError::InvalidShape {
                        message: format!(
                            "cat: shape mismatch at tensor {} dim {} ({} vs {})",
                            i, d, a, b
                        ),
                    });
                }
            }
            total_dim_size += t_shape[axis];
        }
        
        // Build output shape
        let mut out_shape: Vec<usize> = base_shape.to_vec();
        out_shape[axis] = total_dim_size;
        let out_shape = Shape::from(out_shape);
        
        let mut result = Self::zeros(out_shape.clone(), dtype, device);
        
        // Calculate strides for copying
        let pre_axis_size: usize = base_shape[..axis].iter().product();
        let post_axis_size: usize = base_shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        match dtype {
            DType::F32 => {
                let mut dst = result.storage.write_as::<f32>();
                let mut axis_offset = 0;
                
                for t in tensors {
                    let t_contiguous = t.contiguous()?;
                    let src = t_contiguous.storage.read_as::<f32>();
                    let t_axis_size = t.shape()[axis];
                    
                    for pre in 0..pre_axis_size {
                        for a in 0..t_axis_size {
                            for post in 0..post_axis_size {
                                let src_idx = pre * t_axis_size * post_axis_size + a * post_axis_size + post;
                                let dst_idx = pre * total_dim_size * post_axis_size + (axis_offset + a) * post_axis_size + post;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                    axis_offset += t_axis_size;
                }
            }
            DType::F64 => {
                let mut dst = result.storage.write_as::<f64>();
                let mut axis_offset = 0;
                
                for t in tensors {
                    let t_contiguous = t.contiguous()?;
                    let src = t_contiguous.storage.read_as::<f64>();
                    let t_axis_size = t.shape()[axis];
                    
                    for pre in 0..pre_axis_size {
                        for a in 0..t_axis_size {
                            for post in 0..post_axis_size {
                                let src_idx = pre * t_axis_size * post_axis_size + a * post_axis_size + post;
                                let dst_idx = pre * total_dim_size * post_axis_size + (axis_offset + a) * post_axis_size + post;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                    axis_offset += t_axis_size;
                }
            }
            DType::I64 => {
                let mut dst = result.storage.write_as::<i64>();
                let mut axis_offset = 0;
                
                for t in tensors {
                    let t_contiguous = t.contiguous()?;
                    let src = t_contiguous.storage.read_as::<i64>();
                    let t_axis_size = t.shape()[axis];
                    
                    for pre in 0..pre_axis_size {
                        for a in 0..t_axis_size {
                            for post in 0..post_axis_size {
                                let src_idx = pre * t_axis_size * post_axis_size + a * post_axis_size + post;
                                let dst_idx = pre * total_dim_size * post_axis_size + (axis_offset + a) * post_axis_size + post;
                                dst[dst_idx] = src[src_idx];
                            }
                        }
                    }
                    axis_offset += t_axis_size;
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "cat",
                    dtype,
                });
            }
        }
        
        // Set requires_grad if any input requires grad
        result.requires_grad = tensors.iter().any(|t| t.requires_grad);
        
        // Record operation for autograd
        if result.requires_grad && autograd_ops::is_autograd_enabled() {
            let tensor_ids: Vec<u64> = tensors.iter().map(|t| t.tensor_id).collect();
            let sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
            autograd_ops::record_operation(
                Box::new(autograd_ops::CatBackward { 
                    dim: axis,
                    sizes,
                }),
                &tensor_ids,
                result.tensor_id,
                vec![],
            );
        }
        
        Ok(result)
    }

    /// Stack tensors along a new dimension.
    /// 
    /// All tensors must have exactly the same shape.
    /// 
    /// # Arguments
    /// * `tensors` - Slice of tensors to stack
    /// * `dim` - The dimension at which to insert the new axis
    /// 
    /// # Example
    /// ```ignore
    /// let a = Tensor::from_slice(&[1.0, 2.0], [2], Device::Cpu)?;
    /// let b = Tensor::from_slice(&[3.0, 4.0], [2], Device::Cpu)?;
    /// let c = Tensor::stack(&[&a, &b], 0)?;  // Shape: [2, 2]
    /// ```
    pub fn stack(tensors: &[&Tensor], dim: i64) -> Result<Self> {
        if tensors.is_empty() {
            return Err(FerrumError::InvalidShape {
                message: "stack requires at least one tensor".to_string(),
            });
        }
        
        let first = tensors[0];
        let ndim = first.shape.ndim() + 1;  // +1 for new dimension
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for {} dimensions", dim, ndim),
            });
        }
        
        // Unsqueeze all tensors at the specified dimension, then cat
        let unsqueezed: Result<Vec<_>> = tensors.iter()
            .map(|t| t.unsqueeze(dim))
            .collect();
        let unsqueezed = unsqueezed?;
        let refs: Vec<&Tensor> = unsqueezed.iter().collect();
        
        Self::cat(&refs, dim)
    }

    // ========================================================================
    // PROPERTIES
    // ========================================================================

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Get the shape as a Shape object.
    #[inline]
    pub fn shape_obj(&self) -> &Shape {
        &self.shape
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Get the total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the strides.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the data type.
    #[inline]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get the device.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Check if this tensor requires gradients.
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the size of a specific dimension.
    pub fn size(&self, dim: i64) -> Result<usize> {
        self.shape.dim(dim)
    }

    /// Check if the tensor is contiguous in memory.
    pub fn is_contiguous(&self) -> bool {
        self.shape.is_contiguous(&self.strides)
    }

    /// Check if this is a scalar (0-dimensional tensor).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.shape.is_scalar()
    }

    /// Get the storage reference count.
    #[inline]
    pub fn storage_ref_count(&self) -> usize {
        self.storage.ref_count()
    }

    // ========================================================================
    // GRADIENT METHODS
    // ========================================================================

    /// Enable gradient computation for this tensor.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        if requires_grad && !self.dtype.supports_grad() {
            panic!(
                "Cannot require gradients for non-floating point dtype {:?}",
                self.dtype
            );
        }
        self.requires_grad = requires_grad;
        if requires_grad && self.grad.is_none() {
            self.grad = Some(Arc::new(RwLock::new(None)));
        }
    }

    /// Create a new tensor with gradient tracking enabled.
    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.set_requires_grad(requires_grad);
        self
    }

    /// Get the gradient tensor (if computed).
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.as_ref().and_then(|g| g.read().clone())
    }

    /// Set the gradient tensor.
    pub fn set_grad(&self, grad: Option<Tensor>) {
        if let Some(ref g) = self.grad {
            *g.write() = grad;
        }
    }

    /// Zero the gradient.
    pub fn zero_grad(&self) {
        self.set_grad(None);
    }

    /// Accumulate gradient (used during backward pass).
    #[allow(dead_code)]
    pub(crate) fn accumulate_grad(&self, grad: &Tensor) -> Result<()> {
        if let Some(ref g) = self.grad {
            let mut guard = g.write();
            if let Some(ref mut existing) = *guard {
                // Add to existing gradient
                *existing = existing.add(grad)?;
            } else {
                *guard = Some(grad.clone());
            }
        }
        Ok(())
    }

    /// Detach this tensor from the computation graph.
    pub fn detach(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: false,
            grad: None,
            tensor_id: autograd_ops::new_tensor_id(),
        }
    }

    // ========================================================================
    // SHAPE MANIPULATION
    // ========================================================================

    /// Reshape the tensor to a new shape (view if possible).
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();

        if !self.shape.can_reshape_to(&new_shape) {
            return Err(FerrumError::ReshapeError {
                src_numel: self.numel(),
                target: Box::new(new_shape.clone()),
                target_numel: new_shape.numel(),
            });
        }

        // If contiguous, we can create a view
        if self.is_contiguous() {
            let new_strides = new_shape.strides_contiguous();
            return Ok(Self {
                storage: self.storage.clone(),
                shape: new_shape,
                strides: new_strides,
                offset: self.offset,
                dtype: self.dtype,
                device: self.device,
                requires_grad: self.requires_grad,
                grad: self.grad.clone(),
                tensor_id: self.tensor_id, // Keep same ID for views
            });
        }

        // Non-contiguous: need to make contiguous first
        let contiguous = self.contiguous()?;
        contiguous.reshape(new_shape)
    }

    /// Flatten the tensor to 1D.
    pub fn flatten(&self) -> Result<Self> {
        self.reshape([self.numel()])
    }

    /// Ensure the tensor is contiguous in memory.
    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        // Need to copy data to new contiguous storage
        let mut new_tensor = unsafe { Self::uninit(self.shape.clone(), self.dtype, self.device) };

        // Copy data element by element respecting strides
        self.copy_to_contiguous(&mut new_tensor)?;

        new_tensor.requires_grad = self.requires_grad;
        new_tensor.grad = self.grad.clone();

        Ok(new_tensor)
    }

    /// Copy data to a contiguous tensor (helper).
    fn copy_to_contiguous(&self, dst: &mut Tensor) -> Result<()> {
        match self.dtype {
            DType::F32 => self.copy_to_contiguous_typed::<f32>(dst),
            DType::F64 => self.copy_to_contiguous_typed::<f64>(dst),
            DType::I32 => self.copy_to_contiguous_typed::<i32>(dst),
            DType::I64 => self.copy_to_contiguous_typed::<i64>(dst),
            _ => Err(FerrumError::not_implemented(format!(
                "copy_to_contiguous for {:?}",
                self.dtype
            ))),
        }
    }

    fn copy_to_contiguous_typed<T: Copy + bytemuck::Pod>(&self, dst: &mut Tensor) -> Result<()> {
        let src_data = self.storage.read_as::<T>();
        let mut dst_data = dst.storage.write_as::<T>();

        let ndim = self.ndim();
        let shape = self.shape();
        let strides = self.strides();
        let offset = self.offset;

        // Iterate through all elements
        let mut coords = vec![0usize; ndim];
        let numel = self.numel();

        for dst_idx in 0..numel {
            // Calculate source index using strides
            let mut src_idx = offset;
            for d in 0..ndim {
                src_idx += coords[d] * strides[d];
            }

            dst_data[dst_idx] = src_data[src_idx];

            // Increment coordinates
            for d in (0..ndim).rev() {
                coords[d] += 1;
                if coords[d] < shape[d] {
                    break;
                }
                coords[d] = 0;
            }
        }

        Ok(())
    }

    /// Transpose two dimensions.
    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<Self> {
        let ndim = self.ndim();
        let d0 = self.shape.normalize_dim(dim0)?;
        let d1 = self.shape.normalize_dim(dim1)?;

        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(d0, d1);

        let new_shape = self.shape.transpose(&perm)?;

        let mut new_strides = self.strides.clone();
        new_strides.swap(d0, d1);

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            tensor_id: self.tensor_id,
        })
    }

    /// Transpose (for 2D tensors, swaps dimensions).
    pub fn t(&self) -> Result<Self> {
        if self.ndim() != 2 {
            return Err(FerrumError::InvalidShape {
                message: format!("t() requires 2D tensor, got {}D", self.ndim()),
            });
        }
        self.transpose(0, 1)
    }

    /// Permute dimensions.
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let new_shape = self.shape.transpose(dims)?;
        let new_strides: SmallVec<[usize; MAX_INLINE_DIMS]> =
            dims.iter().map(|&d| self.strides[d]).collect();

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            tensor_id: self.tensor_id,
        })
    }

    /// Add a dimension of size 1.
    pub fn unsqueeze(&self, dim: i64) -> Result<Self> {
        let new_shape = self.shape.unsqueeze(dim)?;
        let actual_dim = if dim < 0 {
            (self.ndim() as i64 + dim + 1) as usize
        } else {
            dim as usize
        };

        let mut new_strides = self.strides.clone();
        // Insert stride of 0 for the new dimension (or calculate based on next dim)
        let new_stride = if actual_dim < self.ndim() {
            self.strides[actual_dim] * self.shape[actual_dim]
        } else {
            1
        };
        new_strides.insert(actual_dim, new_stride);

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            tensor_id: self.tensor_id,
        })
    }

    /// Remove dimensions of size 1.
    pub fn squeeze(&self, dim: Option<i64>) -> Result<Self> {
        let new_shape = self.shape.squeeze(dim)?;

        // Filter out strides for squeezed dimensions
        let new_strides: SmallVec<[usize; MAX_INLINE_DIMS]> = match dim {
            None => self
                .shape
                .dims()
                .iter()
                .zip(self.strides.iter())
                .filter(|(&d, _)| d != 1)
                .map(|(_, &s)| s)
                .collect(),
            Some(d) => {
                let actual_dim = self.shape.normalize_dim(d)?;
                let mut strides = self.strides.clone();
                strides.remove(actual_dim);
                strides
            }
        };

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            tensor_id: self.tensor_id,
        })
    }

    /// Expand tensor to a larger size (broadcasting).
    pub fn expand(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        let new_strides = self.shape.broadcast_strides(&new_shape)?;

        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: self.grad.clone(),
            tensor_id: self.tensor_id,
        })
    }

    /// Returns a new tensor that is a narrowed version of this tensor.
    /// 
    /// The dimension `dim` is narrowed from `start` to `start + length`.
    /// 
    /// # Arguments
    /// * `dim` - The dimension along which to narrow
    /// * `start` - The starting index
    /// * `length` - The length of the narrowed dimension
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let shape = self.shape();
        
        if dim >= shape.len() {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for {} dimensions", dim, shape.len()),
            });
        }
        
        if start + length > shape[dim] {
            return Err(FerrumError::InvalidShape {
                message: format!(
                    "narrow: start ({}) + length ({}) exceeds dim size ({})",
                    start, length, shape[dim]
                ),
            });
        }
        
        // Calculate new shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        new_dims[dim] = length;
        let new_shape = Shape::from(new_dims);
        
        // Calculate new offset
        let new_offset = self.offset + start * self.strides[dim];
        
        Ok(Self {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides.clone(),
            offset: new_offset,
            dtype: self.dtype,
            device: self.device,
            requires_grad: self.requires_grad,
            grad: None, // Grad not propagated for views
            tensor_id: autograd_ops::new_tensor_id(), // New ID for the view
        })
    }

    // ========================================================================
    // DATA ACCESS
    // ========================================================================

    /// Get a single element as f64.
    pub fn item(&self) -> Result<f64> {
        if self.numel() != 1 {
            return Err(FerrumError::InvalidShape {
                message: format!(
                    "item() requires tensor with 1 element, got {}",
                    self.numel()
                ),
            });
        }

        match self.dtype {
            DType::F32 => {
                let data = self.storage.read_as::<f32>();
                Ok(data[self.offset] as f64)
            }
            DType::F64 => {
                let data = self.storage.read_as::<f64>();
                Ok(data[self.offset])
            }
            DType::I32 => {
                let data = self.storage.read_as::<i32>();
                Ok(data[self.offset] as f64)
            }
            DType::I64 => {
                let data = self.storage.read_as::<i64>();
                Ok(data[self.offset] as f64)
            }
            _ => Err(FerrumError::not_implemented(format!(
                "item() for {:?}",
                self.dtype
            ))),
        }
    }

    /// Convert to a Vec (copies data).
    pub fn to_vec<T: Element + bytemuck::Pod>(&self) -> Result<Vec<T>> {
        if T::DTYPE != self.dtype {
            return Err(FerrumError::DTypeMismatch {
                operation: "to_vec",
                expected: T::DTYPE,
                actual: self.dtype,
            });
        }

        let contiguous = self.contiguous()?;
        let data = contiguous.storage.read_as::<T>();
        Ok(data.as_slice()[contiguous.offset..contiguous.offset + contiguous.numel()].to_vec())
    }

    /// Fill tensor with a scalar value.
    fn fill_scalar(&mut self, value: f64) {
        match self.dtype {
            DType::F32 => {
                let mut data = self.storage.write_as::<f32>();
                for x in data.as_mut_slice().iter_mut() {
                    *x = value as f32;
                }
            }
            DType::F64 => {
                let mut data = self.storage.write_as::<f64>();
                for x in data.as_mut_slice().iter_mut() {
                    *x = value;
                }
            }
            DType::I32 => {
                let mut data = self.storage.write_as::<i32>();
                for x in data.as_mut_slice().iter_mut() {
                    *x = value as i32;
                }
            }
            DType::I64 => {
                let mut data = self.storage.write_as::<i64>();
                for x in data.as_mut_slice().iter_mut() {
                    *x = value as i64;
                }
            }
            _ => panic!("fill_scalar not supported for {:?}", self.dtype),
        }
    }

    // ========================================================================
    // TYPE/DEVICE CONVERSION
    // ========================================================================

    /// Convert tensor to a different dtype.
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype == dtype {
            return Ok(self.clone());
        }

        let contiguous = self.contiguous()?;
        let mut new_tensor = Self::zeros(self.shape.clone(), dtype, self.device);

        // Type conversion dispatch
        match (self.dtype, dtype) {
            (DType::F32, DType::F64) => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = new_tensor.storage.write_as::<f64>();
                for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = *s as f64;
                }
            }
            (DType::F64, DType::F32) => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = new_tensor.storage.write_as::<f32>();
                for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = *s as f32;
                }
            }
            (DType::F32, DType::I32) => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = new_tensor.storage.write_as::<i32>();
                for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = *s as i32;
                }
            }
            (DType::I32, DType::F32) => {
                let src = contiguous.storage.read_as::<i32>();
                let mut dst = new_tensor.storage.write_as::<f32>();
                for (d, s) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = *s as f32;
                }
            }
            _ => {
                return Err(FerrumError::not_implemented(format!(
                    "dtype conversion from {:?} to {:?}",
                    self.dtype, dtype
                )));
            }
        }

        new_tensor.requires_grad = self.requires_grad && dtype.supports_grad();
        Ok(new_tensor)
    }

    /// Move tensor to a different device.
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }

        // For now, only CPU is supported
        if !device.is_cpu() {
            return Err(FerrumError::not_implemented(format!(
                "Device transfer to {:?}",
                device
            )));
        }

        // CPU to CPU is a no-op
        Ok(self.clone())
    }

    // ========================================================================
    // BASIC ARITHMETIC WITH AUTOGRAD SUPPORT
    // ========================================================================

    /// Element-wise addition with automatic differentiation.
    pub fn add(&self, other: &Tensor) -> Result<Self> {
        let result = self.binary_op(other, "add", |a, b| a + b, |a, b| a + b)?;
        
        // Record operation for autograd
        if (self.requires_grad || other.requires_grad) && autograd_ops::is_autograd_enabled() {
            let backward = autograd_ops::AddBackward {
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
            };
            autograd_ops::record_operation(
                Box::new(backward),
                &[self.tensor_id, other.tensor_id],
                result.tensor_id,
                vec![self.clone(), other.clone()], // Save inputs to set gradients
            );
        }
        
        Ok(result)
    }

    /// Element-wise subtraction with automatic differentiation.
    pub fn sub(&self, other: &Tensor) -> Result<Self> {
        let result = self.binary_op(other, "sub", |a, b| a - b, |a, b| a - b)?;
        
        // Record operation for autograd
        if (self.requires_grad || other.requires_grad) && autograd_ops::is_autograd_enabled() {
            let backward = autograd_ops::SubBackward {
                a_shape: self.shape.clone(),
                b_shape: other.shape.clone(),
            };
            autograd_ops::record_operation(
                Box::new(backward),
                &[self.tensor_id, other.tensor_id],
                result.tensor_id,
                vec![self.clone(), other.clone()], // Save inputs to set gradients
            );
        }
        
        Ok(result)
    }

    /// Element-wise multiplication with automatic differentiation.
    pub fn mul(&self, other: &Tensor) -> Result<Self> {
        let result = self.binary_op(other, "mul", |a, b| a * b, |a, b| a * b)?;
        
        // Record operation for autograd - save inputs for backward
        if (self.requires_grad || other.requires_grad) && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::MulBackward),
                &[self.tensor_id, other.tensor_id],
                result.tensor_id,
                vec![self.clone(), other.clone()],
            );
        }
        
        Ok(result)
    }

    /// Element-wise division with automatic differentiation.
    pub fn div(&self, other: &Tensor) -> Result<Self> {
        let result = self.binary_op(other, "div", |a, b| a / b, |a, b| a / b)?;
        
        // Record operation for autograd - save inputs for backward
        if (self.requires_grad || other.requires_grad) && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::DivBackward),
                &[self.tensor_id, other.tensor_id],
                result.tensor_id,
                vec![self.clone(), other.clone()],
            );
        }
        
        Ok(result)
    }

    /// Add a scalar.
    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        self.scalar_op(scalar, "add_scalar", |a, s| a + s, |a, s| a + s)
    }

    /// Multiply by a scalar.
    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        self.scalar_op(scalar, "mul_scalar", |a, s| a * s, |a, s| a * s)
    }

    // ========================================================================
    // IN-PLACE OPERATIONS (for optimizer updates, no autograd)
    // ========================================================================

    /// In-place subtraction: self = self - other
    ///
    /// This operation does NOT record to autograd and should only be used
    /// for optimizer parameter updates.
    pub fn sub_inplace(&self, other: &Tensor) -> Result<()> {
        // Verify shapes match exactly for in-place operation
        if self.shape != other.shape {
            return Err(FerrumError::shape_mismatch(
                "sub_inplace",
                format!("{:?}", self.shape.dims()),
                format!("{:?}", other.shape.dims()),
            ));
        }

        // Verify dtype matches
        if self.dtype != other.dtype {
            return Err(FerrumError::DTypeMismatch {
                operation: "sub_inplace",
                expected: self.dtype,
                actual: other.dtype,
            });
        }

        let other_contiguous = other.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let mut self_data = self.storage.write_as::<f32>();
                let other_data = other_contiguous.storage.read_as::<f32>();
                
                for (s, o) in self_data.as_mut_slice().iter_mut().zip(other_data.as_slice().iter()) {
                    *s -= *o;
                }
            }
            DType::F64 => {
                let mut self_data = self.storage.write_as::<f64>();
                let other_data = other_contiguous.storage.read_as::<f64>();
                
                for (s, o) in self_data.as_mut_slice().iter_mut().zip(other_data.as_slice().iter()) {
                    *s -= *o;
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "sub_inplace",
                    dtype: self.dtype,
                });
            }
        }

        Ok(())
    }

    /// In-place addition: self = self + other
    ///
    /// This operation does NOT record to autograd and should only be used
    /// for optimizer parameter updates or gradient accumulation.
    pub fn add_inplace(&self, other: &Tensor) -> Result<()> {
        if self.shape != other.shape {
            return Err(FerrumError::shape_mismatch(
                "add_inplace",
                format!("{:?}", self.shape.dims()),
                format!("{:?}", other.shape.dims()),
            ));
        }

        if self.dtype != other.dtype {
            return Err(FerrumError::DTypeMismatch {
                operation: "add_inplace",
                expected: self.dtype,
                actual: other.dtype,
            });
        }

        let other_contiguous = other.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let mut self_data = self.storage.write_as::<f32>();
                let other_data = other_contiguous.storage.read_as::<f32>();
                
                for (s, o) in self_data.as_mut_slice().iter_mut().zip(other_data.as_slice().iter()) {
                    *s += *o;
                }
            }
            DType::F64 => {
                let mut self_data = self.storage.write_as::<f64>();
                let other_data = other_contiguous.storage.read_as::<f64>();
                
                for (s, o) in self_data.as_mut_slice().iter_mut().zip(other_data.as_slice().iter()) {
                    *s += *o;
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "add_inplace",
                    dtype: self.dtype,
                });
            }
        }

        Ok(())
    }

    /// In-place scalar multiplication: self = self * scalar
    pub fn mul_scalar_inplace(&self, scalar: f64) -> Result<()> {
        match self.dtype {
            DType::F32 => {
                let mut self_data = self.storage.write_as::<f32>();
                let s = scalar as f32;
                for v in self_data.as_mut_slice().iter_mut() {
                    *v *= s;
                }
            }
            DType::F64 => {
                let mut self_data = self.storage.write_as::<f64>();
                for v in self_data.as_mut_slice().iter_mut() {
                    *v *= scalar;
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "mul_scalar_inplace",
                    dtype: self.dtype,
                });
            }
        }

        Ok(())
    }

    /// Generic binary operation with broadcasting.
    fn binary_op<F32Op, F64Op>(
        &self,
        other: &Tensor,
        op_name: &'static str,
        f32_op: F32Op,
        f64_op: F64Op,
    ) -> Result<Self>
    where
        F32Op: Fn(f32, f32) -> f32,
        F64Op: Fn(f64, f64) -> f64,
    {
        // Check device and dtype compatibility
        if self.device != other.device {
            return Err(FerrumError::DeviceMismatch {
                operation: op_name,
                device1: self.device,
                device2: other.device,
            });
        }
        if self.dtype != other.dtype {
            return Err(FerrumError::DTypeMismatch {
                operation: op_name,
                expected: self.dtype,
                actual: other.dtype,
            });
        }

        // Calculate broadcast shape
        let result_shape = Shape::broadcast(&self.shape, &other.shape)?;
        let mut result = Self::zeros(result_shape.clone(), self.dtype, self.device);

        // Get contiguous data
        let a = self.contiguous()?;
        let b = other.contiguous()?;

        // Calculate broadcast strides
        let a_strides = a.shape.broadcast_strides(&result_shape)?;
        let b_strides = b.shape.broadcast_strides(&result_shape)?;

        match self.dtype {
            DType::F32 => {
                let a_data = a.storage.read_as::<f32>();
                let b_data = b.storage.read_as::<f32>();
                let mut r_data = result.storage.write_as::<f32>();

                broadcast_binary_op(
                    a_data.as_slice(),
                    b_data.as_slice(),
                    r_data.as_mut_slice(),
                    result_shape.dims(),
                    &a_strides,
                    &b_strides,
                    &f32_op,
                );
            }
            DType::F64 => {
                let a_data = a.storage.read_as::<f64>();
                let b_data = b.storage.read_as::<f64>();
                let mut r_data = result.storage.write_as::<f64>();

                broadcast_binary_op(
                    a_data.as_slice(),
                    b_data.as_slice(),
                    r_data.as_mut_slice(),
                    result_shape.dims(),
                    &a_strides,
                    &b_strides,
                    &f64_op,
                );
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: op_name,
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }

    /// Generic scalar operation.
    fn scalar_op<F32Op, F64Op>(
        &self,
        scalar: f64,
        op_name: &'static str,
        f32_op: F32Op,
        f64_op: F64Op,
    ) -> Result<Self>
    where
        F32Op: Fn(f32, f32) -> f32,
        F64Op: Fn(f64, f64) -> f64,
    {
        let mut result = Self::zeros(self.shape.clone(), self.dtype, self.device);
        let contiguous = self.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<f32>();
                let s = scalar as f32;
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = f32_op(*a, s);
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<f64>();
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = f64_op(*a, scalar);
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: op_name,
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// Negate the tensor with automatic differentiation.
    pub fn neg(&self) -> Result<Self> {
        let result = self.mul_scalar(-1.0)?;
        
        // Record operation for autograd
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::NegBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![],
            );
        }
        
        Ok(result)
    }

    /// Sum all elements with automatic differentiation.
    pub fn sum(&self) -> Result<Self> {
        let contiguous = self.contiguous()?;
        let mut result = Self::zeros(Shape::scalar(), self.dtype, self.device);

        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let sum: f32 = src.as_slice().iter().sum();
                result.storage.write_as::<f32>()[0] = sum;
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let sum: f64 = src.as_slice().iter().sum();
                result.storage.write_as::<f64>()[0] = sum;
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "sum",
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad;
        
        // Record operation for autograd
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::SumBackward {
                    input_shape: self.shape.clone(),
                }),
                &[self.tensor_id],
                result.tensor_id,
                vec![],
            );
        }
        
        Ok(result)
    }

    /// Mean of all elements with automatic differentiation.
    pub fn mean(&self) -> Result<Self> {
        let contiguous = self.contiguous()?;
        let n = self.numel() as f64;
        let mut result = Self::zeros(Shape::scalar(), self.dtype, self.device);

        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let sum: f32 = src.as_slice().iter().sum();
                result.storage.write_as::<f32>()[0] = sum / (n as f32);
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let sum: f64 = src.as_slice().iter().sum();
                result.storage.write_as::<f64>()[0] = sum / n;
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "mean",
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad;
        
        // Record operation for autograd
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::MeanBackward {
                    input_shape: self.shape.clone(),
                }),
                &[self.tensor_id],
                result.tensor_id,
                vec![],
            );
        }
        
        Ok(result)
    }

    /// Sum along a specific dimension with automatic differentiation.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    pub fn sum_dim(&self, dim: i64, keepdim: bool) -> Result<Self> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let contiguous = self.contiguous()?;
        let shape = self.shape();
        
        // Calculate output shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from(new_dims)
        };
        
        // Calculate strides for iteration
        let pre_axis_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let post_axis_size: usize = shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        let mut result = Self::zeros(new_shape.clone(), self.dtype, self.device);
        
        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<f32>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut sum = 0.0f32;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            sum += src[idx];
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = sum;
                    }
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<f64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut sum = 0.0f64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            sum += src[idx];
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = sum;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "sum_dim",
                    dtype: self.dtype,
                });
            }
        }
        
        result.requires_grad = self.requires_grad;
        
        // Record operation for autograd
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::SumDimBackward {
                    input_shape: self.shape.clone(),
                    dim: axis,
                    keepdim,
                }),
                &[self.tensor_id],
                result.tensor_id,
                vec![],
            );
        }
        
        Ok(result)
    }

    /// Mean along a specific dimension with automatic differentiation.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    pub fn mean_dim(&self, dim: i64, keepdim: bool) -> Result<Self> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let n = self.shape()[axis] as f64;
        let sum_result = self.sum_dim(dim, keepdim)?;
        
        // Divide by the size of the reduced dimension
        let mut result = sum_result.mul_scalar(1.0 / n)?;
        result.requires_grad = self.requires_grad;
        
        // We need to override the recorded operation for proper gradient
        // The sum_dim already recorded, but mean needs different gradient
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            // Note: The gradient is handled by the sum_dim backward multiplied by 1/n
            // which is correct because mean = sum / n, so d(mean)/dx = (1/n) * d(sum)/dx
        }
        
        Ok(result)
    }

    /// Find the indices of maximum values along a dimension.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    pub fn argmax(&self, dim: i64, keepdim: bool) -> Result<Self> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let contiguous = self.contiguous()?;
        let shape = self.shape();
        
        // Calculate output shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from(new_dims)
        };
        
        // Calculate strides for iteration
        let pre_axis_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let post_axis_size: usize = shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        // Argmax returns indices, so we use I64 dtype for output
        let mut result = Self::zeros(new_shape.clone(), DType::I64, self.device);
        
        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] > max_val {
                                max_val = src[idx];
                                max_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = max_idx;
                    }
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] > max_val {
                                max_val = src[idx];
                                max_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = max_idx;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "argmax",
                    dtype: self.dtype,
                });
            }
        }
        
        // argmax is not differentiable
        result.requires_grad = false;
        Ok(result)
    }

    /// Find the indices of minimum values along a dimension.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    pub fn argmin(&self, dim: i64, keepdim: bool) -> Result<Self> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let contiguous = self.contiguous()?;
        let shape = self.shape();
        
        // Calculate output shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from(new_dims)
        };
        
        // Calculate strides for iteration
        let pre_axis_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let post_axis_size: usize = shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        // Argmin returns indices, so we use I64 dtype for output
        let mut result = Self::zeros(new_shape.clone(), DType::I64, self.device);
        
        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut min_val = f32::INFINITY;
                        let mut min_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] < min_val {
                                min_val = src[idx];
                                min_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = min_idx;
                    }
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut min_val = f64::INFINITY;
                        let mut min_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] < min_val {
                                min_val = src[idx];
                                min_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        dst[out_idx] = min_idx;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "argmin",
                    dtype: self.dtype,
                });
            }
        }
        
        // argmin is not differentiable
        result.requires_grad = false;
        Ok(result)
    }

    /// Find maximum values along a dimension.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    /// 
    /// # Returns
    /// Tuple of (max_values, indices)
    pub fn max_dim(&self, dim: i64, keepdim: bool) -> Result<(Self, Self)> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let contiguous = self.contiguous()?;
        let shape = self.shape();
        
        // Calculate output shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from(new_dims)
        };
        
        // Calculate strides for iteration
        let pre_axis_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let post_axis_size: usize = shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        let mut values = Self::zeros(new_shape.clone(), self.dtype, self.device);
        let mut indices = Self::zeros(new_shape.clone(), DType::I64, self.device);
        
        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut val_dst = values.storage.write_as::<f32>();
                let mut idx_dst = indices.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] > max_val {
                                max_val = src[idx];
                                max_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        val_dst[out_idx] = max_val;
                        idx_dst[out_idx] = max_idx;
                    }
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut val_dst = values.storage.write_as::<f64>();
                let mut idx_dst = indices.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut max_val = f64::NEG_INFINITY;
                        let mut max_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] > max_val {
                                max_val = src[idx];
                                max_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        val_dst[out_idx] = max_val;
                        idx_dst[out_idx] = max_idx;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "max_dim",
                    dtype: self.dtype,
                });
            }
        }
        
        values.requires_grad = self.requires_grad;
        indices.requires_grad = false;
        
        Ok((values, indices))
    }

    /// Find minimum values along a dimension.
    /// 
    /// # Arguments
    /// * `dim` - The dimension to reduce
    /// * `keepdim` - If true, keeps the reduced dimension with size 1
    /// 
    /// # Returns
    /// Tuple of (min_values, indices)
    pub fn min_dim(&self, dim: i64, keepdim: bool) -> Result<(Self, Self)> {
        let ndim = self.shape.ndim();
        let axis = if dim < 0 {
            (ndim as i64 + dim) as usize
        } else {
            dim as usize
        };
        
        if axis >= ndim {
            return Err(FerrumError::InvalidShape {
                message: format!("dim {} out of bounds for shape {:?}", dim, self.shape()),
            });
        }
        
        let contiguous = self.contiguous()?;
        let shape = self.shape();
        
        // Calculate output shape
        let mut new_dims: Vec<usize> = shape.to_vec();
        if keepdim {
            new_dims[axis] = 1;
        } else {
            new_dims.remove(axis);
        }
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::from(new_dims)
        };
        
        // Calculate strides for iteration
        let pre_axis_size: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let post_axis_size: usize = shape[axis + 1..].iter().product();
        let pre_axis_size = if pre_axis_size == 0 { 1 } else { pre_axis_size };
        let post_axis_size = if post_axis_size == 0 { 1 } else { post_axis_size };
        
        let mut values = Self::zeros(new_shape.clone(), self.dtype, self.device);
        let mut indices = Self::zeros(new_shape.clone(), DType::I64, self.device);
        
        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut val_dst = values.storage.write_as::<f32>();
                let mut idx_dst = indices.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut min_val = f32::INFINITY;
                        let mut min_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] < min_val {
                                min_val = src[idx];
                                min_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        val_dst[out_idx] = min_val;
                        idx_dst[out_idx] = min_idx;
                    }
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut val_dst = values.storage.write_as::<f64>();
                let mut idx_dst = indices.storage.write_as::<i64>();
                
                for pre in 0..pre_axis_size {
                    for post in 0..post_axis_size {
                        let mut min_val = f64::INFINITY;
                        let mut min_idx = 0i64;
                        for a in 0..axis_size {
                            let idx = pre * axis_size * post_axis_size + a * post_axis_size + post;
                            if src[idx] < min_val {
                                min_val = src[idx];
                                min_idx = a as i64;
                            }
                        }
                        let out_idx = pre * post_axis_size + post;
                        val_dst[out_idx] = min_val;
                        idx_dst[out_idx] = min_idx;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "min_dim",
                    dtype: self.dtype,
                });
            }
        }
        
        values.requires_grad = self.requires_grad;
        indices.requires_grad = false;
        
        Ok((values, indices))
    }

    /// Element-wise power with automatic differentiation.
    pub fn pow(&self, exponent: f64) -> Result<Self> {
        let mut result = Self::zeros(self.shape.clone(), self.dtype, self.device);
        let contiguous = self.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<f32>();
                let exp = exponent as f32;
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = a.powf(exp);
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<f64>();
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = a.powf(exponent);
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "pow",
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad;
        
        // Record operation for autograd
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::PowBackward { exponent }),
                &[self.tensor_id],
                result.tensor_id,
                vec![self.clone()],
            );
        }
        
        Ok(result)
    }

    /// Square root.
    pub fn sqrt(&self) -> Result<Self> {
        self.pow(0.5)
    }

    /// Exponential with automatic differentiation.
    pub fn exp(&self) -> Result<Self> {
        let result = self.unary_op("exp", f32::exp, f64::exp)?;
        
        // Record operation for autograd - save result for backward
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::ExpBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![result.clone()], // Save exp(x) for backward
            );
        }
        
        Ok(result)
    }

    /// Natural logarithm with automatic differentiation.
    pub fn log(&self) -> Result<Self> {
        let result = self.unary_op("log", f32::ln, f64::ln)?;
        
        // Record operation for autograd - save input for backward
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::LogBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![self.clone()],
            );
        }
        
        Ok(result)
    }

    /// Generic unary operation.
    fn unary_op<F32Op, F64Op>(
        &self,
        op_name: &'static str,
        f32_op: F32Op,
        f64_op: F64Op,
    ) -> Result<Self>
    where
        F32Op: Fn(f32) -> f32,
        F64Op: Fn(f64) -> f64,
    {
        let mut result = Self::zeros(self.shape.clone(), self.dtype, self.device);
        let contiguous = self.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let src = contiguous.storage.read_as::<f32>();
                let mut dst = result.storage.write_as::<f32>();
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = f32_op(*a);
                }
            }
            DType::F64 => {
                let src = contiguous.storage.read_as::<f64>();
                let mut dst = result.storage.write_as::<f64>();
                for (d, a) in dst.as_mut_slice().iter_mut().zip(src.as_slice().iter()) {
                    *d = f64_op(*a);
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: op_name,
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad;
        Ok(result)
    }

    /// ReLU activation with automatic differentiation.
    pub fn relu(&self) -> Result<Self> {
        let result = self.unary_op("relu", |x| x.max(0.0), |x| x.max(0.0))?;
        
        // Record operation for autograd - save input for backward mask
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::ReLUBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![self.clone()],
            );
        }
        
        Ok(result)
    }

    /// Sigmoid activation with automatic differentiation.
    pub fn sigmoid(&self) -> Result<Self> {
        let result = self.unary_op(
            "sigmoid",
            |x| 1.0 / (1.0 + (-x).exp()),
            |x| 1.0 / (1.0 + (-x).exp()),
        )?;
        
        // Record operation for autograd - save sigmoid output for backward
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::SigmoidBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![result.clone()], // Save sigmoid(x) for backward
            );
        }
        
        Ok(result)
    }

    /// Tanh activation with automatic differentiation.
    pub fn tanh(&self) -> Result<Self> {
        let result = self.unary_op("tanh", f32::tanh, f64::tanh)?;
        
        // Record operation for autograd - save tanh output for backward
        if self.requires_grad && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::TanhBackward),
                &[self.tensor_id],
                result.tensor_id,
                vec![result.clone()], // Save tanh(x) for backward
            );
        }
        
        Ok(result)
    }

    // ========================================================================
    // MATRIX OPERATIONS
    // ========================================================================

    /// Matrix multiplication with automatic differentiation.
    ///
    /// Supports:
    /// - 2D @ 2D: standard matrix multiplication
    /// - Batched matmul when both tensors have >2 dimensions
    pub fn matmul(&self, other: &Tensor) -> Result<Self> {
        if self.device != other.device {
            return Err(FerrumError::DeviceMismatch {
                operation: "matmul",
                device1: self.device,
                device2: other.device,
            });
        }
        if self.dtype != other.dtype {
            return Err(FerrumError::DTypeMismatch {
                operation: "matmul",
                expected: self.dtype,
                actual: other.dtype,
            });
        }

        let a_shape = self.shape();
        let b_shape = other.shape();

        // Handle different dimension cases
        let result = match (a_shape.len(), b_shape.len()) {
            (2, 2) => self.matmul_2d(other)?,
            (2, 1) => {
                // Matrix @ Vector
                let b_2d = other.unsqueeze(-1)?;
                let result = self.matmul_2d(&b_2d)?;
                result.squeeze(Some(-1))?
            }
            (1, 2) => {
                // Vector @ Matrix
                let a_2d = self.unsqueeze(0)?;
                let result = a_2d.matmul_2d(other)?;
                result.squeeze(Some(0))?
            }
            _ => return Err(FerrumError::not_implemented(format!(
                "matmul for shapes {:?} and {:?}",
                a_shape, b_shape
            ))),
        };
        
        // Record operation for autograd
        if (self.requires_grad || other.requires_grad) && autograd_ops::is_autograd_enabled() {
            autograd_ops::record_operation(
                Box::new(autograd_ops::MatMulBackward),
                &[self.tensor_id, other.tensor_id],
                result.tensor_id,
                vec![self.clone(), other.clone()],
            );
        }
        
        Ok(result)
    }

    /// 2D matrix multiplication.
    fn matmul_2d(&self, other: &Tensor) -> Result<Self> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(FerrumError::shape_mismatch(
                "matmul_2d",
                "2D tensors",
                format!("{:?} and {:?}", a_shape, b_shape),
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if k != b_shape[0] {
            return Err(FerrumError::shape_mismatch(
                "matmul",
                format!("[{}, K] @ [K, {}]", m, n),
                format!("[{}, {}] @ [{}, {}]", m, k, b_shape[0], n),
            ));
        }

        let mut result = Self::zeros([m, n], self.dtype, self.device);

        let a = self.contiguous()?;
        let b = other.contiguous()?;

        match self.dtype {
            DType::F32 => {
                let a_data = a.storage.read_as::<f32>();
                let b_data = b.storage.read_as::<f32>();
                let mut c_data = result.storage.write_as::<f32>();

                // Simple triple loop (will be replaced with optimized kernel)
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for p in 0..k {
                            sum += a_data[i * k + p] * b_data[p * n + j];
                        }
                        c_data[i * n + j] = sum;
                    }
                }
            }
            DType::F64 => {
                let a_data = a.storage.read_as::<f64>();
                let b_data = b.storage.read_as::<f64>();
                let mut c_data = result.storage.write_as::<f64>();

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f64;
                        for p in 0..k {
                            sum += a_data[i * k + p] * b_data[p * n + j];
                        }
                        c_data[i * n + j] = sum;
                    }
                }
            }
            _ => {
                return Err(FerrumError::UnsupportedDType {
                    operation: "matmul",
                    dtype: self.dtype,
                });
            }
        }

        result.requires_grad = self.requires_grad || other.requires_grad;
        Ok(result)
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Broadcast binary operation helper.
fn broadcast_binary_op<T: Copy, F: Fn(T, T) -> T>(
    a: &[T],
    b: &[T],
    result: &mut [T],
    shape: &[usize],
    a_strides: &[usize],
    b_strides: &[usize],
    op: &F,
) {
    let ndim = shape.len();

    let mut coords = vec![0usize; ndim];

    for result_elem in result.iter_mut() {
        // Calculate indices
        let mut a_idx = 0;
        let mut b_idx = 0;
        for d in 0..ndim {
            a_idx += coords[d] * a_strides[d];
            b_idx += coords[d] * b_strides[d];
        }

        *result_elem = op(a[a_idx], b[b_idx]);

        // Increment coordinates
        for d in (0..ndim).rev() {
            coords[d] += 1;
            if coords[d] < shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
}

// ============================================================================
// DISPLAY IMPLEMENTATIONS
// ============================================================================

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .field("contiguous", &self.is_contiguous())
            .finish()
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(")?;

        // Print shape and dtype
        write!(f, "shape={}, dtype={}", self.shape, self.dtype)?;

        // Print first few values if small
        if self.numel() <= 10 {
            write!(f, ", data=[")?;
            let contiguous = self.contiguous().map_err(|_| fmt::Error)?;
            match self.dtype {
                DType::F32 => {
                    let data = contiguous.storage.read_as::<f32>();
                    for (i, &v) in data.as_slice().iter().take(10).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", v)?;
                    }
                }
                DType::F64 => {
                    let data = contiguous.storage.read_as::<f64>();
                    for (i, &v) in data.as_slice().iter().take(10).enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", v)?;
                    }
                }
                _ => {
                    write!(f, "...")?;
                }
            }
            write!(f, "]")?;
        }

        write!(f, ")")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = Tensor::zeros([2, 3], DType::F32, Device::Cpu);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_ones() {
        let t = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let data = t.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, [2, 3], Device::Cpu).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec::<f32>().unwrap(), data);
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::arange(0.0, 12.0, 1.0, DType::F32, Device::Cpu);
        let reshaped = t.reshape([3, 4]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 4]);
        assert_eq!(reshaped.storage_ref_count(), 2); // Shared storage
    }

    #[test]
    fn test_transpose() {
        let t =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], Device::Cpu).unwrap();
        let transposed = t.t().unwrap();
        assert_eq!(transposed.shape(), &[3, 2]);

        // After transpose, data should be reordered when made contiguous
        let contiguous = transposed.contiguous().unwrap();
        let data = contiguous.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_add() {
        let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let b = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let c = a.add(&b).unwrap();
        let data = c.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let b = Tensor::full([3], 2.0, DType::F32, Device::Cpu);
        let c = a.add(&b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        let data = c.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 3.0));
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);
        let b = Tensor::ones([3, 4], DType::F32, Device::Cpu);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 4]);
        let data = c.to_vec::<f32>().unwrap();
        assert!(data.iter().all(|&x| x == 3.0)); // Each element is sum of 3 ones
    }

    #[test]
    fn test_sum_mean() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4], Device::Cpu).unwrap();

        let sum = t.sum().unwrap();
        assert_eq!(sum.item().unwrap(), 10.0);

        let mean = t.mean().unwrap();
        assert_eq!(mean.item().unwrap(), 2.5);
    }

    #[test]
    fn test_requires_grad() {
        let mut t = Tensor::randn([2, 3], DType::F32, Device::Cpu);
        assert!(!t.requires_grad());

        t.set_requires_grad(true);
        assert!(t.requires_grad());
        assert!(t.grad().is_none());
    }

    #[test]
    fn test_activations() {
        let t = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0], [4], Device::Cpu).unwrap();

        let relu = t.relu().unwrap();
        let relu_data = relu.to_vec::<f32>().unwrap();
        assert_eq!(relu_data, vec![0.0, 0.0, 1.0, 2.0]);

        let sigmoid = Tensor::zeros([1], DType::F32, Device::Cpu)
            .sigmoid()
            .unwrap();
        let sigmoid_val = sigmoid.item().unwrap();
        assert!((sigmoid_val - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let t = Tensor::randn([2, 3], DType::F32, Device::Cpu);

        let unsqueezed = t.unsqueeze(0).unwrap();
        assert_eq!(unsqueezed.shape(), &[1, 2, 3]);

        let squeezed = unsqueezed.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed.shape(), &[2, 3]);
    }

    #[test]
    fn test_sum_dim() {
        // Test sum along dimension 0
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], Device::Cpu).unwrap();
        
        let sum0 = t.sum_dim(0, false).unwrap();
        assert_eq!(sum0.shape(), &[3]);
        let data = sum0.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![5.0, 7.0, 9.0]); // [1+4, 2+5, 3+6]
        
        let sum0_keepdim = t.sum_dim(0, true).unwrap();
        assert_eq!(sum0_keepdim.shape(), &[1, 3]);
        
        // Test sum along dimension 1
        let sum1 = t.sum_dim(1, false).unwrap();
        assert_eq!(sum1.shape(), &[2]);
        let data = sum1.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![6.0, 15.0]); // [1+2+3, 4+5+6]
    }

    #[test]
    fn test_mean_dim() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], Device::Cpu).unwrap();
        
        let mean1 = t.mean_dim(1, false).unwrap();
        assert_eq!(mean1.shape(), &[2]);
        let data = mean1.to_vec::<f32>().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6); // mean(1,2,3) = 2
        assert!((data[1] - 5.0).abs() < 1e-6); // mean(4,5,6) = 5
    }

    #[test]
    fn test_argmax_argmin() {
        let t = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 6.0, 4.0, 5.0], [2, 3], Device::Cpu).unwrap();
        
        // argmax along dim 1 (within each row)
        let argmax1 = t.argmax(1, false).unwrap();
        assert_eq!(argmax1.shape(), &[2]);
        let data = argmax1.to_vec::<i64>().unwrap();
        assert_eq!(data, vec![1, 0]); // max at indices 1 (3.0) and 0 (6.0)
        
        // argmin along dim 1
        let argmin1 = t.argmin(1, false).unwrap();
        assert_eq!(argmin1.shape(), &[2]);
        let data = argmin1.to_vec::<i64>().unwrap();
        assert_eq!(data, vec![0, 1]); // min at indices 0 (1.0) and 1 (4.0)
    }

    #[test]
    fn test_max_min_dim() {
        let t = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 6.0, 4.0, 5.0], [2, 3], Device::Cpu).unwrap();
        
        let (max_vals, max_indices) = t.max_dim(1, false).unwrap();
        assert_eq!(max_vals.shape(), &[2]);
        assert_eq!(max_indices.shape(), &[2]);
        
        let vals = max_vals.to_vec::<f32>().unwrap();
        assert_eq!(vals, vec![3.0, 6.0]);
        
        let indices = max_indices.to_vec::<i64>().unwrap();
        assert_eq!(indices, vec![1, 0]);
    }

    #[test]
    fn test_cat() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], [3], Device::Cpu).unwrap();
        
        let c = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[6]);
        let data = c.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat_2d() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2], Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[5.0f32, 6.0, 7.0, 8.0], [2, 2], Device::Cpu).unwrap();
        
        // Cat along dim 0
        let c0 = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(c0.shape(), &[4, 2]);
        let data0 = c0.to_vec::<f32>().unwrap();
        assert_eq!(data0, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        
        // Cat along dim 1
        let c1 = Tensor::cat(&[&a, &b], 1).unwrap();
        assert_eq!(c1.shape(), &[2, 4]);
        let data1 = c1.to_vec::<f32>().unwrap();
        assert_eq!(data1, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    #[test]
    fn test_stack() {
        let a = Tensor::from_slice(&[1.0f32, 2.0], [2], Device::Cpu).unwrap();
        let b = Tensor::from_slice(&[3.0f32, 4.0], [2], Device::Cpu).unwrap();
        
        let c = Tensor::stack(&[&a, &b], 0).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data = c.to_vec::<f32>().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_narrow() {
        let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], Device::Cpu).unwrap();
        
        // Narrow along dim 1
        let n = t.narrow(1, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2, 2]);
        let data = n.contiguous().unwrap().to_vec::<f32>().unwrap();
        assert_eq!(data, vec![2.0, 3.0, 5.0, 6.0]);
    }
}
