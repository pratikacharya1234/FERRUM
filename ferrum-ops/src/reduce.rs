//! Reduction operations (sum, mean, max, min, etc.)

use ferrum_core::{Result, Tensor};

/// Sum all elements
pub fn sum(x: &Tensor) -> Result<Tensor> {
    x.sum()
}

/// Mean of all elements
pub fn mean(x: &Tensor) -> Result<Tensor> {
    x.mean()
}

/// Sum along a specific dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
pub fn sum_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    x.sum_dim(dim, keepdim)
}

/// Mean along a specific dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
pub fn mean_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    x.mean_dim(dim, keepdim)
}

/// Find indices of maximum values along a dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
pub fn argmax(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    x.argmax(dim, keepdim)
}

/// Find indices of minimum values along a dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
pub fn argmin(x: &Tensor, dim: i64, keepdim: bool) -> Result<Tensor> {
    x.argmin(dim, keepdim)
}

/// Find maximum values and their indices along a dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
/// 
/// # Returns
/// Tuple of (max_values, indices)
pub fn max_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<(Tensor, Tensor)> {
    x.max_dim(dim, keepdim)
}

/// Find minimum values and their indices along a dimension
/// 
/// # Arguments
/// * `x` - Input tensor
/// * `dim` - The dimension to reduce
/// * `keepdim` - If true, keeps the reduced dimension with size 1
/// 
/// # Returns
/// Tuple of (min_values, indices)
pub fn min_dim(x: &Tensor, dim: i64, keepdim: bool) -> Result<(Tensor, Tensor)> {
    x.min_dim(dim, keepdim)
}

/// Concatenate tensors along a dimension
/// 
/// All tensors must have the same shape except in the concatenating dimension.
pub fn cat(tensors: &[&Tensor], dim: i64) -> Result<Tensor> {
    Tensor::cat(tensors, dim)
}

/// Stack tensors along a new dimension
/// 
/// All tensors must have exactly the same shape.
pub fn stack(tensors: &[&Tensor], dim: i64) -> Result<Tensor> {
    Tensor::stack(tensors, dim)
}
