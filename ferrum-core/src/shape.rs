//! Shape and stride management for N-dimensional tensors.
//!
//! This module provides efficient representation and manipulation of tensor dimensions:
//!
//! - [`Shape`] - Immutable dimension specification
//! - Stride calculation for row-major (C-style) layout
//! - Broadcasting rules compatible with NumPy/PyTorch
//!
//! ## Memory Layout
//!
//! FERRUM uses row-major (C-style) ordering by default:
//!
//! ```text
//! Shape: [2, 3]
//! Strides: [3, 1]  (last dimension is contiguous)
//!
//! Logical:        Physical memory:
//! [[0, 1, 2],     [0, 1, 2, 3, 4, 5]
//!  [3, 4, 5]]
//! ```
//!
//! ## Broadcasting
//!
//! Two shapes are broadcast-compatible if, for each dimension (aligned from the right):
//! 1. They are equal, OR
//! 2. One of them is 1
//!
//! ```rust
//! use ferrum_core::Shape;
//!
//! let a = Shape::from([2, 3, 4]);
//! let b = Shape::from([4]);
//! let result = Shape::broadcast(&a, &b).unwrap();
//! assert_eq!(result.dims(), &[2, 3, 4]);
//! ```

use smallvec::SmallVec;
use std::fmt;
use std::ops::Index;

use crate::error::{FerrumError, Result};

/// Maximum dimensions for stack allocation (covers most practical cases).
const MAX_INLINE_DIMS: usize = 6;

/// Shape of an N-dimensional tensor.
///
/// Internally uses SmallVec for efficient handling of common cases (≤6 dims)
/// without heap allocation.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: SmallVec<[usize; MAX_INLINE_DIMS]>,
}

impl Shape {
    /// Create a scalar shape (0 dimensions).
    #[inline]
    pub fn scalar() -> Self {
        Shape {
            dims: SmallVec::new(),
        }
    }

    /// Create a shape from a slice of dimensions.
    pub fn from_slice(dims: &[usize]) -> Self {
        Shape {
            dims: SmallVec::from_slice(dims),
        }
    }

    /// Get the dimensions as a slice.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the number of dimensions (rank).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Check if this is a scalar (0 dimensions).
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get the total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get a specific dimension size.
    ///
    /// Supports negative indexing (-1 = last dimension).
    pub fn dim(&self, index: i64) -> Result<usize> {
        let ndim = self.ndim();
        let actual_idx = if index < 0 {
            if (-index as usize) > ndim {
                return Err(FerrumError::InvalidAxis { axis: index, ndim });
            }
            ndim - (-index as usize)
        } else {
            let idx = index as usize;
            if idx >= ndim {
                return Err(FerrumError::InvalidAxis { axis: index, ndim });
            }
            idx
        };
        Ok(self.dims[actual_idx])
    }

    /// Calculate strides for contiguous row-major layout.
    pub fn strides_contiguous(&self) -> SmallVec<[usize; MAX_INLINE_DIMS]> {
        let ndim = self.ndim();
        if ndim == 0 {
            return SmallVec::new();
        }

        let mut strides = SmallVec::with_capacity(ndim);
        strides.resize(ndim, 0);

        let mut stride = 1usize;
        for i in (0..ndim).rev() {
            strides[i] = stride;
            stride *= self.dims[i];
        }

        strides
    }

    /// Check if given strides represent contiguous storage.
    pub fn is_contiguous(&self, strides: &[usize]) -> bool {
        if self.ndim() != strides.len() {
            return false;
        }

        let expected = self.strides_contiguous();

        // Handle dimensions of size 1 (stride doesn't matter)
        for i in 0..self.ndim() {
            if self.dims[i] != 1 && strides[i] != expected[i] {
                return false;
            }
        }
        true
    }

    /// Compute broadcast shape between two shapes.
    ///
    /// Returns `None` if shapes are not broadcast-compatible.
    pub fn broadcast(a: &Shape, b: &Shape) -> Result<Shape> {
        let a_dims = a.dims();
        let b_dims = b.dims();
        let max_ndim = a_dims.len().max(b_dims.len());

        let mut result = SmallVec::with_capacity(max_ndim);

        for i in 0..max_ndim {
            let a_dim = if i < a_dims.len() {
                a_dims[a_dims.len() - 1 - i]
            } else {
                1
            };
            let b_dim = if i < b_dims.len() {
                b_dims[b_dims.len() - 1 - i]
            } else {
                1
            };

            if a_dim == b_dim {
                result.push(a_dim);
            } else if a_dim == 1 {
                result.push(b_dim);
            } else if b_dim == 1 {
                result.push(a_dim);
            } else {
                return Err(FerrumError::BroadcastError {
                    lhs: Box::new(a.clone()),
                    rhs: Box::new(b.clone()),
                });
            }
        }

        result.reverse();
        Ok(Shape { dims: result })
    }

    /// Calculate broadcast strides for this shape to match a target shape.
    ///
    /// Returns strides where broadcast dimensions have stride 0.
    pub fn broadcast_strides(&self, target: &Shape) -> Result<SmallVec<[usize; MAX_INLINE_DIMS]>> {
        let src_dims = self.dims();
        let tgt_dims = target.dims();

        if src_dims.len() > tgt_dims.len() {
            return Err(FerrumError::BroadcastError {
                lhs: Box::new(self.clone()),
                rhs: Box::new(target.clone()),
            });
        }

        let src_strides = self.strides_contiguous();
        let mut result = SmallVec::with_capacity(tgt_dims.len());

        // Pad with zeros for extra dimensions
        for _ in 0..(tgt_dims.len() - src_dims.len()) {
            result.push(0);
        }

        // Calculate strides for aligned dimensions
        for i in 0..src_dims.len() {
            let src_dim = src_dims[i];
            let tgt_dim = tgt_dims[tgt_dims.len() - src_dims.len() + i];

            if src_dim == tgt_dim {
                result.push(src_strides[i]);
            } else if src_dim == 1 {
                result.push(0); // Broadcast dimension
            } else {
                return Err(FerrumError::BroadcastError {
                    lhs: Box::new(self.clone()),
                    rhs: Box::new(target.clone()),
                });
            }
        }

        Ok(result)
    }

    /// Validate that this shape can be reshaped to target shape.
    pub fn can_reshape_to(&self, target: &Shape) -> bool {
        self.numel() == target.numel()
    }

    /// Transpose dimensions according to permutation.
    pub fn transpose(&self, perm: &[usize]) -> Result<Shape> {
        if perm.len() != self.ndim() {
            return Err(FerrumError::InvalidShape {
                message: format!(
                    "Permutation length {} doesn't match ndim {}",
                    perm.len(),
                    self.ndim()
                ),
            });
        }

        // Validate permutation
        let mut seen = vec![false; perm.len()];
        for &p in perm {
            if p >= perm.len() || seen[p] {
                return Err(FerrumError::InvalidShape {
                    message: format!("Invalid permutation: {:?}", perm),
                });
            }
            seen[p] = true;
        }

        let new_dims: SmallVec<[usize; MAX_INLINE_DIMS]> =
            perm.iter().map(|&i| self.dims[i]).collect();

        Ok(Shape { dims: new_dims })
    }

    /// Insert a dimension of size 1 at the specified position.
    pub fn unsqueeze(&self, dim: i64) -> Result<Shape> {
        let ndim = self.ndim();
        let actual_dim = if dim < 0 {
            let d = ndim as i64 + dim + 1;
            if d < 0 {
                return Err(FerrumError::InvalidAxis { axis: dim, ndim });
            }
            d as usize
        } else {
            let d = dim as usize;
            if d > ndim {
                return Err(FerrumError::InvalidAxis { axis: dim, ndim });
            }
            d
        };

        let mut new_dims = self.dims.clone();
        new_dims.insert(actual_dim, 1);
        Ok(Shape { dims: new_dims })
    }

    /// Remove a dimension of size 1 at the specified position.
    pub fn squeeze(&self, dim: Option<i64>) -> Result<Shape> {
        match dim {
            None => {
                // Remove all size-1 dimensions
                let new_dims: SmallVec<[usize; MAX_INLINE_DIMS]> =
                    self.dims.iter().copied().filter(|&d| d != 1).collect();
                Ok(Shape { dims: new_dims })
            }
            Some(d) => {
                let actual_dim = self.normalize_dim(d)?;
                if self.dims[actual_dim] != 1 {
                    return Err(FerrumError::InvalidShape {
                        message: format!(
                            "Cannot squeeze dimension {} with size {}",
                            actual_dim, self.dims[actual_dim]
                        ),
                    });
                }
                let mut new_dims = self.dims.clone();
                new_dims.remove(actual_dim);
                Ok(Shape { dims: new_dims })
            }
        }
    }

    /// Normalize a potentially negative dimension index.
    pub fn normalize_dim(&self, dim: i64) -> Result<usize> {
        let ndim = self.ndim();
        if dim < 0 {
            let d = ndim as i64 + dim;
            if d < 0 {
                return Err(FerrumError::InvalidAxis { axis: dim, ndim });
            }
            Ok(d as usize)
        } else {
            let d = dim as usize;
            if d >= ndim {
                return Err(FerrumError::InvalidAxis { axis: dim, ndim });
            }
            Ok(d)
        }
    }
}

impl Default for Shape {
    fn default() -> Self {
        Shape::scalar()
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims.as_slice())
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        write!(f, "]")
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Shape {
            dims: SmallVec::from_slice(&dims),
        }
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Shape::from_slice(dims)
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Shape {
            dims: SmallVec::from_vec(dims),
        }
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape::scalar()
    }
}

impl From<usize> for Shape {
    fn from(dim: usize) -> Self {
        Shape {
            dims: smallvec::smallvec![dim],
        }
    }
}

impl From<(usize,)> for Shape {
    fn from((d0,): (usize,)) -> Self {
        Shape {
            dims: smallvec::smallvec![d0],
        }
    }
}

impl From<(usize, usize)> for Shape {
    fn from((d0, d1): (usize, usize)) -> Self {
        Shape {
            dims: smallvec::smallvec![d0, d1],
        }
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from((d0, d1, d2): (usize, usize, usize)) -> Self {
        Shape {
            dims: smallvec::smallvec![d0, d1, d2],
        }
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from((d0, d1, d2, d3): (usize, usize, usize, usize)) -> Self {
        Shape {
            dims: smallvec::smallvec![d0, d1, d2, d3],
        }
    }
}

impl std::iter::FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        Shape {
            dims: iter.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.ndim(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_scalar_shape() {
        let s = Shape::scalar();
        assert!(s.is_scalar());
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
    }

    #[test]
    fn test_strides_contiguous() {
        let s = Shape::from([2, 3, 4]);
        let strides = s.strides_contiguous();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn test_broadcast_same() {
        let a = Shape::from([2, 3]);
        let b = Shape::from([2, 3]);
        let result = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_scalar() {
        let a = Shape::from([2, 3]);
        let b = Shape::scalar();
        let result = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 3]);
    }

    #[test]
    fn test_broadcast_extend() {
        let a = Shape::from([2, 3, 4]);
        let b = Shape::from([4]);
        let result = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcast_ones() {
        let a = Shape::from([2, 1, 4]);
        let b = Shape::from([1, 3, 1]);
        let result = Shape::broadcast(&a, &b).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_broadcast_fail() {
        let a = Shape::from([2, 3]);
        let b = Shape::from([2, 4]);
        assert!(Shape::broadcast(&a, &b).is_err());
    }

    #[test]
    fn test_transpose() {
        let s = Shape::from([2, 3, 4]);
        let t = s.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(t.dims(), &[4, 2, 3]);
    }

    #[test]
    fn test_unsqueeze() {
        let s = Shape::from([2, 3]);

        let s0 = s.unsqueeze(0).unwrap();
        assert_eq!(s0.dims(), &[1, 2, 3]);

        let s1 = s.unsqueeze(1).unwrap();
        assert_eq!(s1.dims(), &[2, 1, 3]);

        let s_neg = s.unsqueeze(-1).unwrap();
        assert_eq!(s_neg.dims(), &[2, 3, 1]);
    }

    #[test]
    fn test_squeeze() {
        let s = Shape::from([1, 2, 1, 3, 1]);

        let squeezed = s.squeeze(None).unwrap();
        assert_eq!(squeezed.dims(), &[2, 3]);

        let squeezed_0 = s.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed_0.dims(), &[2, 1, 3, 1]);
    }

    #[test]
    fn test_negative_indexing() {
        let s = Shape::from([2, 3, 4]);
        assert_eq!(s.dim(-1).unwrap(), 4);
        assert_eq!(s.dim(-2).unwrap(), 3);
        assert_eq!(s.dim(-3).unwrap(), 2);
    }
}
