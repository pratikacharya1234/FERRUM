//! Element-wise binary operations.

use ferrum_core::{Result, Tensor};

/// Element-wise addition with broadcasting
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.add(b)
}

/// Element-wise subtraction with broadcasting
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.sub(b)
}

/// Element-wise multiplication with broadcasting
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.mul(b)
}

/// Element-wise division with broadcasting
pub fn div(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.div(b)
}
