//! Element-wise unary operations.

use ferrum_core::{Result, Tensor};

/// Apply ReLU activation: max(0, x)
pub fn relu(x: &Tensor) -> Result<Tensor> {
    x.relu()
}

/// Apply sigmoid activation: 1 / (1 + exp(-x))
pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    x.sigmoid()
}

/// Apply tanh activation
pub fn tanh(x: &Tensor) -> Result<Tensor> {
    x.tanh()
}

/// Apply exponential
pub fn exp(x: &Tensor) -> Result<Tensor> {
    x.exp()
}

/// Apply natural logarithm
pub fn log(x: &Tensor) -> Result<Tensor> {
    x.log()
}

/// Apply square root
pub fn sqrt(x: &Tensor) -> Result<Tensor> {
    x.sqrt()
}

/// Apply negation
pub fn neg(x: &Tensor) -> Result<Tensor> {
    x.neg()
}
