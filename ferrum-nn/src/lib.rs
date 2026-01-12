//! # FERRUM NN
//!
//! Neural network layers and modules for deep learning.
//!
//! This crate provides PyTorch-style layer API:
//!
//! ```rust,ignore
//! use ferrum_nn::{Linear, ReLU, Sequential};
//!
//! let model = Sequential::new()
//!     .add(Linear::new(784, 256))
//!     .add(ReLU::new())
//!     .add(Linear::new(256, 10));
//!
//! let output = model.forward(&input)?;
//! ```

pub mod activation;
pub mod container;
pub mod conv;
pub mod embedding;
pub mod init;
pub mod linear;
pub mod loss;
pub mod module;
pub mod models;
pub mod norm;
pub mod rnn;
pub mod transformer;

pub mod prelude {
    //! Convenient re-exports.
    pub use crate::activation::{ReLU, Sigmoid, Tanh, Softmax, LogSoftmax, GELU, SiLU, LeakyReLU, ELU};
    pub use crate::activation::{softmax, log_softmax, gelu, silu};
    pub use crate::container::Sequential;
    pub use crate::conv::{Conv1d, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d};
    pub use crate::embedding::Embedding;
    pub use crate::linear::Linear;
    pub use crate::loss::*;
    pub use crate::models::{ResNet, VGG, MLP, BasicBlock};
    pub use crate::module::Module;
    pub use crate::norm::{LayerNorm, BatchNorm1d, Dropout};
    pub use crate::norm::{layer_norm, batch_norm};
    pub use crate::rnn::{RNNCell, LSTMCell, GRUCell, RNN, LSTM, GRU, Nonlinearity};
    pub use crate::transformer::{
        MultiHeadAttention, TransformerEncoderLayer, LayerNorm as TransformerLayerNorm,
        PositionalEncoding,
    };
}

pub use prelude::*;
