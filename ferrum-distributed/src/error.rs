//! Error types for distributed operations.

use thiserror::Error;

/// Distributed training errors.
#[derive(Error, Debug)]
pub enum DistributedError {
    /// Process group already initialized.
    #[error("Process group already initialized")]
    AlreadyInitialized,

    /// Process group not initialized.
    #[error("Process group not initialized")]
    NotInitialized,

    /// Invalid rank.
    #[error("Invalid rank {0} for world size {1}")]
    InvalidRank(usize, usize),

    /// Communication error.
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// Timeout during collective operation.
    #[error("Timeout during {0}")]
    Timeout(String),

    /// Backend not supported.
    #[error("Backend not supported: {0}")]
    UnsupportedBackend(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Tensor error.
    #[error("Tensor error: {0}")]
    TensorError(String),

    /// Internal error.
    #[error("Internal error: {0}")]
    InternalError(String),
}

/// Result type for distributed operations.
pub type Result<T> = std::result::Result<T, DistributedError>;

impl From<ferrum_core::error::FerrumError> for DistributedError {
    fn from(err: ferrum_core::error::FerrumError) -> Self {
        DistributedError::TensorError(err.to_string())
    }
}
