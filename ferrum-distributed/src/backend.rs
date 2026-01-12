//! Communication backends for distributed training.

use crate::error::{DistributedError, Result};

/// Communication backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA Collective Communications Library (for GPU).
    Nccl,
    /// Facebook's Gloo library (for CPU).
    Gloo,
    /// Message Passing Interface.
    Mpi,
}

impl Backend {
    /// Check if this backend is available.
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Nccl => {
                #[cfg(feature = "nccl")]
                return true;
                #[cfg(not(feature = "nccl"))]
                false
            }
            Backend::Gloo => {
                // Gloo is always available as our default simulated backend
                true
            }
            Backend::Mpi => {
                #[cfg(feature = "mpi")]
                return true;
                #[cfg(not(feature = "mpi"))]
                false
            }
        }
    }

    /// Get the name of the backend.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Nccl => "nccl",
            Backend::Gloo => "gloo",
            Backend::Mpi => "mpi",
        }
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for Backend {
    type Err = DistributedError;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "nccl" => Ok(Backend::Nccl),
            "gloo" => Ok(Backend::Gloo),
            "mpi" => Ok(Backend::Mpi),
            _ => Err(DistributedError::UnsupportedBackend(s.to_string())),
        }
    }
}

/// Backend configuration options.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Master address for rendezvous.
    pub master_addr: String,
    /// Master port for rendezvous.
    pub master_port: u16,
    /// Timeout for operations in seconds.
    pub timeout_seconds: u64,
    /// Number of threads for communication.
    pub num_threads: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            timeout_seconds: 1800, // 30 minutes
            num_threads: 4,
        }
    }
}

impl BackendConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let master_addr = std::env::var("MASTER_ADDR")
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        let master_port = std::env::var("MASTER_PORT")
            .unwrap_or_else(|_| "29500".to_string())
            .parse()
            .unwrap_or(29500);
        let timeout_seconds = std::env::var("DIST_TIMEOUT")
            .unwrap_or_else(|_| "1800".to_string())
            .parse()
            .unwrap_or(1800);

        Self {
            master_addr,
            master_port,
            timeout_seconds,
            num_threads: 4,
        }
    }

    /// Set master address.
    pub fn with_master_addr(mut self, addr: impl Into<String>) -> Self {
        self.master_addr = addr.into();
        self
    }

    /// Set master port.
    pub fn with_master_port(mut self, port: u16) -> Self {
        self.master_port = port;
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, seconds: u64) -> Self {
        self.timeout_seconds = seconds;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_names() {
        assert_eq!(Backend::Nccl.name(), "nccl");
        assert_eq!(Backend::Gloo.name(), "gloo");
        assert_eq!(Backend::Mpi.name(), "mpi");
    }

    #[test]
    fn test_backend_from_str() {
        assert_eq!("nccl".parse::<Backend>().unwrap(), Backend::Nccl);
        assert_eq!("gloo".parse::<Backend>().unwrap(), Backend::Gloo);
        assert_eq!("mpi".parse::<Backend>().unwrap(), Backend::Mpi);
        assert!("invalid".parse::<Backend>().is_err());
    }

    #[test]
    fn test_gloo_available() {
        assert!(Backend::Gloo.is_available());
    }

    #[test]
    fn test_default_config() {
        let config = BackendConfig::default();
        assert_eq!(config.master_addr, "127.0.0.1");
        assert_eq!(config.master_port, 29500);
    }
}
