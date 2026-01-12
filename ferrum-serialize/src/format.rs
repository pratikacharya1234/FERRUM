//! File format definitions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use ferrum_core::DType;

/// Magic number for FERRUM model files.
pub const MAGIC: &[u8; 8] = b"FERRUM01";

/// File header.
#[derive(Debug, Serialize, Deserialize)]
pub struct Header {
    /// Format version.
    pub version: u32,
    /// Number of tensors.
    pub num_tensors: usize,
    /// Total data size in bytes.
    pub data_size: u64,
    /// Tensor metadata.
    pub tensors: HashMap<String, TensorMeta>,
}

/// Metadata for a single tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMeta {
    /// Shape dimensions.
    pub shape: Vec<usize>,
    /// Data type.
    pub dtype: DTypeInfo,
    /// Byte offset in data section.
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

/// Serializable dtype info.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DTypeInfo {
    F32,
    F64,
    F16,
    BF16,
    I32,
    I64,
    U8,
    Bool,
}

impl From<DType> for DTypeInfo {
    fn from(dtype: DType) -> Self {
        match dtype {
            DType::F32 => DTypeInfo::F32,
            DType::F64 => DTypeInfo::F64,
            DType::F16 => DTypeInfo::F16,
            DType::BF16 => DTypeInfo::BF16,
            DType::I32 => DTypeInfo::I32,
            DType::I64 => DTypeInfo::I64,
            DType::U8 => DTypeInfo::U8,
            DType::Bool => DTypeInfo::Bool,
        }
    }
}

impl From<DTypeInfo> for DType {
    fn from(info: DTypeInfo) -> Self {
        match info {
            DTypeInfo::F32 => DType::F32,
            DTypeInfo::F64 => DType::F64,
            DTypeInfo::F16 => DType::F16,
            DTypeInfo::BF16 => DType::BF16,
            DTypeInfo::I32 => DType::I32,
            DTypeInfo::I64 => DType::I64,
            DTypeInfo::U8 => DType::U8,
            DTypeInfo::Bool => DType::Bool,
        }
    }
}
