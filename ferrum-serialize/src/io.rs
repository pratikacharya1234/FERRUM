//! Save and load functions.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use ferrum_core::{Device, FerrumError, Result, Tensor};

use crate::format::{Header, TensorMeta, MAGIC};

/// Save tensors to a file.
pub fn save<P: AsRef<Path>>(tensors: &HashMap<String, Tensor>, path: P) -> Result<()> {
    let file = File::create(path).map_err(|e| FerrumError::SerializationError {
        message: format!("Failed to create file: {}", e),
    })?;
    let mut writer = BufWriter::new(file);

    // Write magic number
    writer
        .write_all(MAGIC)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to write magic: {}", e),
        })?;

    // Prepare metadata
    let mut tensor_metas = HashMap::new();
    let mut current_offset = 0u64;

    for (name, tensor) in tensors {
        let size = (tensor.numel() * tensor.dtype().size_of()) as u64;
        tensor_metas.insert(
            name.clone(),
            TensorMeta {
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype().into(),
                offset: current_offset,
                size,
            },
        );
        current_offset += size;
    }

    let header = Header {
        version: 1,
        num_tensors: tensors.len(),
        data_size: current_offset,
        tensors: tensor_metas,
    };

    // Serialize header
    let header_bytes =
        bincode::serialize(&header).map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to serialize header: {}", e),
        })?;

    // Write header length
    let header_len = header_bytes.len() as u64;
    writer
        .write_all(&header_len.to_le_bytes())
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to write header length: {}", e),
        })?;

    // Write header
    writer
        .write_all(&header_bytes)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to write header: {}", e),
        })?;

    // Write tensor data in order of offset
    let mut sorted_tensors: Vec<_> = tensors.iter().collect();
    sorted_tensors.sort_by_key(|(name, _)| header.tensors.get(*name).unwrap().offset);

    for (_, tensor) in sorted_tensors {
        let contiguous = tensor.contiguous()?;
        let data: Vec<u8> = match tensor.dtype() {
            ferrum_core::DType::F32 => {
                let vec = contiguous.to_vec::<f32>()?;
                bytemuck::cast_slice(&vec).to_vec()
            }
            ferrum_core::DType::F64 => {
                let vec = contiguous.to_vec::<f64>()?;
                bytemuck::cast_slice(&vec).to_vec()
            }
            ferrum_core::DType::I32 => {
                let vec = contiguous.to_vec::<i32>()?;
                bytemuck::cast_slice(&vec).to_vec()
            }
            ferrum_core::DType::I64 => {
                let vec = contiguous.to_vec::<i64>()?;
                bytemuck::cast_slice(&vec).to_vec()
            }
            _ => {
                return Err(FerrumError::SerializationError {
                    message: format!("Unsupported dtype for serialization: {:?}", tensor.dtype()),
                });
            }
        };
        writer
            .write_all(&data)
            .map_err(|e| FerrumError::SerializationError {
                message: format!("Failed to write tensor data: {}", e),
            })?;
    }

    writer
        .flush()
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to flush: {}", e),
        })?;

    Ok(())
}

/// Load tensors from a file.
pub fn load<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Tensor>> {
    let file = File::open(path).map_err(|e| FerrumError::SerializationError {
        message: format!("Failed to open file: {}", e),
    })?;
    let mut reader = BufReader::new(file);

    // Read and verify magic
    let mut magic = [0u8; 8];
    reader
        .read_exact(&mut magic)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to read magic: {}", e),
        })?;
    if &magic != MAGIC {
        return Err(FerrumError::SerializationError {
            message: "Invalid file format (bad magic number)".to_string(),
        });
    }

    // Read header length
    let mut header_len_bytes = [0u8; 8];
    reader
        .read_exact(&mut header_len_bytes)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to read header length: {}", e),
        })?;
    let header_len = u64::from_le_bytes(header_len_bytes) as usize;

    // Read header
    let mut header_bytes = vec![0u8; header_len];
    reader
        .read_exact(&mut header_bytes)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to read header: {}", e),
        })?;

    let header: Header =
        bincode::deserialize(&header_bytes).map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to deserialize header: {}", e),
        })?;

    // Read tensor data
    let mut data = vec![0u8; header.data_size as usize];
    reader
        .read_exact(&mut data)
        .map_err(|e| FerrumError::SerializationError {
            message: format!("Failed to read tensor data: {}", e),
        })?;

    // Reconstruct tensors
    let mut tensors = HashMap::new();
    for (name, meta) in &header.tensors {
        let dtype: ferrum_core::DType = meta.dtype.into();
        let tensor_data = &data[meta.offset as usize..(meta.offset + meta.size) as usize];
        let shape = ferrum_core::Shape::from_slice(&meta.shape);

        let tensor = match dtype {
            ferrum_core::DType::F32 => {
                let slice: &[f32] = bytemuck::cast_slice(tensor_data);
                Tensor::from_slice(slice, shape.clone(), Device::Cpu)?
            }
            ferrum_core::DType::F64 => {
                let slice: &[f64] = bytemuck::cast_slice(tensor_data);
                Tensor::from_slice(slice, shape.clone(), Device::Cpu)?
            }
            ferrum_core::DType::I32 => {
                let slice: &[i32] = bytemuck::cast_slice(tensor_data);
                Tensor::from_slice(slice, shape.clone(), Device::Cpu)?
            }
            ferrum_core::DType::I64 => {
                let slice: &[i64] = bytemuck::cast_slice(tensor_data);
                Tensor::from_slice(slice, shape.clone(), Device::Cpu)?
            }
            _ => {
                return Err(FerrumError::SerializationError {
                    message: format!("Unsupported dtype for deserialization: {:?}", dtype),
                });
            }
        };

        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
}
