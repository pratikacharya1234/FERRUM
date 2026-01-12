//! Embedding layers for discrete token representations.

use ferrum_core::{Device, DType, Result, Tensor, FerrumError};

use crate::module::Module;

/// Lookup table for embedding vectors.
/// 
/// This module stores embeddings of a fixed dictionary and size.
/// It is commonly used to store word embeddings and retrieve them using indices.
/// 
/// # Example
/// ```ignore
/// let embedding = Embedding::new(1000, 128, Device::Cpu);  // 1000 words, 128 dims
/// 
/// // Input: indices tensor of shape [batch, seq_len]
/// let indices = Tensor::from_slice(&[1i64, 5, 3], [1, 3], Device::Cpu)?;
/// let embeddings = embedding.forward(&indices)?;  // Shape: [1, 3, 128]
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Weight matrix of shape [num_embeddings, embedding_dim]
    pub weight: Tensor,
    /// Number of embeddings
    pub num_embeddings: usize,
    /// Size of each embedding vector
    pub embedding_dim: usize,
    /// Padding index (embeddings at this index are not updated during training)
    pub padding_idx: Option<usize>,
}

impl Embedding {
    /// Create a new embedding table with random initialization.
    /// 
    /// # Arguments
    /// * `num_embeddings` - Size of the dictionary (vocabulary)
    /// * `embedding_dim` - Size of each embedding vector
    /// * `device` - Device to place the embedding table on
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: Device) -> Self {
        // Initialize with normal distribution (like PyTorch)
        let mut weight = Tensor::normal(
            [num_embeddings, embedding_dim],
            0.0,
            1.0,
            DType::F32,
            device,
        );
        weight.set_requires_grad(true);
        
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }
    
    /// Create embedding with specified data type.
    pub fn new_with_dtype(
        num_embeddings: usize,
        embedding_dim: usize,
        dtype: DType,
        device: Device,
    ) -> Self {
        let mut weight = Tensor::normal(
            [num_embeddings, embedding_dim],
            0.0,
            1.0,
            dtype,
            device,
        );
        weight.set_requires_grad(true);
        
        Self {
            weight,
            num_embeddings,
            embedding_dim,
            padding_idx: None,
        }
    }
    
    /// Set padding index (embeddings at this index output zeros and don't update).
    pub fn with_padding_idx(mut self, padding_idx: usize) -> Self {
        self.padding_idx = Some(padding_idx);
        
        // Zero out the padding embedding
        // In a full implementation, we'd modify the weight directly
        // For now, we'll handle this in forward pass
        
        self
    }
    
    /// Create embedding from an existing weight tensor.
    pub fn from_pretrained(weight: Tensor, freeze: bool) -> Result<Self> {
        let shape = weight.shape();
        if shape.len() != 2 {
            return Err(FerrumError::InvalidShape {
                message: format!("Embedding weight must be 2D, got {:?}", shape),
            });
        }
        
        let num_embeddings = shape[0];
        let embedding_dim = shape[1];
        let mut weight = weight;
        weight.set_requires_grad(!freeze);
        
        Ok(Self {
            num_embeddings,
            embedding_dim,
            weight,
            padding_idx: None,
        })
    }
    
    /// Forward pass: look up embeddings for given indices.
    /// 
    /// # Arguments
    /// * `indices` - Tensor of indices (any shape), dtype must be I64
    /// 
    /// # Returns
    /// Tensor of shape [*indices_shape, embedding_dim]
    fn forward_indices(&self, indices: &Tensor) -> Result<Tensor> {
        if indices.dtype() != DType::I64 {
            return Err(FerrumError::DTypeMismatch {
                operation: "Embedding.forward",
                expected: DType::I64,
                actual: indices.dtype(),
            });
        }
        
        let indices_contiguous = indices.contiguous()?;
        let indices_data = indices_contiguous.to_vec::<i64>()?;
        let indices_shape = indices.shape();
        
        // Output shape: indices_shape + [embedding_dim]
        let mut out_shape: Vec<usize> = indices_shape.to_vec();
        out_shape.push(self.embedding_dim);
        
        let numel = indices_data.len();
        let out_numel = numel * self.embedding_dim;
        
        // Get embedding weight data
        let weight_contiguous = self.weight.contiguous()?;
        
        match self.weight.dtype() {
            DType::F32 => {
                let weight_data = weight_contiguous.to_vec::<f32>()?;
                let mut output_data = vec![0.0f32; out_numel];
                
                for (i, &idx) in indices_data.iter().enumerate() {
                    let idx = idx as usize;
                    if idx >= self.num_embeddings {
                        return Err(FerrumError::InvalidShape {
                            message: format!(
                                "Index {} out of bounds for embedding with {} entries",
                                idx, self.num_embeddings
                            ),
                        });
                    }
                    
                    // Handle padding index
                    if self.padding_idx == Some(idx) {
                        // Leave as zeros
                        continue;
                    }
                    
                    // Copy embedding vector
                    let src_start = idx * self.embedding_dim;
                    let dst_start = i * self.embedding_dim;
                    output_data[dst_start..dst_start + self.embedding_dim]
                        .copy_from_slice(&weight_data[src_start..src_start + self.embedding_dim]);
                }
                
                Tensor::from_slice(&output_data, out_shape, self.weight.device())
            }
            DType::F64 => {
                let weight_data = weight_contiguous.to_vec::<f64>()?;
                let mut output_data = vec![0.0f64; out_numel];
                
                for (i, &idx) in indices_data.iter().enumerate() {
                    let idx = idx as usize;
                    if idx >= self.num_embeddings {
                        return Err(FerrumError::InvalidShape {
                            message: format!(
                                "Index {} out of bounds for embedding with {} entries",
                                idx, self.num_embeddings
                            ),
                        });
                    }
                    
                    if self.padding_idx == Some(idx) {
                        continue;
                    }
                    
                    let src_start = idx * self.embedding_dim;
                    let dst_start = i * self.embedding_dim;
                    output_data[dst_start..dst_start + self.embedding_dim]
                        .copy_from_slice(&weight_data[src_start..src_start + self.embedding_dim]);
                }
                
                Tensor::from_slice(&output_data, out_shape, self.weight.device())
            }
            dtype => Err(FerrumError::UnsupportedDType {
                operation: "Embedding.forward",
                dtype,
            }),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_indices(input)
    }
    
    fn name(&self) -> &str {
        "Embedding"
    }
    
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
    
    fn num_parameters(&self) -> usize {
        self.num_embeddings * self.embedding_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::Shape;
    
    #[test]
    fn test_embedding_creation() {
        let embed = Embedding::new(100, 64, Device::Cpu);
        assert_eq!(embed.num_embeddings, 100);
        assert_eq!(embed.embedding_dim, 64);
        assert_eq!(embed.weight.shape(), &[100, 64]);
    }
    
    #[test]
    fn test_embedding_forward() {
        let embed = Embedding::new(10, 4, Device::Cpu);
        
        // Lookup single index
        let indices = Tensor::from_slice(&[3i64], Shape::from([1]), Device::Cpu).unwrap();
        let output = embed.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[1, 4]);
    }
    
    #[test]
    fn test_embedding_batch() {
        let embed = Embedding::new(100, 32, Device::Cpu);
        
        // Batch of sequences
        let indices = Tensor::from_slice(
            &[1i64, 5, 3, 2, 8, 9],
            Shape::from([2, 3]),
            Device::Cpu
        ).unwrap();
        let output = embed.forward(&indices).unwrap();
        assert_eq!(output.shape(), &[2, 3, 32]);
    }
    
    #[test]
    fn test_embedding_from_pretrained() {
        let weight = Tensor::randn([50, 16], DType::F32, Device::Cpu);
        let embed = Embedding::from_pretrained(weight, false).unwrap();
        
        assert_eq!(embed.num_embeddings, 50);
        assert_eq!(embed.embedding_dim, 16);
    }
}
