//! Transformer architecture components.
//!
//! Implements Multi-Head Attention, Transformer Encoder/Decoder layers.

use ferrum_core::{Device, DType, Result, Tensor};
use crate::module::Module;
use crate::linear::Linear;

/// Multi-Head Attention mechanism.
pub struct MultiHeadAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f32,
    training: bool,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: Linear::new(embed_dim, embed_dim),
            k_proj: Linear::new(embed_dim, embed_dim),
            v_proj: Linear::new(embed_dim, embed_dim),
            out_proj: Linear::new(embed_dim, embed_dim),
            scale,
            training: true,
        }
    }

    /// Compute attention: softmax(Q @ K^T / sqrt(d_k)) @ V
    pub fn forward_attn(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shape = query.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Project Q, K, V
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape for multi-head: [batch, seq, embed] -> [batch, heads, seq, head_dim]
        let q = self.reshape_for_attention(&q, batch, seq_len)?;
        let k = self.reshape_for_attention(&k, batch, seq_len)?;
        let v = self.reshape_for_attention(&v, batch, seq_len)?;

        // Attention scores: Q @ K^T
        let scores = self.compute_attention_scores(&q, &k)?;

        // Apply mask if provided
        let scores = if let Some(m) = mask {
            self.apply_mask(&scores, m)?
        } else {
            scores
        };

        // Softmax
        let attn_weights = self.softmax_last_dim(&scores)?;

        // Apply attention to values
        let attn_output = self.apply_attention(&attn_weights, &v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed]
        let output = self.reshape_from_attention(&attn_output, batch, seq_len)?;

        // Final projection
        self.out_proj.forward(&output)
    }

    fn reshape_for_attention(&self, x: &Tensor, batch: usize, seq_len: usize) -> Result<Tensor> {
        // [batch, seq, embed] -> [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        let data = x.to_vec::<f32>()?;
        let mut reshaped = vec![0.0f32; batch * self.num_heads * seq_len * self.head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                for h in 0..self.num_heads {
                    for d in 0..self.head_dim {
                        let src_idx = b * seq_len * self.embed_dim + s * self.embed_dim + h * self.head_dim + d;
                        let dst_idx = b * self.num_heads * seq_len * self.head_dim 
                            + h * seq_len * self.head_dim + s * self.head_dim + d;
                        reshaped[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_slice(&reshaped, (batch, self.num_heads, seq_len, self.head_dim), x.device())
    }

    fn reshape_from_attention(&self, x: &Tensor, batch: usize, seq_len: usize) -> Result<Tensor> {
        let data = x.to_vec::<f32>()?;
        let mut reshaped = vec![0.0f32; batch * seq_len * self.embed_dim];

        for b in 0..batch {
            for h in 0..self.num_heads {
                for s in 0..seq_len {
                    for d in 0..self.head_dim {
                        let src_idx = b * self.num_heads * seq_len * self.head_dim 
                            + h * seq_len * self.head_dim + s * self.head_dim + d;
                        let dst_idx = b * seq_len * self.embed_dim + s * self.embed_dim + h * self.head_dim + d;
                        reshaped[dst_idx] = data[src_idx];
                    }
                }
            }
        }

        Tensor::from_slice(&reshaped, (batch, seq_len, self.embed_dim), x.device())
    }

    fn compute_attention_scores(&self, q: &Tensor, k: &Tensor) -> Result<Tensor> {
        let q_data = q.to_vec::<f32>()?;
        let k_data = k.to_vec::<f32>()?;
        let shape = q.shape();
        let (batch, heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);

        let mut scores = vec![0.0f32; batch * heads * seq_len * seq_len];

        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        let mut sum = 0.0f32;
                        for d in 0..head_dim {
                            let q_idx = b * heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d;
                            let k_idx = b * heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                            sum += q_data[q_idx] * k_data[k_idx];
                        }
                        let s_idx = b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                        scores[s_idx] = sum * self.scale;
                    }
                }
            }
        }

        Tensor::from_slice(&scores, (batch, heads, seq_len, seq_len), q.device())
    }

    fn apply_mask(&self, scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let s_data = scores.to_vec::<f32>()?;
        let m_data = mask.to_vec::<f32>()?;
        let mut result = s_data.clone();
        let len = result.len();

        for (i, &m) in m_data.iter().enumerate() {
            if m == 0.0 {
                result[i % len] = f32::NEG_INFINITY;
            }
        }

        Tensor::from_slice(&result, scores.shape().to_vec(), scores.device())
    }

    fn softmax_last_dim(&self, x: &Tensor) -> Result<Tensor> {
        let data = x.to_vec::<f32>()?;
        let shape = x.shape();
        let last_dim = shape[shape.len() - 1];
        let outer = data.len() / last_dim;

        let mut result = vec![0.0f32; data.len()];

        for i in 0..outer {
            let start = i * last_dim;
            let end = start + last_dim;
            let slice = &data[start..end];

            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = slice.iter().map(|&v| (v - max_val).exp()).sum();

            for (j, &v) in slice.iter().enumerate() {
                result[start + j] = (v - max_val).exp() / exp_sum;
            }
        }

        Tensor::from_slice(&result, shape.to_vec(), x.device())
    }

    fn apply_attention(&self, weights: &Tensor, v: &Tensor) -> Result<Tensor> {
        let w_data = weights.to_vec::<f32>()?;
        let v_data = v.to_vec::<f32>()?;
        let shape = v.shape();
        let (batch, heads, seq_len, head_dim) = (shape[0], shape[1], shape[2], shape[3]);

        let mut output = vec![0.0f32; batch * heads * seq_len * head_dim];

        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..seq_len {
                            let w_idx = b * heads * seq_len * seq_len + h * seq_len * seq_len + i * seq_len + j;
                            let v_idx = b * heads * seq_len * head_dim + h * seq_len * head_dim + j * head_dim + d;
                            sum += w_data[w_idx] * v_data[v_idx];
                        }
                        let o_idx = b * heads * seq_len * head_dim + h * seq_len * head_dim + i * head_dim + d;
                        output[o_idx] = sum;
                    }
                }
            }
        }

        Tensor::from_slice(&output, (batch, heads, seq_len, head_dim), v.device())
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_attn(input, input, input, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "MultiHeadAttention" }
}

/// Transformer Encoder Layer.
pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    training: bool,
}

impl TransformerEncoderLayer {
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize) -> Self {
        Self {
            self_attn: MultiHeadAttention::new(d_model, nhead),
            linear1: Linear::new(d_model, dim_feedforward),
            linear2: Linear::new(dim_feedforward, d_model),
            norm1: LayerNorm::new(d_model),
            norm2: LayerNorm::new(d_model),
            training: true,
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Self attention with residual
        let attn_out = self.self_attn.forward(input)?;
        let x = input.add(&attn_out)?;
        let x = self.norm1.forward(&x)?;

        // FFN with residual
        let ff = self.linear1.forward(&x)?;
        let ff = ff.relu()?;
        let ff = self.linear2.forward(&ff)?;
        let x = x.add(&ff)?;
        self.norm2.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "TransformerEncoderLayer" }
}

/// Simple Layer Normalization.
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    normalized_shape: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            weight: Tensor::ones([normalized_shape], DType::F32, Device::Cpu).with_requires_grad(true),
            bias: Tensor::zeros([normalized_shape], DType::F32, Device::Cpu).with_requires_grad(true),
            normalized_shape,
            eps: 1e-5,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let data = input.to_vec::<f32>()?;
        let shape = input.shape();
        let last_dim = self.normalized_shape;
        let outer = data.len() / last_dim;

        let weight_data = self.weight.to_vec::<f32>()?;
        let bias_data = self.bias.to_vec::<f32>()?;

        let mut result = vec![0.0f32; data.len()];

        for i in 0..outer {
            let start = i * last_dim;
            let slice = &data[start..start + last_dim];

            let mean: f32 = slice.iter().sum::<f32>() / last_dim as f32;
            let variance: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / last_dim as f32;
            let std = (variance + self.eps).sqrt();

            for j in 0..last_dim {
                result[start + j] = (slice[j] - mean) / std * weight_data[j] + bias_data[j];
            }
        }

        Tensor::from_slice(&result, shape.to_vec(), input.device())
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn name(&self) -> &str { "LayerNorm" }
}

/// Positional Encoding for transformer.
pub struct PositionalEncoding {
    encoding: Tensor,
    max_len: usize,
    d_model: usize,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_len: usize) -> Self {
        let mut pe = vec![0.0f32; max_len * d_model];

        for pos in 0..max_len {
            for i in 0..d_model / 2 {
                let angle = pos as f32 / 10000.0f32.powf(2.0 * i as f32 / d_model as f32);
                pe[pos * d_model + 2 * i] = angle.sin();
                pe[pos * d_model + 2 * i + 1] = angle.cos();
            }
        }

        let encoding = Tensor::from_slice(&pe, (max_len, d_model), Device::Cpu).unwrap();

        Self { encoding, max_len, d_model }
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let seq_len = shape[1];
        
        // Add positional encoding
        let pe_data = self.encoding.to_vec::<f32>()?;
        let input_data = input.to_vec::<f32>()?;
        
        let batch = shape[0];
        let mut result = input_data.clone();

        for b in 0..batch {
            for s in 0..seq_len.min(self.max_len) {
                for d in 0..self.d_model {
                    let idx = b * seq_len * self.d_model + s * self.d_model + d;
                    let pe_idx = s * self.d_model + d;
                    result[idx] += pe_data[pe_idx];
                }
            }
        }

        Tensor::from_slice(&result, shape.to_vec(), input.device())
    }

    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn name(&self) -> &str { "PositionalEncoding" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(64);
        let input = Tensor::randn((2, 64), DType::F32, Device::Cpu);
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 64]);
    }
}
