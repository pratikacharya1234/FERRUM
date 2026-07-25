//! Mixed Precision Training support.
//!
//! Provides FP16/BF16 training with automatic loss scaling.

use ferrum_core::{Device, DType, Result, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Gradient scaler for mixed precision training.
/// 
/// Scales loss to prevent gradient underflow in FP16,
/// and unscales gradients before optimizer step.
pub struct GradScaler {
    scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    step_count: AtomicUsize,
    enabled: bool,
}

impl GradScaler {
    /// Create a new gradient scaler with default settings.
    pub fn new() -> Self {
        Self {
            scale: 65536.0, // 2^16, reasonable starting scale
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            step_count: AtomicUsize::new(0),
            enabled: true,
        }
    }

    /// Create a scaler with custom initial scale.
    pub fn with_scale(init_scale: f32) -> Self {
        Self {
            scale: init_scale,
            ..Self::new()
        }
    }

    /// Get current scale factor.
    pub fn get_scale(&self) -> f32 {
        self.scale
    }

    /// Check if scaling is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Scale the loss for backward pass.
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        if !self.enabled {
            return Ok(loss.clone());
        }
        loss.mul_scalar(self.scale as f64)
    }

    /// Unscale gradients and check for infs/nans.
    /// Returns true if gradients are valid, false if inf/nan detected.
    pub fn unscale_gradients(&self, gradients: &mut [Tensor]) -> Result<bool> {
        if !self.enabled {
            return Ok(true);
        }

        let inv_scale = 1.0 / self.scale;
        let mut found_inf = false;

        for grad in gradients.iter_mut() {
            let data = grad.to_vec::<f32>()?;
            
            // Check for inf/nan
            for &val in &data {
                if val.is_infinite() || val.is_nan() {
                    found_inf = true;
                    break;
                }
            }

            if found_inf {
                break;
            }

            // Unscale
            let unscaled: Vec<f32> = data.iter().map(|&x| x * inv_scale).collect();
            *grad = Tensor::from_slice(&unscaled, grad.shape(), grad.device())?;
        }

        Ok(!found_inf)
    }

    /// Update the scale factor based on gradient validity.
    pub fn update(&mut self, found_inf: bool) {
        if !self.enabled {
            return;
        }

        if found_inf {
            // Reduce scale on overflow
            self.scale *= self.backoff_factor;
        } else {
            // Potentially increase scale
            let count = self.step_count.fetch_add(1, Ordering::SeqCst);
            if (count + 1) % self.growth_interval == 0 {
                self.scale *= self.growth_factor;
            }
        }
    }

    /// Combined step: unscale, check, and update.
    /// Returns true if optimizer step should proceed.
    pub fn step(&mut self, gradients: &mut [Tensor]) -> Result<bool> {
        let valid = self.unscale_gradients(gradients)?;
        self.update(!valid);
        Ok(valid)
    }
}

impl Default for GradScaler {
    fn default() -> Self {
        Self::new()
    }
}

/// Autocasting context for mixed precision.
/// 
/// Automatically casts inputs to appropriate precision.
pub struct Autocast {
    enabled: bool,
    dtype: DType,
}

impl Autocast {
    /// Create autocast context with FP16.
    pub fn new() -> Self {
        Self {
            enabled: true,
            dtype: DType::F32, // Would be F16 with actual support
        }
    }

    /// Create autocast context with specific dtype.
    pub fn with_dtype(dtype: DType) -> Self {
        Self {
            enabled: true,
            dtype,
        }
    }

    /// Check if autocasting is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Cast tensor to autocast dtype if needed.
    pub fn cast(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.enabled || tensor.dtype() == self.dtype {
            return Ok(tensor.clone());
        }
        tensor.to_dtype(self.dtype)
    }

    /// Cast tensor back to FP32.
    pub fn cast_back(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.dtype() == DType::F32 {
            return Ok(tensor.clone());
        }
        tensor.to_dtype(DType::F32)
    }
}

impl Default for Autocast {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert tensor to half precision (FP16).
///
/// FP16 format: 1 sign bit, 5 exponent bits, 10 mantissa bits
/// Range: ±65504, smallest normal: ~6e-8
pub fn to_half(tensor: &Tensor) -> Result<Tensor> {
    let data = tensor.to_vec::<f32>()?;
    let fp16_data: Vec<u16> = data.iter().map(|&x| f32_to_f16(x)).collect();
    
    // Store as raw bits in a tensor (using F32 but with FP16 values packed)
    // In a real implementation, we'd have an F16 dtype
    // For now, we'll store the FP16 values as f32 for compatibility
    let converted: Vec<f32> = fp16_data.iter().map(|&bits| f16_to_f32(bits)).collect();
    Tensor::from_slice(&converted, tensor.shape().to_vec(), tensor.device())
}

/// Convert tensor to bfloat16.
///
/// BF16 format: 1 sign bit, 8 exponent bits, 7 mantissa bits
/// Same range as FP32 but reduced precision
pub fn to_bfloat16(tensor: &Tensor) -> Result<Tensor> {
    let data = tensor.to_vec::<f32>()?;
    let bf16_data: Vec<u16> = data.iter().map(|&x| f32_to_bf16(x)).collect();
    
    // Store as BF16 values converted back to f32 for compatibility
    let converted: Vec<f32> = bf16_data.iter().map(|&bits| bf16_to_f32(bits)).collect();
    Tensor::from_slice(&converted, tensor.shape().to_vec(), tensor.device())
}

/// Convert tensor to full precision (FP32).
pub fn to_float(tensor: &Tensor) -> Result<Tensor> {
    tensor.to_dtype(DType::F32)
}

/// Convert f32 to f16 (IEEE 754 half-precision).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp = ((bits >> 23) as i32 - 127 + 15) as u16;
    let mantissa = (bits >> 13) as u16 & 0x3FF;
    
    if exp == 0 {
        // Zero or denormalized
        if mantissa == 0 {
            sign
        } else {
            sign | (mantissa >> 1)
        }
    } else if exp == 0x1F {
        // Inf or NaN
        sign | 0x7C00 | (mantissa >> 1)
    } else if exp >= 0x1F {
        // Overflow -> Inf
        sign | 0x7C00
    } else if exp == 1 {
        // Denormalized
        sign | (mantissa >> 1)
    } else {
        sign | (exp << 10) | mantissa
    }
}

/// Convert f16 to f32.
fn f16_to_f32(val: u16) -> f32 {
    let sign = (val >> 15) as u32 & 1;
    let exp = ((val >> 10) & 0x1F) as i32;
    let mantissa = (val & 0x3FF) as u32;
    
    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            // Denormalized
            let mut m = mantissa;
            let mut e = -14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            f32::from_bits((sign << 31) | (((e + 127) as u32) << 23) | (m << 13))
        }
    } else if exp == 31 {
        f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 13))
    } else {
        f32::from_bits((sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mantissa << 13))
    }
}

/// Convert f32 to bf16 (brain floating point).
fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) as u16 & 0x8000;
    let exp = ((bits >> 23) as i32 - 127 + 127) as u16;
    let mantissa = (bits >> 16) as u16 & 0x7F;
    
    if exp == 0 {
        sign | (mantissa >> 1)
    } else if exp >= 0xFF {
        // Inf or NaN
        sign | 0x7F80 | mantissa
    } else {
        sign | (exp << 7) | mantissa
    }
}

/// Convert bf16 to f32.
fn bf16_to_f32(val: u16) -> f32 {
    let sign = (val >> 15) as u32 & 1;
    let exp = ((val >> 7) & 0xFF) as i32;
    let mantissa = (val & 0x7F) as u32;
    
    if exp == 0 {
        if mantissa == 0 {
            f32::from_bits(sign << 31)
        } else {
            f32::from_bits((sign << 31) | (mantissa << 16))
        }
    } else if exp == 0xFF {
        f32::from_bits((sign << 31) | 0x7F800000 | (mantissa << 16))
    } else {
        f32::from_bits((sign << 31) | (((exp - 127 + 127) as u32) << 23) | (mantissa << 16))
    }
}

/// Check if tensor contains inf or nan values.
pub fn has_inf_or_nan(tensor: &Tensor) -> Result<bool> {
    let data = tensor.to_vec::<f32>()?;
    Ok(data.iter().any(|&x| x.is_infinite() || x.is_nan()))
}

/// Clip gradients by global norm.
pub fn clip_grad_norm(parameters: &mut [Tensor], max_norm: f32) -> Result<f32> {
    // Compute total norm
    let mut total_norm_sq = 0.0f32;
    
    for param in parameters.iter() {
        let data = param.to_vec::<f32>()?;
        for &val in &data {
            total_norm_sq += val * val;
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    // Clip if needed
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for param in parameters.iter_mut() {
            let data = param.to_vec::<f32>()?;
            let clipped: Vec<f32> = data.iter().map(|&x| x * clip_coef).collect();
            *param = Tensor::from_slice(&clipped, param.shape(), param.device())?;
        }
    }
    
    Ok(total_norm)
}

/// Clip gradients by value.
pub fn clip_grad_value(parameters: &mut [Tensor], clip_value: f32) -> Result<()> {
    for param in parameters.iter_mut() {
        let data = param.to_vec::<f32>()?;
        let clipped: Vec<f32> = data.iter().map(|&x| x.clamp(-clip_value, clip_value)).collect();
        *param = Tensor::from_slice(&clipped, param.shape(), param.device())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_scaler() {
        let mut scaler = GradScaler::new();
        assert_eq!(scaler.get_scale(), 65536.0);

        // Test scaling loss
        let loss = Tensor::from_slice(&[1.0f32], [1], Device::Cpu).unwrap();
        let scaled = scaler.scale(&loss).unwrap();
        let scaled_val = scaled.to_vec::<f32>().unwrap()[0];
        assert!((scaled_val - 65536.0).abs() < 1e-3);

        // Test update on overflow
        scaler.update(true);
        assert!((scaler.get_scale() - 32768.0).abs() < 1e-3);
    }

    #[test]
    fn test_unscale_gradients() {
        let scaler = GradScaler::with_scale(2.0);
        let mut grads = vec![
            Tensor::from_slice(&[2.0f32, 4.0, 6.0], [3], Device::Cpu).unwrap(),
        ];

        let valid = scaler.unscale_gradients(&mut grads).unwrap();
        assert!(valid);

        let data = grads[0].to_vec::<f32>().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_has_inf_or_nan() {
        let normal = Tensor::from_slice(&[1.0f32, 2.0, 3.0], [3], Device::Cpu).unwrap();
        assert!(!has_inf_or_nan(&normal).unwrap());

        let with_inf = Tensor::from_slice(&[1.0f32, f32::INFINITY, 3.0], [3], Device::Cpu).unwrap();
        assert!(has_inf_or_nan(&with_inf).unwrap());

        let with_nan = Tensor::from_slice(&[1.0f32, f32::NAN, 3.0], [3], Device::Cpu).unwrap();
        assert!(has_inf_or_nan(&with_nan).unwrap());
    }

    #[test]
    fn test_clip_grad_norm() {
        let mut params = vec![
            Tensor::from_slice(&[3.0f32, 4.0], [2], Device::Cpu).unwrap(), // norm = 5
        ];

        let norm = clip_grad_norm(&mut params, 2.5).unwrap();
        assert!((norm - 5.0).abs() < 1e-5);

        let new_data = params[0].to_vec::<f32>().unwrap();
        let new_norm = (new_data[0] * new_data[0] + new_data[1] * new_data[1]).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_value() {
        let mut params = vec![
            Tensor::from_slice(&[-5.0f32, 0.0, 5.0], [3], Device::Cpu).unwrap(),
        ];

        clip_grad_value(&mut params, 2.0).unwrap();

        let data = params[0].to_vec::<f32>().unwrap();
        assert!((data[0] - (-2.0)).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 2.0).abs() < 1e-5);
    }
}
