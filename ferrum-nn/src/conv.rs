//! Convolutional neural network layers.
//!
//! Implements Conv1d, Conv2d, and pooling operations.

use ferrum_core::{Device, DType, Result, Tensor, FerrumError};
use crate::module::Module;

/// 1D Convolution layer.
pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    training: bool,
}

impl Conv1d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_config(in_channels, out_channels, kernel_size, 1, 0, 1, 1, true)
    }

    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Self {
        let k = 1.0 / ((in_channels * kernel_size) as f64).sqrt();
        let weight = Tensor::uniform(
            [out_channels, in_channels / groups, kernel_size],
            -k, k, DType::F32, Device::Cpu,
        ).with_requires_grad(true);

        let bias_tensor = if bias {
            Some(Tensor::uniform([out_channels], -k, k, DType::F32, Device::Cpu)
                .with_requires_grad(true))
        } else {
            None
        };

        Self { weight, bias: bias_tensor, stride, padding, dilation, groups, training: true }
    }

    pub fn stride(mut self, stride: usize) -> Self { self.stride = stride; self }
    pub fn padding(mut self, padding: usize) -> Self { self.padding = padding; self }
}

impl Module for Conv1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        conv1d_forward(input, &self.weight, self.bias.as_ref(), 
                       self.stride, self.padding, self.dilation, self.groups)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias { params.push(bias.clone()); }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Conv1d" }
}

/// 2D Convolution layer.
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    training: bool,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_config(in_channels, out_channels, (kernel_size, kernel_size),
            (1, 1), (0, 0), (1, 1), 1, true)
    }

    pub fn with_config(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
    ) -> Self {
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let k = 1.0 / (fan_in as f64).sqrt();

        let weight = Tensor::uniform(
            [out_channels, in_channels / groups, kernel_size.0, kernel_size.1],
            -k, k, DType::F32, Device::Cpu,
        ).with_requires_grad(true);

        let bias_tensor = if bias {
            Some(Tensor::uniform([out_channels], -k, k, DType::F32, Device::Cpu)
                .with_requires_grad(true))
        } else {
            None
        };

        Self { weight, bias: bias_tensor, stride, padding, dilation, groups, training: true }
    }

    pub fn stride(mut self, s: usize) -> Self { self.stride = (s, s); self }
    pub fn padding(mut self, p: usize) -> Self { self.padding = (p, p); self }
}

impl Module for Conv2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        conv2d_forward(input, &self.weight, self.bias.as_ref(),
                       self.stride, self.padding, self.dilation, self.groups)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias { params.push(bias.clone()); }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "Conv2d" }
}

/// 2D Max Pooling layer.
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self { kernel_size: (kernel_size, kernel_size), stride: (kernel_size, kernel_size), padding: (0, 0) }
    }

    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size: (kernel_size, kernel_size), stride: (stride, stride), padding: (0, 0) }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        max_pool2d_forward(input, self.kernel_size, self.stride, self.padding)
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn name(&self) -> &str { "MaxPool2d" }
}

/// 2D Average Pooling layer.
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    pub fn new(kernel_size: usize) -> Self {
        Self { kernel_size: (kernel_size, kernel_size), stride: (kernel_size, kernel_size), padding: (0, 0) }
    }

    pub fn with_stride(kernel_size: usize, stride: usize) -> Self {
        Self { kernel_size: (kernel_size, kernel_size), stride: (stride, stride), padding: (0, 0) }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        avg_pool2d_forward(input, self.kernel_size, self.stride, self.padding)
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn name(&self) -> &str { "AvgPool2d" }
}

/// Adaptive Average Pooling to target output size.
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self { Self { output_size } }
    pub fn square(size: usize) -> Self { Self { output_size: (size, size) } }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        adaptive_avg_pool2d_forward(input, self.output_size)
    }
    fn parameters(&self) -> Vec<Tensor> { vec![] }
    fn name(&self) -> &str { "AdaptiveAvgPool2d" }
}

// ============ Functional API ============

fn conv1d_forward(
    input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
    stride: usize, padding: usize, dilation: usize, groups: usize,
) -> Result<Tensor> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 3 {
        return Err(FerrumError::ShapeMismatch {
            operation: "conv1d",
            expected: "3D [N, C, L]".to_string(),
            actual: format!("{:?}", input_shape),
        });
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_length = input_shape[2];
    let out_channels = weight_shape[0];
    let kernel_size = weight_shape[2];

    let out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    let input_data = input.to_vec::<f32>()?;
    let weight_data = weight.to_vec::<f32>()?;
    let mut output_data = vec![0.0f32; batch * out_channels * out_length];

    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    for n in 0..batch {
        for g in 0..groups {
            for oc in 0..out_c_per_group {
                let out_c = g * out_c_per_group + oc;
                for ol in 0..out_length {
                    let mut sum = 0.0f32;
                    for ic in 0..in_c_per_group {
                        let in_c = g * in_c_per_group + ic;
                        for k in 0..kernel_size {
                            let il = ol * stride + k * dilation;
                            if il >= padding && il < in_length + padding {
                                let in_idx = n * in_channels * in_length + in_c * in_length + (il - padding);
                                let w_idx = out_c * in_c_per_group * kernel_size + ic * kernel_size + k;
                                sum += input_data[in_idx] * weight_data[w_idx];
                            }
                        }
                    }
                    output_data[n * out_channels * out_length + out_c * out_length + ol] = sum;
                }
            }
        }
    }

    if let Some(bias) = bias {
        let bias_data = bias.to_vec::<f32>()?;
        for n in 0..batch {
            for oc in 0..out_channels {
                for ol in 0..out_length {
                    let idx = n * out_channels * out_length + oc * out_length + ol;
                    output_data[idx] += bias_data[oc];
                }
            }
        }
    }

    Tensor::from_slice(&output_data, (batch, out_channels, out_length), input.device())
}

fn conv2d_forward(
    input: &Tensor, weight: &Tensor, bias: Option<&Tensor>,
    stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize), groups: usize,
) -> Result<Tensor> {
    let input_shape = input.shape();
    let weight_shape = weight.shape();

    if input_shape.len() != 4 {
        return Err(FerrumError::ShapeMismatch {
            operation: "conv2d",
            expected: "4D [N, C, H, W]".to_string(),
            actual: format!("{:?}", input_shape),
        });
    }

    let batch = input_shape[0];
    let in_channels = input_shape[1];
    let in_h = input_shape[2];
    let in_w = input_shape[3];
    let out_channels = weight_shape[0];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    let out_h = (in_h + 2 * padding.0 - dilation.0 * (kh - 1) - 1) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - dilation.1 * (kw - 1) - 1) / stride.1 + 1;

    let input_data = input.to_vec::<f32>()?;
    let weight_data = weight.to_vec::<f32>()?;
    let mut output_data = vec![0.0f32; batch * out_channels * out_h * out_w];

    let in_c_per_group = in_channels / groups;
    let out_c_per_group = out_channels / groups;

    for n in 0..batch {
        for g in 0..groups {
            for oc in 0..out_c_per_group {
                let out_c = g * out_c_per_group + oc;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        for ic in 0..in_c_per_group {
                            let in_c = g * in_c_per_group + ic;
                            for i in 0..kh {
                                for j in 0..kw {
                                    let ih = oh * stride.0 + i * dilation.0;
                                    let iw = ow * stride.1 + j * dilation.1;
                                    if ih >= padding.0 && ih < in_h + padding.0 &&
                                       iw >= padding.1 && iw < in_w + padding.1 {
                                        let in_idx = n * in_channels * in_h * in_w 
                                            + in_c * in_h * in_w 
                                            + (ih - padding.0) * in_w 
                                            + (iw - padding.1);
                                        let w_idx = out_c * in_c_per_group * kh * kw 
                                            + ic * kh * kw + i * kw + j;
                                        sum += input_data[in_idx] * weight_data[w_idx];
                                    }
                                }
                            }
                        }
                        let out_idx = n * out_channels * out_h * out_w + out_c * out_h * out_w + oh * out_w + ow;
                        output_data[out_idx] = sum;
                    }
                }
            }
        }
    }

    if let Some(bias) = bias {
        let bias_data = bias.to_vec::<f32>()?;
        for n in 0..batch {
            for oc in 0..out_channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let idx = n * out_channels * out_h * out_w + oc * out_h * out_w + oh * out_w + ow;
                        output_data[idx] += bias_data[oc];
                    }
                }
            }
        }
    }

    Tensor::from_slice(&output_data, (batch, out_channels, out_h, out_w), input.device())
}

fn max_pool2d_forward(
    input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize),
) -> Result<Tensor> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrumError::ShapeMismatch {
            operation: "max_pool2d",
            expected: "4D [N, C, H, W]".to_string(),
            actual: format!("{:?}", shape),
        });
    }

    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let out_h = (in_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let input_data = input.to_vec::<f32>()?;
    let mut output_data = vec![f32::NEG_INFINITY; batch * channels * out_h * out_w];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut max_val = f32::NEG_INFINITY;
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;
                            if ih >= padding.0 && ih < in_h + padding.0 &&
                               iw >= padding.1 && iw < in_w + padding.1 {
                                let idx = n * channels * in_h * in_w + c * in_h * in_w 
                                    + (ih - padding.0) * in_w + (iw - padding.1);
                                max_val = max_val.max(input_data[idx]);
                            }
                        }
                    }
                    output_data[n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow] = max_val;
                }
            }
        }
    }

    Tensor::from_slice(&output_data, (batch, channels, out_h, out_w), input.device())
}

fn avg_pool2d_forward(
    input: &Tensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize),
) -> Result<Tensor> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrumError::ShapeMismatch {
            operation: "avg_pool2d",
            expected: "4D [N, C, H, W]".to_string(),
            actual: format!("{:?}", shape),
        });
    }

    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let out_h = (in_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
    let out_w = (in_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

    let input_data = input.to_vec::<f32>()?;
    let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    let mut count = 0;
                    for kh in 0..kernel_size.0 {
                        for kw in 0..kernel_size.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;
                            if ih >= padding.0 && ih < in_h + padding.0 &&
                               iw >= padding.1 && iw < in_w + padding.1 {
                                let idx = n * channels * in_h * in_w + c * in_h * in_w 
                                    + (ih - padding.0) * in_w + (iw - padding.1);
                                sum += input_data[idx];
                                count += 1;
                            }
                        }
                    }
                    let out_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }

    Tensor::from_slice(&output_data, (batch, channels, out_h, out_w), input.device())
}

fn adaptive_avg_pool2d_forward(input: &Tensor, output_size: (usize, usize)) -> Result<Tensor> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrumError::ShapeMismatch {
            operation: "adaptive_avg_pool2d",
            expected: "4D [N, C, H, W]".to_string(),
            actual: format!("{:?}", shape),
        });
    }

    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let (out_h, out_w) = output_size;

    let input_data = input.to_vec::<f32>()?;
    let mut output_data = vec![0.0f32; batch * channels * out_h * out_w];

    for n in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih_start = (oh * in_h) / out_h;
                    let ih_end = ((oh + 1) * in_h) / out_h;
                    let iw_start = (ow * in_w) / out_w;
                    let iw_end = ((ow + 1) * in_w) / out_w;

                    let mut sum = 0.0f32;
                    let mut count = 0;
                    for ih in ih_start..ih_end {
                        for iw in iw_start..iw_end {
                            let idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                            sum += input_data[idx];
                            count += 1;
                        }
                    }

                    let out_idx = n * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                    output_data[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                }
            }
        }
    }

    Tensor::from_slice(&output_data, (batch, channels, out_h, out_w), input.device())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_forward() {
        let conv = Conv2d::new(1, 1, 3);
        let input = Tensor::ones((1, 1, 5, 5), DType::F32, Device::Cpu);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        let conv = Conv2d::new(1, 1, 3).padding(1);
        let input = Tensor::ones((1, 1, 5, 5), DType::F32, Device::Cpu);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_maxpool2d() {
        let pool = MaxPool2d::new(2);
        let input = Tensor::ones((1, 1, 4, 4), DType::F32, Device::Cpu);
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_avgpool2d() {
        let pool = AvgPool2d::new(2);
        let input = Tensor::ones((1, 1, 4, 4), DType::F32, Device::Cpu);
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_adaptive_avgpool2d() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Tensor::ones((1, 1, 7, 7), DType::F32, Device::Cpu);
        let output = pool.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
    }
}
