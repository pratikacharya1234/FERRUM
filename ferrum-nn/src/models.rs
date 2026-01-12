//! Pre-built model architectures.
//!
//! Implementations of popular deep learning models.

use ferrum_core::{Device, DType, Result, Tensor};
use crate::module::Module;
use crate::linear::Linear;
use crate::conv::{Conv2d, MaxPool2d, AdaptiveAvgPool2d};
use crate::activation::ReLU;
use crate::container::Sequential;

/// ResNet basic block (for ResNet-18/34).
pub struct BasicBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    downsample: Option<Conv2d>,
    stride: usize,
    training: bool,
}

impl BasicBlock {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize) -> Self {
        let downsample = if stride != 1 || in_channels != out_channels {
            Some(Conv2d::with_config(
                in_channels, out_channels, (1, 1), (stride, stride), (0, 0), (1, 1), 1, false
            ))
        } else {
            None
        };

        Self {
            conv1: Conv2d::with_config(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), (1, 1), 1, false),
            conv2: Conv2d::with_config(out_channels, out_channels, (3, 3), (1, 1), (1, 1), (1, 1), 1, false),
            downsample,
            stride,
            training: true,
        }
    }
}

impl Module for BasicBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let identity = if let Some(ref ds) = self.downsample {
            ds.forward(input)?
        } else {
            input.clone()
        };

        let x = self.conv1.forward(input)?;
        let x = x.relu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.add(&identity)?;
        x.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        if let Some(ref ds) = self.downsample {
            params.extend(ds.parameters());
        }
        params
    }

    fn name(&self) -> &str { "BasicBlock" }
}

/// ResNet model.
pub struct ResNet {
    conv1: Conv2d,
    maxpool: MaxPool2d,
    layer1: Vec<BasicBlock>,
    layer2: Vec<BasicBlock>,
    layer3: Vec<BasicBlock>,
    layer4: Vec<BasicBlock>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear,
    num_classes: usize,
    training: bool,
}

impl ResNet {
    /// Create ResNet-18.
    pub fn resnet18(num_classes: usize) -> Self {
        Self::new(&[2, 2, 2, 2], num_classes)
    }

    /// Create ResNet-34.
    pub fn resnet34(num_classes: usize) -> Self {
        Self::new(&[3, 4, 6, 3], num_classes)
    }

    fn new(layers: &[usize; 4], num_classes: usize) -> Self {
        let conv1 = Conv2d::with_config(3, 64, (7, 7), (2, 2), (3, 3), (1, 1), 1, false);
        let maxpool = MaxPool2d::with_stride(3, 2);

        let layer1 = Self::make_layer(64, 64, layers[0], 1);
        let layer2 = Self::make_layer(64, 128, layers[1], 2);
        let layer3 = Self::make_layer(128, 256, layers[2], 2);
        let layer4 = Self::make_layer(256, 512, layers[3], 2);

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(512, num_classes);

        Self {
            conv1, maxpool, layer1, layer2, layer3, layer4,
            avgpool, fc, num_classes, training: true,
        }
    }

    fn make_layer(in_channels: usize, out_channels: usize, blocks: usize, stride: usize) -> Vec<BasicBlock> {
        let mut layers = Vec::new();
        layers.push(BasicBlock::new(in_channels, out_channels, stride));
        for _ in 1..blocks {
            layers.push(BasicBlock::new(out_channels, out_channels, 1));
        }
        layers
    }

    fn forward_layer(&self, x: &Tensor, layer: &[BasicBlock]) -> Result<Tensor> {
        let mut out = x.clone();
        for block in layer {
            out = block.forward(&out)?;
        }
        Ok(out)
    }
}

impl Module for ResNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(input)?;
        let x = x.relu()?;
        let x = self.maxpool.forward(&x)?;

        let x = self.forward_layer(&x, &self.layer1)?;
        let x = self.forward_layer(&x, &self.layer2)?;
        let x = self.forward_layer(&x, &self.layer3)?;
        let x = self.forward_layer(&x, &self.layer4)?;

        let x = self.avgpool.forward(&x)?;
        
        // Flatten
        let batch = x.shape()[0];
        let x = x.reshape([batch, 512])?;
        
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        for block in &self.layer1 { params.extend(block.parameters()); }
        for block in &self.layer2 { params.extend(block.parameters()); }
        for block in &self.layer3 { params.extend(block.parameters()); }
        for block in &self.layer4 { params.extend(block.parameters()); }
        params.extend(self.fc.parameters());
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "ResNet" }
}

/// VGG model configuration.
pub struct VGG {
    features: Vec<Box<dyn Module>>,
    classifier: Vec<Linear>,
    training: bool,
}

impl VGG {
    /// Create VGG-16.
    pub fn vgg16(num_classes: usize) -> Self {
        Self::new(&[64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512], num_classes)
    }

    /// Create VGG-19.
    pub fn vgg19(num_classes: usize) -> Self {
        Self::new(&[64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512], num_classes)
    }

    fn new(cfg: &[usize], num_classes: usize) -> Self {
        let mut features: Vec<Box<dyn Module>> = Vec::new();
        let mut in_channels = 3;

        for &v in cfg {
            features.push(Box::new(Conv2d::new(in_channels, v, 3).padding(1)));
            in_channels = v;
        }

        let classifier = vec![
            Linear::new(512 * 7 * 7, 4096),
            Linear::new(4096, 4096),
            Linear::new(4096, num_classes),
        ];

        Self { features, classifier, training: true }
    }
}

impl Module for VGG {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for layer in &self.features {
            x = layer.forward(&x)?;
            x = x.relu()?;
        }

        // Flatten
        let batch = x.shape()[0];
        let flat_size = x.numel() / batch;
        x = x.reshape([batch, flat_size])?;

        for (i, layer) in self.classifier.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.classifier.len() - 1 {
                x = x.relu()?;
            }
        }

        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.features {
            params.extend(layer.parameters());
        }
        for layer in &self.classifier {
            params.extend(layer.parameters());
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "VGG" }
}

/// Simple MLP model.
pub struct MLP {
    layers: Vec<Linear>,
    training: bool,
}

impl MLP {
    pub fn new(sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Linear::new(sizes[i], sizes[i + 1]));
        }
        Self { layers, training: true }
    }
}

impl Module for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut x = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;
            if i < self.layers.len() - 1 {
                x = x.relu()?;
            }
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    fn train(&mut self) { self.training = true; }
    fn eval(&mut self) { self.training = false; }
    fn is_training(&self) -> bool { self.training }
    fn name(&self) -> &str { "MLP" }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_block() {
        let block = BasicBlock::new(64, 64, 1);
        let input = Tensor::randn((1, 64, 32, 32), DType::F32, Device::Cpu);
        let output = block.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 64, 32, 32]);
    }

    #[test]
    fn test_mlp() {
        let mlp = MLP::new(&[784, 256, 128, 10]);
        let input = Tensor::randn((2, 784), DType::F32, Device::Cpu);
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 10]);
    }
}
