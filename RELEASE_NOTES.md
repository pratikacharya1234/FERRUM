# FERRUM v1.0.0 Release Notes

**Release Date**: January 12, 2026
**License**: Apache 2.0

---

## Overview

FERRUM v1.0.0 is the first public release of the Rust deep learning framework. This release provides a PyTorch-like API for building and training neural networks entirely in Rust.

---

## Features

### Core Features

- **Tensor System**: N-dimensional arrays with broadcasting, views, and standard operations
- **Autograd**: Automatic differentiation with tape-based reverse-mode AD
- **Neural Networks**: Linear layers, activations, normalization, dropout
- **Optimizers**: SGD (with momentum) and Adam
- **Data Loading**: PyTorch-style DataLoader with batching and shuffling
- **Serialization**: Save and load model weights

### Crates

| Crate | Description |
|-------|-------------|
| ferrum | Main facade crate |
| ferrum-core | Tensor primitives |
| ferrum-autograd | Automatic differentiation |
| ferrum-ops | Tensor operations |
| ferrum-nn | Neural network layers |
| ferrum-optim | Optimizers |
| ferrum-data | Data loading |
| ferrum-distributed | Distributed training |
| ferrum-cuda | GPU support (simulated) |
| ferrum-serialize | Model persistence |

### Test Coverage

- 154 unit tests passing
- XOR training achieves 100% accuracy

---

## Installation

```toml
[dependencies]
ferrum = { git = "https://github.com/pratikacharya1234/FERRUM" }
```

---

## Quick Example

```rust
use ferrum::prelude::*;

fn main() -> Result<()> {
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU::new())
        .add(Linear::new(256, 10));

    let input = Tensor::randn([32, 784], DType::F32, Device::Cpu);
    let output = model.forward(&input)?;

    println!("Output shape: {:?}", output.shape());
    Ok(())
}
```

---

## Supported Features

### Layers
- Linear (fully connected)
- ReLU, Sigmoid, Tanh
- GELU, SiLU, LeakyReLU, ELU
- Softmax, LogSoftmax
- LayerNorm, BatchNorm1d
- Dropout
- Sequential

### Loss Functions
- MSE Loss
- Binary Cross Entropy
- Cross Entropy
- Negative Log Likelihood
- L1 Loss
- Smooth L1 Loss

### Optimizers
- SGD (with momentum, weight decay)
- Adam

### Data Loading
- Dataset trait
- TensorDataset
- DataLoader
- Samplers: Sequential, Random, Weighted, Distributed

---

## Limitations

This release has the following limitations:

1. **GPU**: CUDA support is simulated (kernel stubs only, no actual GPU execution)
2. **Performance**: No BLAS integration, uses naive O(n^3) matmul
3. **Layers**: No Conv2d, LSTM, Transformer yet
4. **Distributed**: Single-process simulation only

---

## Roadmap

### v1.1 (Planned)
- Convolutional layers (Conv2d, MaxPool2d)
- OpenBLAS integration

### v1.2 (Planned)
- Recurrent layers (LSTM, GRU)
- Real CUDA support

### v1.3 (Planned)
- Transformer layers
- Attention mechanisms
- Pre-trained model support

---

## Breaking Changes

N/A - This is the initial release.

---

## Contributors

- Pratik Acharya

---

## Links

- Repository: https://github.com/pratikacharya1234/FERRUM
- Documentation: See docs/API_REFERENCE.md
- Quick Start: See QUICKSTART.md
