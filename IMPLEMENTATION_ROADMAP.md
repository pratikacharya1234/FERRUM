# FERRUM Implementation Roadmap

**Current Version**: 1.0.0
**Status**: Production Ready (Core Features)

---

## Completed (v1.0.0)

### Core Infrastructure
- [x] N-dimensional tensor system
- [x] All data types (F32, F64, I32, I64, U8)
- [x] Broadcasting and views
- [x] Device abstraction (CPU, CUDA placeholder)

### Autograd
- [x] Gradient tape with automatic differentiation
- [x] backward() function
- [x] no_grad() context
- [x] Gradient accumulation for reused tensors
- [x] Gradient checking tests

### Neural Network Layers
- [x] Linear (fully connected)
- [x] Activations: ReLU, Sigmoid, Tanh, GELU, SiLU, LeakyReLU, ELU
- [x] Softmax, LogSoftmax
- [x] LayerNorm, BatchNorm1d
- [x] Dropout
- [x] Sequential container

### Training
- [x] SGD optimizer (momentum, weight decay, Nesterov)
- [x] Adam optimizer
- [x] Loss functions: MSE, BCE, CrossEntropy, NLL, L1, SmoothL1
- [x] XOR example with 100% accuracy

### Data Loading
- [x] Dataset trait
- [x] TensorDataset
- [x] DataLoader with batching
- [x] Samplers: Sequential, Random, Weighted, Subset, Distributed, Batch
- [x] Transforms: Normalize, Compose

### Distributed (Infrastructure)
- [x] ProcessGroup
- [x] Collectives: broadcast, all_reduce, reduce, gather, scatter, barrier
- [x] DistributedDataParallel wrapper

### Serialization
- [x] Save/load tensors
- [x] Model state dict

---

## Phase 2: CPU Optimization

**Priority**: High
**Estimated Time**: 2-3 weeks

### Goals
- [ ] Integrate OpenBLAS for fast matrix operations
- [ ] Optional MKL support
- [ ] SIMD optimizations for element-wise operations
- [ ] Benchmark suite

### Tasks
1. Add openblas-src dependency with feature flag
2. Replace naive matmul with BLAS dgemm/sgemm
3. Implement SIMD for add, mul, exp, etc.
4. Create benchmark comparing with PyTorch

### Expected Performance
- 10-100x speedup for large matrix operations
- Competitive with PyTorch CPU for standard benchmarks

---

## Phase 3: Real CUDA GPU Support

**Priority**: High
**Estimated Time**: 4-6 weeks

### Goals
- [ ] Real CUDA kernel execution
- [ ] cuBLAS integration for matrix operations
- [ ] Efficient memory management
- [ ] Multi-GPU support

### Tasks
1. CUDA FFI bindings using cuda-sys
2. Implement kernels: add, mul, matmul, reductions
3. cuBLAS integration for matmul
4. Unified memory or explicit transfers
5. Multi-stream execution

### Dependencies
- CUDA Toolkit 11.0+
- cuBLAS, cuDNN

---

## Phase 4: Convolutional Layers

**Priority**: High
**Estimated Time**: 3-4 weeks

### Goals
- [ ] Conv1d, Conv2d, Conv3d
- [ ] MaxPool2d, AvgPool2d
- [ ] Adaptive pooling
- [ ] Transposed convolution

### Layers to Implement
```rust
Conv2d::new(in_channels, out_channels, kernel_size, stride, padding)
MaxPool2d::new(kernel_size, stride, padding)
AvgPool2d::new(kernel_size, stride, padding)
AdaptiveAvgPool2d::new(output_size)
ConvTranspose2d::new(in_channels, out_channels, kernel_size)
```

### Tasks
1. Im2col implementation for CPU
2. cuDNN integration for GPU
3. Backward pass for all layers
4. Benchmarks against PyTorch

---

## Phase 5: Recurrent Layers

**Priority**: Medium
**Estimated Time**: 3-4 weeks

### Goals
- [ ] RNN (vanilla)
- [ ] LSTM
- [ ] GRU
- [ ] Bidirectional variants

### Layers to Implement
```rust
RNN::new(input_size, hidden_size, num_layers)
LSTM::new(input_size, hidden_size, num_layers)
GRU::new(input_size, hidden_size, num_layers)
```

### Features
- Batch-first option
- Bidirectional support
- Dropout between layers
- Packed sequences

---

## Phase 6: Transformers and Attention

**Priority**: Medium
**Estimated Time**: 4-6 weeks

### Goals
- [ ] Multi-head attention
- [ ] Transformer encoder/decoder
- [ ] Positional encoding
- [ ] Flash Attention optimization

### Layers to Implement
```rust
MultiheadAttention::new(embed_dim, num_heads)
TransformerEncoderLayer::new(d_model, nhead, dim_feedforward)
TransformerDecoderLayer::new(d_model, nhead, dim_feedforward)
TransformerEncoder::new(encoder_layer, num_layers)
TransformerDecoder::new(decoder_layer, num_layers)
PositionalEncoding::new(d_model, max_len)
```

### Optimizations
- Flash Attention for memory efficiency
- KV caching for inference

---

## Phase 7: Model Zoo

**Priority**: Low
**Estimated Time**: Ongoing

### Goals
- [ ] Pre-trained model loading
- [ ] Common architectures
- [ ] Model conversion from PyTorch

### Models to Include
- ResNet (18, 34, 50, 101, 152)
- VGG (11, 13, 16, 19)
- MobileNet
- BERT, GPT-2
- ViT (Vision Transformer)

### Infrastructure
- Weight download utility
- PyTorch checkpoint converter
- Hugging Face model compatibility

---

## Phase 8: Real Distributed Training (NCCL)

**Priority**: Medium
**Estimated Time**: 3-4 weeks

### Goals
- [ ] NCCL backend for GPU collective operations
- [ ] Multi-node training
- [ ] Gradient compression

### Tasks
1. NCCL bindings
2. Replace simulated collectives with real NCCL calls
3. MPI integration for multi-node
4. Ring-AllReduce optimization

---

## Phase 9: Mixed Precision Training

**Priority**: Medium
**Estimated Time**: 2-3 weeks

### Goals
- [ ] FP16/BF16 tensor support
- [ ] Automatic mixed precision (AMP)
- [ ] Loss scaling
- [ ] Tensor cores utilization

### API
```rust
let scaler = GradScaler::new();
let model = model.half();  // Convert to FP16

// Training loop
let loss = model.forward(&input.half())?;
scaler.scale(&loss)?.backward()?;
scaler.step(&optimizer)?;
scaler.update();
```

---

## Phase 10: Testing and Benchmarks

**Priority**: Ongoing
**Estimated Time**: Ongoing

### Goals
- [ ] Comprehensive unit tests for all operations
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] CI/CD pipeline

### Benchmarks to Create
- Tensor operations vs NumPy
- Neural network layers vs PyTorch
- Training throughput vs PyTorch
- Memory usage comparison

### Testing Goals
- 90%+ code coverage
- Fuzz testing for edge cases
- Property-based testing

---

## Phase 11: Documentation

**Priority**: Ongoing
**Estimated Time**: Ongoing

### Goals
- [x] API Reference
- [x] Quick Start Guide
- [ ] Tutorial series
- [ ] Architecture documentation
- [ ] Contribution guide improvements
- [ ] Rustdoc comments for all public APIs

### Tutorials to Write
1. Building your first neural network
2. Image classification with Conv2d
3. Sequence modeling with LSTM
4. Fine-tuning transformers
5. Distributed training guide
6. Custom layer implementation

---

## Version Timeline

| Version | Target | Key Features |
|---------|--------|--------------|
| v1.0.0 | Done | Core framework, autograd, basic layers |
| v1.1.0 | Q2 2026 | BLAS optimization, Conv2d |
| v1.2.0 | Q3 2026 | LSTM/GRU, real CUDA |
| v1.3.0 | Q4 2026 | Transformers, model zoo |
| v2.0.0 | 2027 | Production-grade, PyTorch parity |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute.

Priority areas:
1. BLAS integration
2. CUDA kernels
3. Conv2d implementation
4. Test coverage
