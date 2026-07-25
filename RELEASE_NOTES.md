# FERRUM Release Notes

## Unreleased (July 2026)

### Added — GPU op dispatch (VERIFIED on A100, 2026-07-24: 28/28 correctness gate)
- **`Tensor` ops now dispatch to the GPU for `Device::Cuda` tensors.**
  `ferrum-core` gained a `gpu::GpuBackend` trait + registry (no dependency
  cycle: `ferrum-cuda` implements it and installs it via
  `ferrum_cuda::register_gpu_backend()`). Dispatched ops: matmul (cuBLAS
  SGEMM, TF32, cached handle), add/sub/mul/div with broadcasting (strided
  NVRTC kernel), relu/exp/log/sqrt/sigmoid/tanh/neg, scalar ops and `pow`,
  sum/mean/`sum_dim` (reduction kernels), `contiguous()` on views (strided
  copy kernel), and the in-place optimizer updates. F32 only. Ops without a
  GPU implementation fail loudly (`NotImplemented`) — no silent CPU
  fallback for CUDA tensors.
- **`to_device` now really copies** (`cuMemcpyHtoD`/`DtoH`); it previously
  just relabeled the device field. `item()`/`to_vec()` read CUDA tensors
  through `cuMemcpyDtoH`; constructors (`ones`/`full`/`randn`/`uniform`/
  `normal`) build on the host and upload.
- **`examples/gpu_smoke.rs`** — correctness gate comparing every dispatched
  op (plus one full autograd step) against CPU results. Run it on the GPU
  machine before trusting GPU training.
- `Linear::to_device`, device selection in `examples/cifar10.rs`
  (`CIFAR10_DEVICE=cpu|cuda`, auto-detect by default), and `run_colab.sh`
  now runs smoke test → benchmarks → GPU CIFAR-10 training.
- **Fixed during GPU verification: `CudaDevice::new` created a second CUDA
  context per call**, invalidating kernel/module handles cached under the
  first context (surfaced as `cuLaunchKernel` error 400 on the first kernel
  reuse after cuBLAS handle creation). It now reuses the thread's current
  context and only destroys a context it created.
- **Fixed: NVRTC kernels compiled with `--use_fast_math`.** Fast-math
  `powf` computes `exp2(y·log2(x))`, which is NaN for negative bases —
  Adam's `grad.pow(2.0)` turned every weight NaN on the first optimizer
  step (GPU training showed `loss NaN` at 10% accuracy while all
  fixed-input smoke checks passed). Fast math is disabled and `pow(2.0)` /
  `pow(0.5)` are special-cased to `x*x` / `sqrtf(x)`; the smoke test now
  covers mixed-sign `pow` and the full Adam denominator.
- Status: all 217 CPU tests pass, CPU training unchanged, `gpu_smoke`
  passes **41/41 on A100-SXM4-40GB (CUDA 12.8)**, and **CIFAR-10 GPU
  training is verified: loss 1.93 → 1.39 over 3 epochs (test acc 39.1%,
  matching CPU's curve) at 0.2 s/epoch vs 33 s CPU — ~165× faster.**

### Fixed
- **`ferrum_nn::Linear` produced transposed weight gradients**, making it
  untrainable. Root cause: forward used `input.matmul(&weight.t()?)`, and
  transposed views share their base tensor's autograd id, so the gradient
  (shaped like `weight.t()`) was deposited on `weight`. `Linear` now stores its
  weight `[in_features, out_features]` and multiplies without a transpose.
- **XOR example bias gradients had the wrong shape.** The example pre-expanded
  the bias (`unsqueeze(0).expand(...)`) before adding; `expand` is not tracked
  by autograd, so the bias gradient kept the expanded shape. The example now
  uses implicit broadcasting, whose backward reduces gradients correctly.
  Still trains to 4/4 (100%), with lower final loss.

- **`Tensor::expand` produced out-of-bounds views when applied to an existing
  view.** It computed broadcast strides from the shape's *contiguous* layout
  instead of the tensor's actual strides, so expanding an already-expanded
  (stride-0) tensor built views that indexed past their storage. Reduction
  backwards (`SumBackward`/`MeanBackward`/`SumDimBackward`) stack exactly such
  expands; chaining `sum_dim` → `mean` in a loss panicked with an
  index-out-of-bounds in `contiguous()`. `expand` now derives strides from the
  tensor's real strides.

### Added
- **`examples/cifar10.rs`** — first end-to-end training example on a real
  dataset: CIFAR-10 binary loader, 3072→256→10 MLP built from
  `ferrum_nn::Linear`, differentiable cross-entropy composed from primitives
  with verified backward passes, Adam. Configurable via `CIFAR10_*` env vars.
  Verified run (5000 train / 1000 test images, 3 epochs, CPU): train loss
  1.93 → 1.40, test accuracy 34.8% → 39.8% vs the 10% random baseline.
- **`examples/grad_probe.rs`** — gradient sanity checker that verifies
  gradient existence/shape for the training patterns (broadcast bias add,
  `Linear`, primitive log-softmax, in-place optimizer updates). Run it after
  touching autograd or layer code.

### Changed
- FP16 `cublasGemmEx` benchmarked on A100 across algorithm constants:
  `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99) gives 272 TFLOPS vs 230 for
  `CUBLAS_GEMM_DEFAULT` (−1). The deprecated constant is kept deliberately and
  documented in `ferrum-cuda/src/cublas.rs`. Reaching PyTorch's ~312 TFLOPS
  requires a `cublasLt` implementation (roadmap).
- Documentation rewritten to state verified status honestly — see
  [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), including the key
  limitation that `Tensor` compute ops are CPU-only and the benchmarked CUDA
  layer is not yet dispatched from the Tensor API.

---

# FERRUM v1.0.0

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
| ferrum-cuda | GPU support |
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

## Limitations (v1.0.0, corrected July 2026)

The original limitations list overstated capabilities. The accurate list:

1. **GPU**: CUDA is used for storage (alloc/copy) only; tensor compute ops run
   on CPU. GPU kernels exist in `ferrum-cuda` but are not dispatched from the
   Tensor API.
2. **Performance**: `Tensor::matmul` on CPU is a naive triple loop; the faster
   variants in `ferrum-ops` are not wired in.
3. **Layers**: Linear + activations train end to end. Conv2d, LSTM, GRU, and
   Transformer have forward implementations only; trainability undemonstrated.
   (`Linear` itself had a transposed-gradient bug until July 2026 — see
   Unreleased.)
4. **Distributed**: local/simulated collectives; no verified multi-node run.
5. **Serialization**: no test coverage.

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
