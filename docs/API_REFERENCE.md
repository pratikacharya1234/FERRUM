# FERRUM API Reference

An overview of the API surface that is exercised by the working examples and
tests. For exhaustive signatures, generate rustdoc:

```bash
cargo doc --workspace --open
```

Conventions in this document:
- **Verified** — used in a passing example or test in this repository.
- **Exists** — implemented and unit-tested, but not proven in end-to-end
  training. Use with care and check gradients with `examples/grad_probe.rs`.

Status caveats that apply everywhere: GPU compute on `Device::Cuda` tensors
is **F32-only** and requires `ferrum_cuda::register_gpu_backend()` at
startup (verified 28/28 vs CPU on A100 by `examples/gpu_smoke.rs`); ops
without a GPU implementation error loudly rather than falling back to CPU.
The differentiable op set is the one listed under [Autograd](#autograd).
See [../IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md).

---

## Prelude

```rust
use ferrum::prelude::*;
// Brings in: Tensor, Shape, DType, Device, Result, FerrumError, ...
```

---

## Core Types

### `Device`

```rust
Device::Cpu          // verified for all ops
Device::Cuda(usize)  // F32 compute via registered backend; A100-verified
Device::default()    // Cuda(0) if a GPU is detected, else Cpu
```

For GPU compute, call `ferrum_cuda::register_gpu_backend()` once at startup;
without it, CUDA tensors error on compute ops (by design — no silent CPU
fallback).

### `DType`

`F32`, `F64`, `I32`, `I64`, `F16`, `Bool`. Training paths are verified with
`F32`.

### `Result` / `FerrumError`

All fallible APIs return `ferrum_core::Result<T>`. Errors carry the operation
and shapes/devices involved (`ShapeMismatch`, `DeviceMismatch`,
`DTypeMismatch`, `NotImplemented`, …).

---

## Tensor

### Creation (verified)

```rust
Tensor::zeros(shape, dtype, device)
Tensor::ones(shape, dtype, device)
Tensor::randn(shape, dtype, device)              // standard normal
Tensor::uniform(shape, low, high, dtype, device) // uniform in [low, high)
Tensor::from_slice(&data, shape, device)?        // copies from a host slice
```

`shape` is anything `Into<Shape>`, e.g. `[4, 2]`.

### Inspection (verified)

```rust
t.shape()      // &[usize]
t.dtype()
t.device()
t.item()?      // scalar tensors -> f64
t.to_vec::<f32>()?  // copy to host Vec
```

### Element-wise ops (verified, differentiable, broadcasting)

```rust
a.add(&b)?   a.sub(&b)?   a.mul(&b)?   a.div(&b)?
a.pow(2.0)?  a.exp()?     a.log()?     a.neg()?
a.relu()?    a.sigmoid()? a.tanh()?
a.mul_scalar(0.5)?
```

Broadcasting follows NumPy rules; gradients are reduced back to each
operand's original shape.

### Matrix multiplication (verified, differentiable)

```rust
x.matmul(&w)?   // 2D @ 2D, 2D @ 1D, 1D @ 2D
```

CPU implementation is a naive triple loop — correct but slow (roadmap
Phase 3).

**Warning:** don't multiply parameters through a transposed view
(`x.matmul(&w.t()?)`): views share the base tensor's autograd id, so the
gradient arrives transposed on `w`. Store parameters in the layout you
multiply with.

### Reductions (verified, differentiable)

```rust
t.sum()?              // scalar
t.mean()?             // scalar (over ALL elements)
t.sum_dim(dim, keepdim)?
```

### Shape manipulation

```rust
t.reshape(shape)?     // exists
t.contiguous()?       // verified
t.narrow(dim, start, len)?  // verified (batch slicing)
t.squeeze(Some(dim))? t.unsqueeze(dim)?  // verified
t.cat(...), Tensor::stack(...)           // exists; cat is differentiable
t.t()? / t.transpose(d0, d1)?  // verified as views — see matmul warning
t.expand(shape)?      // exists — NOT tracked by autograd; avoid before
                      // differentiable ops, use implicit broadcasting instead
t.to_device(device)?  // exists; CPU<->CUDA copies are storage-level only
```

---

## Autograd

```rust
use ferrum_autograd::tape::GradientTape;

let w = Tensor::randn([3, 2], DType::F32, Device::Cpu).with_requires_grad(true);

GradientTape::with_tape(|_tape| -> Result<()> {
    let loss = x.matmul(&w)?.pow(2.0)?.mean()?;
    loss.backward()?;
    Ok(())
})?;

let g = w.grad();  // Option<Tensor>, same shape as w
```

- Run forward + `backward()` inside `with_tape`; step the optimizer outside.
- Gradients accumulate; call `optimizer.zero_grad()` each iteration.

**Differentiable ops (verified backward)**: `add`, `sub`, `mul`, `div`,
`matmul`, `sum`, `mean`, `sum_dim`, `cat`, `pow`, `exp`, `log`, `relu`,
`sigmoid`, `tanh`, `neg`.

Anything not on that list either has no backward or an unverified one —
gradient-check before relying on it (`examples/grad_probe.rs` shows how).

---

## Neural Networks (`ferrum_nn`)

### `Module` trait

```rust
pub trait Module {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
    fn train(&mut self);        // default no-op
    fn eval(&mut self);         // default no-op
    fn num_parameters(&self) -> usize;  // provided
    fn name(&self) -> &str;
}
```

### `Linear` (verified trainable)

```rust
let fc = Linear::new(in_features, out_features);
let fc = Linear::with_bias(in_features, out_features, false);
let y = fc.forward(&x)?;  // [batch, in] -> [batch, out]
```

Weight is stored `[in_features, out_features]` (input-major) so forward
multiplies without a transpose — this is load-bearing for correct gradients
(fixed July 2026; before that `Linear` received transposed weight gradients
and could not train).

### Other layers (exist; trainability undemonstrated)

Conv1d/Conv2d, MaxPool2d/AvgPool2d/AdaptiveAvgPool2d, RNN/LSTM/GRU (cells and
stacks), MultiHeadAttention, TransformerEncoderLayer, PositionalEncoding,
Embedding, LayerNorm, BatchNorm1d/2d, Dropout, Softmax/LogSoftmax and other
activation modules, Sequential/ModuleList/ModuleDict containers.

They construct and forward with correct shapes (unit-tested), but end-to-end
training through them is not verified — several depend on ops without
verified backwards. Roadmap Phase 4.

### Losses

Loss modules exist in `ferrum_nn`, but the verified pattern in the examples
builds losses from differentiable primitives, e.g. MSE:

```rust
let loss = pred.sub(&target)?.pow(2.0)?.mean()?;
```

and cross-entropy from logits (see `examples/cifar10.rs` for the
numerically-stable version):

```rust
// log_softmax(z) = (z - max) - log(sum(exp(z - max)))
// loss = mean over batch of -log_softmax(z)[target]
```

---

## Optimizers (`ferrum_optim`)

### `Optimizer` trait

```rust
optimizer.zero_grad();
optimizer.step()?;
optimizer.learning_rate();
optimizer.param_groups();
```

### SGD (verified)

```rust
let mut opt = SGD::new(params, SGDConfig::new(0.5));
// SGDConfig also carries momentum / weight_decay / nesterov
```

### Adam (verified)

```rust
let mut opt = Adam::new(params, AdamConfig::default());
// AdamConfig { lr, beta1, beta2, eps, weight_decay }
// defaults: 1e-3, 0.9, 0.999, 1e-8, 0.0
```

Both update parameter storage **in place** — the tensors you passed in (and
any clones, which share storage) see the updated values. Don't copy
parameters back from `param_groups()`; it's unnecessary.

### Schedulers / AMP (exist)

Eight LR schedulers and a gradient scaler exist with unit-tested math, but
haven't been used in a verified training run.

---

## Data (`ferrum_data`) — exists

`Dataset`, `TensorDataset`, `DataLoader` (batching/shuffling), samplers,
transforms. Unit-tested; the CIFAR-10 example currently uses its own minimal
loader instead.

## Distributed (`ferrum_distributed`) — exists

`ProcessGroup`, collectives, `DistributedDataParallel` scaffolding. Local /
simulated only; no verified multi-process run.

## Serialization (`ferrum_serialize`) — exists, untested

Safetensors-style save/load. **Zero test coverage** — verify round-trips
yourself before trusting it with real weights.

---

## Raw CUDA layer (`ferrum_cuda`) — verified via benchmarks

Not reachable from the Tensor API yet; used directly by `examples/bench`:

- CUDA Driver API via `dlopen("libcuda.so.1")`: contexts, memory, copies.
- `gemm_ex(...)`: `cublasGemmEx` wrapper. FP16 currently pinned to algo `99`
  + compute type `74` — measured fastest (272 TFLOPS on A100; the
  non-deprecated default measures 230). See `ferrum-cuda/src/cublas.rs`.
- NVRTC runtime compilation of fused element-wise kernels.
- GPU memory pool.

A100 benchmark results are tabulated in
[../IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md).

---

## Examples Index

| Example | Command | Status |
|---|---|---|
| XOR training | `cargo run --release --example train_xor_autograd` | Verified: 4/4 accuracy |
| CIFAR-10 MLP | `cargo run --release --example cifar10` | Verified training stack on real data (CPU) |
| Gradient probe | `cargo run --release --example grad_probe` | Diagnostic for autograd patterns |
| GPU benchmarks | `cargo run --release --example bench` | Verified on A100 |
| MNIST | `cargo run --release --example mnist` | Forward-only API demo (no training) |
