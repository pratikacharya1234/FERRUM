# FERRUM Quick Start

This guide covers the workflow that is verified to work today: building the
workspace, tensors and autograd on CPU, training real models, and running the
GPU benchmarks. For what "verified" means — and the list of things that do
*not* work yet — read [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
first. The one-line version: **training runs on CPU; the GPU layer is
benchmarked but not yet reachable from the Tensor API.**

---

## 1. Build and Test

```bash
git clone https://github.com/pratikacharya1234/FERRUM.git
cd FERRUM
cargo build --release
cargo test --workspace   # 217 tests, 0 failures as of 2026-07-24
```

No CUDA toolkit is needed to build — CUDA libraries are loaded at runtime via
`dlopen` if present.

---

## 2. Tensors

```rust
use ferrum::prelude::*;

fn main() -> Result<()> {
    // Creation — pass Device::Cpu explicitly (see the Device note below).
    let a = Tensor::zeros([2, 3], DType::F32, Device::Cpu);
    let b = Tensor::randn([2, 3], DType::F32, Device::Cpu);
    let c = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [2, 2], Device::Cpu)?;

    // Element-wise ops (broadcasting supported)
    let sum = a.add(&b)?;
    let prod = b.mul(&b)?;

    // Matrix multiplication
    let x = Tensor::randn([4, 3], DType::F32, Device::Cpu);
    let w = Tensor::randn([3, 2], DType::F32, Device::Cpu);
    let y = x.matmul(&w)?; // [4, 2]

    // Reductions
    let total = y.sum()?;          // scalar
    let mean = y.mean()?;          // scalar
    let rows = y.sum_dim(1, true)?; // [4, 1]

    // Reading values back
    let value: f64 = total.item()?;
    let all: Vec<f32> = y.to_vec()?;

    println!("shape {:?}, mean {}", y.shape(), value);
    Ok(())
}
```

### The Device note (important)

`Device::default()` returns `Cuda(0)` when a GPU is detected. GPU compute
works (verified 28/28 on A100), but needs two things:

1. **Register the backend once at startup** — without it, CUDA tensors
   error loudly on compute:
   ```rust
   ferrum_cuda::register_gpu_backend();
   ```
2. **F32 only** for now — other dtypes on `Device::Cuda` return
   `NotImplemented`.

```rust
ferrum_cuda::register_gpu_backend();
let t = Tensor::randn([8, 8], DType::F32, Device::Cuda(0)); // computes on GPU
let c = Tensor::randn([8, 8], DType::F32, Device::Cpu);     // computes on CPU
```

After changing dispatch or kernel code, re-run the correctness gate on a GPU
machine: `cargo run --release --example gpu_smoke` (must print `28 passed,
0 failed`).

---

## 3. Autograd

FERRUM uses a gradient tape, like PyTorch's dynamic graphs:

```rust
use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;

fn main() -> Result<()> {
    let x = Tensor::randn([4, 3], DType::F32, Device::Cpu);
    let w = Tensor::randn([3, 2], DType::F32, Device::Cpu)
        .with_requires_grad(true);

    GradientTape::with_tape(|_tape| -> Result<()> {
        let y = x.matmul(&w)?;
        let loss = y.pow(2.0)?.mean()?;
        loss.backward()?; // gradients land on w
        Ok(())
    })?;

    let grad = w.grad().expect("gradient exists after backward");
    assert_eq!(grad.shape(), w.shape());
    Ok(())
}
```

### Ops with verified backward passes

`add`, `sub`, `mul`, `div` (broadcast-aware: gradients are reduced back to
each operand's shape), `matmul`, `sum`, `mean`, `sum_dim`, `cat`, `pow`,
`exp`, `log`, `relu`, `sigmoid`, `tanh`, `neg`.

### Two patterns to avoid

1. **Don't pre-expand for broadcasting.** `expand` is not tracked by autograd;
   `bias.unsqueeze(0)?.expand([n, d])?` produces a bias gradient with shape
   `[n, d]` instead of `[d]`. Just write `x.add(&bias)?` — implicit
   broadcasting has a correct backward.
2. **Don't matmul parameters through `t()`.** Transposed views share the base
   tensor's autograd id, so `x.matmul(&w.t()?)` deposits a *transposed*
   gradient on `w`. Store weights in the layout you multiply with.
   (`ferrum_nn::Linear` does this correctly since July 2026.)

`examples/grad_probe.rs` checks these patterns empirically — run it after
changing autograd or layer code:

```bash
cargo run --release --example grad_probe
```

---

## 4. Training a Model: XOR

The complete, verified example (`cargo run --release --example
train_xor_autograd` — trains to 4/4 correct):

```rust
use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;
use ferrum_optim::{Optimizer, SGDConfig, SGD};

fn main() -> Result<()> {
    let x = Tensor::from_slice(
        &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2], Device::Cpu)?;
    let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], [4, 1], Device::Cpu)?;

    // Weights stored [in, out] so forward is x.matmul(&w) — no transpose.
    let w1 = Tensor::randn([2, 4], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?.with_requires_grad(true);
    let b1 = Tensor::zeros([4], DType::F32, Device::Cpu).with_requires_grad(true);
    let w2 = Tensor::randn([4, 1], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?.with_requires_grad(true);
    let b2 = Tensor::zeros([1], DType::F32, Device::Cpu).with_requires_grad(true);

    let mut optimizer = SGD::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        SGDConfig::new(0.5),
    );

    for epoch in 1..=1000 {
        GradientTape::with_tape(|_tape| -> Result<()> {
            optimizer.zero_grad();
            let hidden = x.matmul(&w1)?.add(&b1)?.tanh()?;
            let output = hidden.matmul(&w2)?.add(&b2)?.sigmoid()?;
            let loss = output.sub(&y)?.pow(2.0)?.mean()?;
            if epoch % 100 == 0 {
                println!("epoch {epoch}: loss {}", loss.item()?);
            }
            loss.backward()?;
            Ok(())
        })?;
        optimizer.step()?; // updates the tensors in place (shared storage)
    }
    Ok(())
}
```

Notes:
- `optimizer.step()` mutates parameter storage shared with your tensors — you
  keep using `w1` etc. directly; no copying back from the optimizer.
- `zero_grad()` every iteration, inside the tape, before the forward pass.

---

## 5. Training with `ferrum_nn` Layers: CIFAR-10

`examples/cifar10.rs` is the first end-to-end run on a real dataset: a
3072→256→10 MLP built from `ferrum_nn::Linear`, a differentiable
cross-entropy composed from verified primitives, and Adam.

```bash
curl -L -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
mkdir -p data && tar xzf cifar-10-binary.tar.gz -C data
cargo run --release --example cifar10
```

Environment knobs: `CIFAR10_DIR` (default `data/cifar-10-batches-bin`),
`CIFAR10_TRAIN` (default 5000), `CIFAR10_TEST` (1000), `CIFAR10_EPOCHS` (3).
The subset default exists because CPU matmul is a naive loop — this proves the
training stack is correct, not fast. Verified run (defaults, CPU): train loss
1.93 → 1.40 and test accuracy 34.8% → 39.8% over 3 epochs (~33 s/epoch),
well above the 10% random baseline.

Using layers directly:

```rust
use ferrum::prelude::*;
use ferrum_nn::{Linear, Module};

let fc = Linear::new(3072, 256);      // weight [3072, 256], bias [256]
let out = fc.forward(&input)?;        // input [batch, 3072] -> [batch, 256]
let params = fc.parameters();         // for the optimizer
```

**Verified-trainable layers**: `Linear` plus the activation ops. Conv, LSTM,
GRU, and Transformer layers exist with forward implementations and unit tests,
but their end-to-end trainability is undemonstrated (roadmap Phase 4).

The MNIST example (`--example mnist`) is **forward-only on synthetic data** —
an API demo, not a training proof.

---

## 6. Optimizers

```rust
use ferrum_optim::{Adam, AdamConfig, Optimizer, SGD, SGDConfig};

// SGD
let mut sgd = SGD::new(params.clone(), SGDConfig::new(0.5));

// Adam (defaults: lr 1e-3, betas 0.9/0.999, eps 1e-8)
let mut adam = Adam::new(params, AdamConfig::default());

// Per step:
// zero_grad() inside the tape, then forward/backward, then:
adam.step()?;
```

---

## 7. GPU Benchmarks (requires an NVIDIA GPU)

```bash
cargo run --release --example bench
```

This exercises the raw `ferrum-cuda` layer (cuBLAS, NVRTC fusion, memory
pool). Verified A100 results and their PyTorch references are tabulated in
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md). Remember: these numbers
are not yet reachable through the `Tensor` API.

---

## 8. Troubleshooting

- **Crash or garbage using a `Device::Cuda` tensor in ops** — expected today;
  compute is CPU-only. Keep training tensors on `Device::Cpu`.
- **A gradient has the wrong shape or is missing** — check the two autograd
  anti-patterns in §3, then run `cargo run --release --example grad_probe`.
- **CIFAR-10 example can't find data** — set `CIFAR10_DIR` to the directory
  containing `data_batch_1.bin` … `test_batch.bin`.
- **Training is slow** — CPU matmul is a naive triple loop for now (roadmap
  Phase 3); reduce `CIFAR10_TRAIN` or batch count rather than waiting.

## Further Reading

- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) — verified status, per-crate tests, known issues
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) — phased plan with acceptance tests
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md) — API surface overview
- `cargo doc --workspace --open` — rustdoc for the full API
