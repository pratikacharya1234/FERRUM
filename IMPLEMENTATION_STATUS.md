# FERRUM Implementation Status

**Status**: Experimental — working CPU training stack + benchmarked raw CUDA layer, not yet connected
**Last Updated**: July 24, 2026

This document states what is *verified*, what merely *exists*, and what is
*known broken*. "Verified" means we ran it and looked at the output, not that
code compiles.

---

## Executive Summary

FERRUM has two working halves:

1. **CPU training stack** (`ferrum-core` + `ferrum-autograd` + `ferrum-nn` +
   `ferrum-optim`): trains real models end to end. Verified: XOR at 100%
   accuracy, and CIFAR-10 (real data, 5000-image subset, MLP + Adam, 3 epochs):
   train loss 1.93 → 1.40, test accuracy 34.8% → 39.8% vs 10% random baseline.
2. **Raw CUDA layer** (`ferrum-cuda`): CUDA Driver API + cuBLAS + NVRTC loaded
   via `dlopen` at runtime. Verified on an A100: 144–162 TFLOPS TF32,
   272 TFLOPS FP16, ~30 µs fused element-wise chains, ~2850 GB/s alloc+fill.

**The two halves are connected and end-to-end GPU training is verified**
(2026-07-25): `Tensor` ops on `Device::Cuda` dispatch to cuBLAS + NVRTC
kernels through a registered backend (`ferrum_cuda::register_gpu_backend()`),
confirmed correct on an A100 by `examples/gpu_smoke.rs` (41/41 checks incl.
autograd backward and Adam-path math). **CIFAR-10 trains on GPU with a loss
curve numerically equivalent to CPU (1.93 → 1.39 over 3 epochs, test acc
39.1% vs 39.8% CPU) at 0.2 s/epoch vs 33 s on CPU — ~165× faster.** F32
only; unimplemented ops fail loudly instead of silently computing on CPU.

---

## Test Summary (measured 2026-07-24, `cargo test --workspace --release`)

**217 tests, 0 failures.** Per-crate (unit + integration + doc tests):

| Crate | Tests | Notes |
|---|---|---|
| `ferrum-core` | 60 | Tensor/shape/dtype/device/storage + 7 doc tests |
| `ferrum-cuda` | 43 | FFI structure tests; GPU-dependent paths need real hardware |
| `ferrum-nn` | 29 | Layer construction + forward shapes |
| `ferrum-autograd` | 27 | Tape, backward, gradient checks (incl. 13 integration) |
| `ferrum-data` | 20 | Datasets, samplers, transforms |
| `ferrum-optim` | 19 | SGD/Adam/scheduler math |
| `ferrum-distributed` | 15 | Local/simulated collectives |
| `ferrum` (facade) | 2 | |
| `ferrum-ops` | 2 | |
| `ferrum-serialize` | **0** | **No test coverage — treat save/load as unverified** |
| `ferrum-examples` | 0 | Examples are run manually, not tested |

What the tests do **not** cover: GPU execution correctness (CI has no GPU),
backward passes for conv/recurrent/attention layers, serialization round-trips,
and multi-process distributed runs.

---

## Verified GPU Performance (A100-SXM4-40GB, Colab, CUDA 12)

Via `cargo run --release --example bench` — exercises `ferrum-cuda` directly.

| Benchmark | Result (range over runs) | Reference (PyTorch, same GPU) |
|---|---|---|
| Matmul TF32 4096³ (cuBLAS) | 144–165 TFLOPS | ~156 TFLOPS |
| Matmul FP16 4096³ (`cublasGemmEx`) | 268–272 TFLOPS | ~312 TFLOPS |
| Fused element-wise chain, 1M elements (NVRTC) | 23–30 µs | ~30 µs |
| Alloc + fill 256 MB (memory pool) | 2850–3000 GB/s | ~2000 GB/s |

FP16 details (measured, not guessed):
- `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (99) + `CUBLAS_COMPUTE_32F_FAST_16F` (74):
  **272 TFLOPS** — current configuration.
- `CUBLAS_GEMM_DEFAULT` (−1), same compute type: **230 TFLOPS** — the
  non-deprecated constant is *slower* here; that's why the deprecated one is
  kept, with a comment, in `ferrum-cuda/src/cublas.rs`.
- The remaining ~13% gap to PyTorch is the `cublasGemmEx` vs `cublasLt`
  difference (workspace + heuristic algorithm search), not a tuning issue.

---

## Module Status

Statuses: **Verified** (ran it, checked output) · **Exists** (implemented +
unit-tested, end-to-end unproven) · **Broken/Missing** (known not to work).

| Module | Status | Detail |
|---|---|---|
| `ferrum-core` tensors (CPU) | Verified | Full op set on CPU; naive matmul (slow, correct) |
| `ferrum-core` CUDA storage | Verified | alloc / zero-fill / HtoD / DtoH via dlopen |
| `ferrum-core` CUDA compute | **Verified** (A100, 28/28 smoke) | Ops dispatch to a registered `GpuBackend` (cuBLAS + NVRTC kernels, F32); loud errors otherwise. Includes autograd backward on GPU |
| `ferrum-autograd` | Verified | For the primitive op set listed below |
| `ferrum-cuda` cuBLAS/NVRTC | Verified | Benchmarked on A100 (numbers above) |
| `ferrum-nn` Linear + activations | Verified | Trains end to end (XOR, CIFAR-10 example) |
| `ferrum-nn` Conv/RNN/LSTM/GRU/Transformer | Exists | Forward + shape tests only; trainability undemonstrated |
| `ferrum-optim` SGD/Adam | Verified | In-place updates through shared storage confirmed |
| `ferrum-optim` schedulers/AMP | Exists | Unit-tested math; not used in a real training run |
| `ferrum-data` | Exists | Unit-tested; CIFAR-10 example uses its own loader |
| `ferrum-distributed` | Exists | Simulated/local collectives; no real multi-node run |
| `ferrum-serialize` | Exists | **Zero tests** |

### Autograd: ops with verified backward passes

`add`, `sub`, `mul`, `div` (all with broadcast gradient reduction to each
operand's shape), `matmul`, `sum`, `mean`, `sum_dim`, `cat`, `pow`, `exp`,
`log`, `relu`, `sigmoid`, `tanh`, `neg`.

Verified empirically by `examples/grad_probe.rs`, which checks gradient
existence and shape for the patterns used in training. Run it after touching
autograd or layer code:

```bash
cargo run --release --example grad_probe
```

---

## Known Issues (current, honest)

1. **GPU op dispatch and GPU training: VERIFIED on A100 (2026-07-25).**
   `examples/gpu_smoke.rs` passes **41/41** on an A100-SXM4-40GB (CUDA
   12.8): every dispatched op (matmul incl. transposed views and
   [128,3072] training shapes, broadcast binary, unary, scalar, reductions,
   `contiguous`, expand views, in-place updates), the full cross-entropy
   chain intermediate-by-intermediate, a complete training step with
   gradient parity vs CPU, and the Adam denominator on mixed-sign values.
   CIFAR-10 GPU training matches the CPU loss curve at ~165× speed.
   Two real bugs were found and fixed during verification:
   - `CudaDevice::new` created a second CUDA context, invalidating cached
     kernel handles (CUDA error 400) — it now reuses the current context.
   - NVRTC kernels compiled with `--use_fast_math`, whose `powf` is NaN
     for negative bases — Adam's `grad.pow(2.0)` NaN'd the weights on the
     first optimizer step. Fast math is off; `pow(2.0)` is `x*x`.
2. **CPU matmul is a naive triple loop.** `ferrum-ops` has tiled/parallel/BLAS
   variants that are not wired into `Tensor::matmul`.
3. **`expand` is not tracked by autograd.** A pre-expanded operand receives a
   gradient with the expanded shape. Use implicit broadcasting instead —
   `AddBackward`/`SubBackward` reduce gradients correctly. (The XOR example
   used the broken pattern until 2026-07-24; now fixed. A separate `expand`
   bug — strides computed as if the source were contiguous, breaking
   expand-of-a-view and panicking in chained reduction backwards — was also
   fixed 2026-07-24.)
4. **Transposed views share their base tensor's autograd id.** This is why
   `Linear` stores its weight `[in_features, out_features]` and multiplies
   without `t()`. Any new code that does `x.matmul(&w.t()?)` on a parameter
   will silently deposit a transposed gradient on `w`. Fixed in `Linear` on
   2026-07-24 (gradients were transposed before that — `Linear` was
   untrainable).
5. **The MNIST example does not train.** Forward-only on synthetic data; kept
   as an API demo. XOR and CIFAR-10 are the real training examples.
6. **FP16 matmul uses deprecated algo 99 deliberately** (272 vs 230 TFLOPS —
   see benchmark section).
7. **`ferrum-serialize` is untested.**

---

## GPU-First Claim — Current Reality

`Device::default()` does return `Cuda(0)` when a GPU is present. But because
compute ops are CPU-only, defaulting tensors onto the GPU makes them unusable
for training today. **Pass `Device::Cpu` explicitly** until GPU op dispatch
lands. The "GPU-first" design goal stands; the implementation is not there yet.

---

## Roadmap (ordered by impact)

1. **GPU op dispatch**: route `Tensor::matmul` / element-wise ops to
   cuBLAS/NVRTC for `Device::Cuda` tensors. Turns benchmark numbers into
   training speed.
2. **`cublasLt`**: workspace + `cublasLtMatmulAlgoGetHeuristic` to close
   FP16 272 → ~312 TFLOPS.
3. **CPU BLAS matmul** in `Tensor::matmul`.
4. **Backward coverage** for conv/recurrent/attention layers, with gradient
   checks in `grad_probe` style.
5. **Serialization tests**, ONNX, NCCL.

---

## Repository

**GitHub**: https://github.com/pratikacharya1234/FERRUM
**License**: Apache 2.0

For contributions see [CONTRIBUTING.md](CONTRIBUTING.md).
