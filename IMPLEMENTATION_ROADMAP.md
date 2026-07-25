# FERRUM Implementation Roadmap

**Status**: Experimental
**Last Updated**: July 24, 2026

Companion to [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), which
defines what "done" claims below actually mean (verified vs merely existing).

---

## Done and Verified

### CPU training stack
- [x] N-dimensional tensor system with broadcasting and views (CPU)
- [x] DTypes: F32, F64, I32, I64, F16, Bool
- [x] Tape-based autograd: `add/sub/mul/div` (broadcast-aware), `matmul`,
      `sum/mean/sum_dim`, `cat`, `pow/exp/log`, `relu/sigmoid/tanh/neg`
- [x] Gradient accumulation, `no_grad`, gradient checking
- [x] SGD (momentum/Nesterov/weight decay) and Adam, updating parameters
      in place through shared storage
- [x] End-to-end training: XOR at 100% accuracy; CIFAR-10 MLP example
      (`examples/cifar10.rs`) on real data
- [x] `Linear` transposed-gradient bug found and fixed (July 2026)
- [x] Gradient sanity probe (`examples/grad_probe.rs`)

### Raw CUDA layer (`ferrum-cuda`, all via dlopen — no compile-time toolkit)
- [x] CUDA Driver API: init, contexts, alloc, memcpy, memset, sync
- [x] cuBLAS `cublasGemmEx`: TF32 144–162 TFLOPS, FP16 272 TFLOPS on A100
- [x] NVRTC-compiled fused element-wise kernels (~30 µs for a 1M-element chain)
- [x] GPU memory pool (~2850 GB/s alloc+fill on A100)
- [x] Benchmark suite (`examples/bench`) with PyTorch reference comparisons

### Infrastructure that exists (unit-tested, not battle-tested)
- [x] Layers: Conv1d/2d, pooling, RNN/LSTM/GRU cells and stacks,
      MultiHeadAttention, Transformer encoder, LayerNorm/BatchNorm, Embedding
      — **forward only; trainability undemonstrated**
- [x] LR schedulers (8), grad scaler / AMP scaffolding
- [x] DataLoader, samplers, transforms
- [x] ProcessGroup / DDP scaffolding (local collectives)
- [x] Safetensors-style serialization — **zero tests**

---

## Phase 1: Connect the two halves — GPU op dispatch — **DONE, A100-verified 2026-07-24**

Implemented via a backend registration hook (`ferrum_core::gpu::GpuBackend`
trait + `ferrum_cuda::register_gpu_backend()`), avoiding the
`ferrum-core` → `ferrum-cuda` dependency cycle.

- [x] Dispatch `Tensor::matmul` to cuBLAS for `Device::Cuda` F32 tensors
      (cached handle, TF32; transposed views materialized via strided-copy
      kernel)
- [x] Dispatch element-wise ops to NVRTC kernels (strided broadcast binary,
      unary, scalar; in-place optimizer updates)
- [x] Reductions (`sum`, `mean`, `sum_dim`) on GPU (global + axis kernels)
- [x] Autograd backward on GPU tensors — verified gradient parity with CPU
- [x] Real `to_device` (HtoD/DtoH; it previously only relabeled the device
      field), DtoH `item()`/`to_vec()`, host-generate+upload constructors
- [x] `examples/gpu_smoke.rs` correctness gate: **28/28 on A100** (also
      caught and led to the fix of a double-context bug in `CudaDevice::new`)
- [x] GPU CIFAR-10 training verified (2026-07-25): loss curve matches CPU
      (1.93 → 1.39; test acc 39.1% vs 39.8%), **0.2 s/epoch vs 33 s CPU
      (~165×)**. Found+fixed en route: `--use_fast_math` powf NaN on
      negative bases (Adam grad²)

Remaining polish (not blockers): F16/F64 dtypes on GPU, multi-GPU context
handling, fusing the loss chain to cut small-kernel launch overhead.

## Phase 2: `cublasLt` FP16 matmul

`cublasGemmEx` measures 272 TFLOPS on A100 4096³; PyTorch's ~312 comes from
`cublasLt` (workspace + heuristic algorithm search). Estimated 2–3 h of
careful FFI.

- [ ] dlopen `libcublasLt.so`, matmul descriptors, workspace alloc
- [ ] `cublasLtMatmulAlgoGetHeuristic` + algo cache keyed by shape/dtype
- [ ] Fall back to `cublasGemmEx` when unavailable

## Phase 3: CPU matmul that isn't naive

- [ ] Wire `ferrum-ops` tiled/parallel/BLAS matmul into `Tensor::matmul`
- [ ] SIMD element-wise ops where it matters

## Phase 4: Make the remaining layers trainable

- [ ] Backward coverage (with `grad_probe`-style checks) for conv, pooling,
      recurrent, attention — including the ops they need (`softmax` backward,
      `transpose` as a tracked op, `expand` tracking or removal)
- [ ] CIFAR-10 with a small CNN as the acceptance test

## Phase 5: Trust the periphery

- [ ] Serialization round-trip tests (currently zero)
- [ ] Real multi-process distributed run (then NCCL)
- [ ] ONNX export, TensorBoard logging

---

## Explicit non-claims

To keep this document honest:

- No "production ready" claim until Phases 1–4 are done and a non-toy model
  has trained on GPU with verified accuracy.
- Version timeline tables removed — dates were fiction. Phases land when
  they land, in the order above.
