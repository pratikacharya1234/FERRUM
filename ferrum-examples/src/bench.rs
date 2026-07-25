//! FERRUM Performance Benchmark Harness.
//!
//! Uses ferrum_cuda APIs directly (CudaBuffer + cuBLAS + PTX kernels)
//! instead of ferrum-core Tensor, which dereferences device pointers as CPU memory.
//!
//! Run on a machine with an NVIDIA GPU:
//!
//! ```bash
//! cargo build --example bench --release
//! FERRUM_PROFILE=1 cargo run --example bench --release
//! ```

use std::sync::Arc;
use std::time::Instant;

use ferrum_cuda::cuda_device::{CudaDevice, init_cuda, get_device_count};
use ferrum_cuda::cuda_memory::CudaBuffer;
use ferrum_cuda::cublas::{CublasHandle};
use ferrum_cuda::kernel_launch::{launch_scalar, launch_binary, launch_unary};

fn sync(device: &Arc<CudaDevice>) {
    let _ = device.synchronize();
}

/// Fill a CudaBuffer with random f32 data (generate on CPU, upload to GPU).
fn fill_random_f32(buf: &mut CudaBuffer, n: usize) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..n).map(|_| rng.gen()).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    buf.copy_from_host(&bytes).unwrap();
}

/// Fill a CudaBuffer with random f16 data (generate on CPU, upload to GPU).
fn fill_random_f16(buf: &mut CudaBuffer, n: usize) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let data: Vec<u16> = (0..n).map(|_| {
        let val: f32 = rng.gen();
        half::f16::from_f32(val).to_bits()
    }).collect();
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    buf.copy_from_host(&bytes).unwrap();
}

fn bench_matmul(device: &Arc<CudaDevice>) {
    println!("\n=== Matmul TF32 [4096 x 4096] ===");
    let m = 4096i32;
    let k = 4096i32;
    let n = 4096i32;

    println!("  [1] Allocating A buffer ({} MB)...", (m * k * 4) / 1024 / 1024);
    let mut a = CudaBuffer::new(device.clone(), (m * k * 4) as usize).unwrap();
    println!("  [1] OK, ptr=0x{:x}", a.ptr());

    println!("  [2] Allocating B buffer...");
    let mut b = CudaBuffer::new(device.clone(), (k * n * 4) as usize).unwrap();
    println!("  [2] OK, ptr=0x{:x}", b.ptr());

    println!("  [3] Allocating C buffer...");
    let c = CudaBuffer::new(device.clone(), (m * n * 4) as usize).unwrap();
    println!("  [3] OK, ptr=0x{:x}", c.ptr());

    println!("  [4] Filling A with random data...");
    fill_random_f32(&mut a, (m * k) as usize);
    println!("  [4] OK");

    println!("  [5] Filling B with random data...");
    fill_random_f32(&mut b, (k * n) as usize);
    println!("  [5] OK");

    println!("  [6] Creating cuBLAS handle...");
    let handle = CublasHandle::new(device.clone()).unwrap();
    println!("  [6] OK, handle={:?}", handle.handle());

    println!("  [7] Enabling TF32...");
    let _ = handle.enable_tf32();
    println!("  [7] OK");

    println!("  [8] Warmup matmul (10 iterations)...");
    for i in 0..10 {
        handle.sgemm(
            false, false,
            m, n, k,
            1.0,
            a.ptr(), k,
            b.ptr(), n,
            0.0,
            c.ptr(), n,
        ).unwrap();
        println!("  [8] warmup iter {} OK", i);
    }
    sync(device);
    println!("  [8] Warmup complete");

    println!("  [9] Timed matmul (100 iterations)...");
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        handle.sgemm(
            false, false,
            m, n, k,
            1.0,
            a.ptr(), k,
            b.ptr(), n,
            0.0,
            c.ptr(), n,
        ).unwrap();
    }
    sync(device);
    let elapsed = start.elapsed().as_secs_f64();

    let flops = 2.0 * 4096.0_f64.powi(3) * iters as f64;
    let tflops = flops / elapsed / 1e12;
    let ms_per_iter = elapsed * 1000.0 / iters as f64;
    println!("  Time:    {:.2} ms/iter", ms_per_iter);
    println!("  Throughput: {:.1} TFLOPS", tflops);
    println!("  Target:  150+ TFLOPS (PyTorch A100 TF32)");
}

fn bench_matmul_fp16(device: &Arc<CudaDevice>) {
    println!("\n=== Matmul FP16 [4096 x 4096] ===");
    let m = 4096i32;
    let k = 4096i32;
    let n = 4096i32;

    let mut a = CudaBuffer::new(device.clone(), (m * k * 2) as usize).unwrap();
    let mut b = CudaBuffer::new(device.clone(), (k * n * 2) as usize).unwrap();
    let c = CudaBuffer::new(device.clone(), (m * n * 2) as usize).unwrap();

    fill_random_f16(&mut a, (m * k) as usize);
    fill_random_f16(&mut b, (k * n) as usize);

    let handle = CublasHandle::new(device.clone()).unwrap();

    // gemm_f16 = cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_16F and
    // CUBLAS_GEMM_DEFAULT_TENSOR_OP (the fast Tensor Core path).
    let alpha = 1.0f32;
    let beta = 0.0f32;

    for _ in 0..10 {
        handle.gemm_f16(
            false, false,
            m, n, k,
            alpha,
            a.ptr(), k,
            b.ptr(), n,
            beta,
            c.ptr(), n,
        ).unwrap();
    }
    sync(device);

    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        handle.gemm_f16(
            false, false,
            m, n, k,
            alpha,
            a.ptr(), k,
            b.ptr(), n,
            beta,
            c.ptr(), n,
        ).unwrap();
    }
    sync(device);
    let elapsed = start.elapsed().as_secs_f64();

    let flops = 2.0 * 4096.0_f64.powi(3) * iters as f64;
    let tflops = flops / elapsed / 1e12;
    let ms_per_iter = elapsed * 1000.0 / iters as f64;
    println!("  Time:    {:.2} ms/iter", ms_per_iter);
    println!("  Throughput: {:.1} TFLOPS", tflops);
    println!("  Target:  300+ TFLOPS (PyTorch A100 FP16)");
}

fn bench_elementwise_chain(device: &Arc<CudaDevice>) {
    println!("\n=== Element-wise chain [1M elements] ===");
    let n = 1_000_000u32;

    let mut x = CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap();
    let mut y = CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap();
    let tmp = CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap();
    let out = CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap();

    fill_random_f32(&mut x, n as usize);
    fill_random_f32(&mut y, n as usize);

    // Warmup: x + 1.0 -> tmp; tmp * 2.0 -> out; then add y
    for _ in 0..100 {
        launch_scalar(x.ptr(), tmp.ptr(), 0, 1.0, n).unwrap();
        launch_scalar(tmp.ptr(), out.ptr(), 2, 2.0, n).unwrap();
        launch_unary(out.ptr(), tmp.ptr(), 5, n).unwrap();
        launch_binary(tmp.ptr(), y.ptr(), out.ptr(), 0, n).unwrap();
    }
    sync(device);

    let iters = 1000;
    let start = Instant::now();
    for _ in 0..iters {
        launch_scalar(x.ptr(), tmp.ptr(), 0, 1.0, n).unwrap();
        launch_scalar(tmp.ptr(), out.ptr(), 2, 2.0, n).unwrap();
        launch_unary(out.ptr(), tmp.ptr(), 5, n).unwrap();
        launch_binary(tmp.ptr(), y.ptr(), out.ptr(), 0, n).unwrap();
    }
    sync(device);
    let elapsed = start.elapsed().as_secs_f64();

    let us_per_iter = elapsed * 1e6 / iters as f64;
    let gb_s = (1_000_000.0 * 4.0 * 6.0) / elapsed / 1e9;
    println!("  Time:    {:.1} us/iter", us_per_iter);
    println!("  Bandwidth: {:.1} GB/s (estimated)", gb_s);
    println!("  Target:  <50 us (fused kernel)");
}

fn bench_memory_bandwidth(device: &Arc<CudaDevice>) {
    println!("\n=== Memory Bandwidth [alloc + fill 256 MB] ===");
    let n = 64_000_000u32;

    for _ in 0..10 {
        let _ = CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap();
    }
    sync(device);

    let iters = 100;
    let start = Instant::now();
    let mut bufs = Vec::with_capacity(iters);
    for _ in 0..iters {
        bufs.push(CudaBuffer::new(device.clone(), (n * 4) as usize).unwrap());
    }
    sync(device);
    let elapsed = start.elapsed().as_secs_f64();
    drop(bufs);

    let bytes = n as f64 * 4.0 * iters as f64;
    let gb_s = bytes / elapsed / 1e9;
    println!("  Time:    {:.2} ms/iter", elapsed * 1000.0 / iters as f64);
    println!("  Bandwidth: {:.1} GB/s (alloc+fill)", gb_s);
    println!("  Target:  2000+ GB/s (A100 HBM2e)");
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              FERRUM Performance Benchmark Suite              ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");

    println!("[init] Calling init_cuda()...");
    if init_cuda().is_err() {
        println!("[init] CUDA not available. Benchmarks require an NVIDIA GPU.");
        return;
    }
    println!("[init] init_cuda() OK");

    let device_count = get_device_count().unwrap_or(0);
    println!("[init] CUDA devices: {}", device_count);

    println!("[init] Creating CudaDevice(0)...");
    let device = Arc::new(CudaDevice::new(0).expect("Failed to create CudaDevice"));
    println!("[init] GPU: {}", device.name());
    println!("[init] Device set_current()...");
    device.set_current().expect("Failed to set CUDA context");
    println!("[init] Context OK. Starting benchmarks...");

    bench_matmul(&device);
    bench_matmul_fp16(&device);
    bench_elementwise_chain(&device);
    bench_memory_bandwidth(&device);

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Done. Compare numbers against PyTorch on same hardware.");
}
