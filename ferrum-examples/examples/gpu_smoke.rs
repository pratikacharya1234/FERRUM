//! GPU dispatch correctness smoke test.
//!
//! Runs every op the CUDA dispatch path implements on both CPU and GPU
//! with identical inputs and compares the results. Run this on a GPU
//! machine BEFORE trusting GPU training output:
//!
//! ```sh
//! cargo run --release --example gpu_smoke
//! ```
//!
//! Exits non-zero on any mismatch.

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;

const TOL: f32 = 1e-3;

struct Harness {
    failures: usize,
    passes: usize,
}

impl Harness {
    fn compare(&mut self, name: &str, cpu: &Tensor, gpu: &Tensor) {
        let c = cpu.to_vec::<f32>().expect("cpu to_vec");
        let g = match gpu.to_vec::<f32>() {
            Ok(g) => g,
            Err(e) => {
                println!("FAIL {name}: GPU read failed: {e}");
                self.failures += 1;
                return;
            }
        };
        if c.len() != g.len() {
            println!("FAIL {name}: length {} vs {}", c.len(), g.len());
            self.failures += 1;
            return;
        }
        let max_err = c
            .iter()
            .zip(&g)
            .map(|(a, b)| {
                let scale = a.abs().max(1.0);
                (a - b).abs() / scale
            })
            .fold(0.0f32, f32::max);
        if max_err > TOL {
            println!("FAIL {name}: max relative error {max_err}");
            println!("  cpu: {:?}", &c[..c.len().min(8)]);
            println!("  gpu: {:?}", &g[..g.len().min(8)]);
            self.failures += 1;
        } else {
            println!("ok   {name} (max err {max_err:.2e})");
            self.passes += 1;
        }
    }
}

fn main() -> Result<()> {
    if ferrum_cuda::CudaDevice::new(0).is_err() {
        println!("No CUDA device found — nothing to test. Run this on a GPU machine.");
        return Ok(());
    }
    ferrum_cuda::backend::register_gpu_backend();
    let gpu = Device::Cuda(0);
    let mut h = Harness {
        failures: 0,
        passes: 0,
    };

    // Fixed inputs on CPU (deterministic values, no RNG dependency).
    let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.37 - 2.0).collect();
    let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * -0.21 + 1.5).collect();
    let a = Tensor::from_slice(&a_data, [4, 3], Device::Cpu)?;
    let b = Tensor::from_slice(&b_data, [4, 3], Device::Cpu)?;
    let ag = a.to_device(gpu)?;
    let bg = b.to_device(gpu)?;

    // Round-trip: upload + download only.
    h.compare("to_device round-trip", &a, &ag);

    // Binary ops, same shape.
    h.compare("add", &a.add(&b)?, &ag.add(&bg)?);
    h.compare("sub", &a.sub(&b)?, &ag.sub(&bg)?);
    h.compare("mul", &a.mul(&b)?, &ag.mul(&bg)?);
    h.compare("div", &a.div(&b)?, &ag.div(&bg)?);

    // Broadcast: [4,3] + [3] and [4,3] - [4,1].
    let bias = Tensor::from_slice(&[0.5f32, -1.0, 2.0], [3], Device::Cpu)?;
    let bias_g = bias.to_device(gpu)?;
    h.compare("add broadcast [3]", &a.add(&bias)?, &ag.add(&bias_g)?);
    let col = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], [4, 1], Device::Cpu)?;
    let col_g = col.to_device(gpu)?;
    h.compare("sub broadcast [4,1]", &a.sub(&col)?, &ag.sub(&col_g)?);

    // Unary ops (shift input for log/sqrt domains).
    let pos = a.mul(&a)?.add_scalar(0.1)?;
    let pos_g = ag.mul(&ag)?.add_scalar(0.1)?;
    h.compare("relu", &a.relu()?, &ag.relu()?);
    h.compare("exp", &a.exp()?, &ag.exp()?);
    h.compare("log", &pos.log()?, &pos_g.log()?);
    h.compare("sqrt", &pos.sqrt()?, &pos_g.sqrt()?);
    h.compare("sigmoid", &a.sigmoid()?, &ag.sigmoid()?);
    h.compare("tanh", &a.tanh()?, &ag.tanh()?);
    h.compare("neg", &a.neg()?, &ag.neg()?);

    // Scalar ops.
    h.compare("add_scalar", &a.add_scalar(3.5)?, &ag.add_scalar(3.5)?);
    h.compare("mul_scalar", &a.mul_scalar(-0.7)?, &ag.mul_scalar(-0.7)?);
    h.compare("pow", &pos.pow(1.7)?, &pos_g.pow(1.7)?);
    // pow(2.0) on NEGATIVE values — the Adam grad² path. Fast-math powf
    // is NaN for negative bases; this guards against that regressing.
    h.compare("pow(2.0) mixed-sign", &a.pow(2.0)?, &ag.pow(2.0)?);
    // Full Adam-style denominator on mixed-sign "gradients":
    // sqrt(g^2 * c) + eps, then divide.
    {
        let adam_cpu = a.pow(2.0)?.mul_scalar(0.999)?.sqrt()?.add_scalar(1e-8)?;
        let adam_gpu = ag.pow(2.0)?.mul_scalar(0.999)?.sqrt()?.add_scalar(1e-8)?;
        h.compare("adam denom mixed-sign", &adam_cpu, &adam_gpu);
        h.compare("adam update", &a.div(&adam_cpu)?, &ag.div(&adam_gpu)?);
    }

    // Reductions.
    h.compare("sum", &a.sum()?, &ag.sum()?);
    h.compare("mean", &a.mean()?, &ag.mean()?);
    h.compare("sum_dim keepdim", &a.sum_dim(1, true)?, &ag.sum_dim(1, true)?);
    h.compare("sum_dim squeeze", &a.sum_dim(0, false)?, &ag.sum_dim(0, false)?);

    // Matmul, plain and through transposed views (the backward path).
    let w_data: Vec<f32> = (0..6).map(|i| (i as f32) * 0.11 - 0.3).collect();
    let w = Tensor::from_slice(&w_data, [3, 2], Device::Cpu)?;
    let wg = w.to_device(gpu)?;
    h.compare("matmul", &a.matmul(&w)?, &ag.matmul(&wg)?);
    h.compare(
        "matmul via a.t()",
        &a.t()?.matmul(&a)?,
        &ag.t()?.matmul(&ag)?,
    );
    h.compare(
        "matmul via w.t()",
        &w.matmul(&w.t()?)?,
        &wg.matmul(&wg.t()?)?,
    );

    // contiguous() on a transposed view.
    h.compare("contiguous of t()", &a.t()?.contiguous()?, &ag.t()?.contiguous()?);

    // expand (stride-0 view) consumed by an op.
    let s = Tensor::from_slice(&[2.5f32], [1], Device::Cpu)?;
    let sg = s.to_device(gpu)?;
    h.compare(
        "expand + mul",
        &s.expand([4usize, 3])?.mul(&a)?,
        &sg.expand([4usize, 3])?.mul(&ag)?,
    );

    // In-place update (optimizer path).
    let p = Tensor::from_slice(&a_data, [4, 3], Device::Cpu)?;
    let pg = p.to_device(gpu)?;
    p.sub_inplace(&b)?;
    pg.sub_inplace(&bg)?;
    h.compare("sub_inplace", &p, &pg);

    // End-to-end: one autograd step must produce identical gradients.
    let x = Tensor::from_slice(&a_data, [4, 3], Device::Cpu)?;
    let wt = Tensor::from_slice(&w_data, [3, 2], Device::Cpu)?.with_requires_grad(true);
    let xg = x.to_device(gpu)?;
    let wtg = Tensor::from_slice(&w_data, [3, 2], gpu)?.with_requires_grad(true);

    GradientTape::with_tape(|_t| -> Result<()> {
        let loss = x.matmul(&wt)?.relu()?.pow(2.0)?.mean()?;
        loss.backward()?;
        Ok(())
    })?;
    GradientTape::with_tape(|_t| -> Result<()> {
        let loss = xg.matmul(&wtg)?.relu()?.pow(2.0)?.mean()?;
        loss.backward()?;
        Ok(())
    })?;
    match (wt.grad(), wtg.grad()) {
        (Some(gc), Some(gg)) => h.compare("autograd gradient", &gc, &gg),
        (c, g) => {
            println!(
                "FAIL autograd gradient: cpu grad present: {}, gpu grad present: {}",
                c.is_some(),
                g.is_some()
            );
            h.failures += 1;
        }
    }

    // ── CIFAR-sized section ─────────────────────────────────────────
    // The 28 checks above use tiny shapes. Training runs [128,3072]
    // matmuls and the full cross-entropy chain; replicate both here and
    // compare every intermediate so a divergence names the exact op.

    // Large matmul at training shape. TF32 accumulates error with k, so
    // use a looser tolerance and compare against CPU f32.
    let big_k = 3072usize;
    let big_n = 64usize;
    let batch = 128usize;
    let xb: Vec<f32> = (0..batch * big_k)
        .map(|i| ((i % 977) as f32) * 0.001 - 0.45)
        .collect();
    let wb: Vec<f32> = (0..big_k * big_n)
        .map(|i| ((i % 641) as f32) * 0.0007 - 0.21)
        .collect();
    let xc = Tensor::from_slice(&xb, [batch, big_k], Device::Cpu)?;
    let wc = Tensor::from_slice(&wb, [big_k, big_n], Device::Cpu)?;
    let xg = xc.to_device(gpu)?;
    let wg2 = wc.to_device(gpu)?;
    {
        let cpu_out = xc.matmul(&wc)?;
        let gpu_out = xg.matmul(&wg2)?;
        let c = cpu_out.to_vec::<f32>()?;
        let g = gpu_out.to_vec::<f32>()?;
        let max_err = c
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs() / a.abs().max(1.0))
            .fold(0.0f32, f32::max);
        let nan = g.iter().filter(|v| !v.is_finite()).count();
        if nan > 0 || max_err > 5e-2 {
            println!("FAIL matmul [128,{big_k}]@[{big_k},{big_n}]: max rel err {max_err}, non-finite: {nan}");
            println!("  cpu: {:?}", &c[..6]);
            println!("  gpu: {:?}", &g[..6]);
            h.failures += 1;
        } else {
            println!("ok   matmul [128,{big_k}]@[{big_k},{big_n}] (max rel err {max_err:.2e}, TF32)");
            h.passes += 1;
        }
    }

    // Exact cross-entropy chain at [batch, 10], step by step.
    let classes = 10usize;
    let logits_data: Vec<f32> = (0..batch * classes)
        .map(|i| ((i % 37) as f32) * 0.11 - 2.0)
        .collect();
    let labels: Vec<u8> = (0..batch).map(|i| (i % classes) as u8).collect();
    let lc = Tensor::from_slice(&logits_data, [batch, classes], Device::Cpu)?;
    let lg = lc.to_device(gpu)?;

    let mut row_max = Vec::with_capacity(batch);
    for row in logits_data.chunks_exact(classes) {
        row_max.push(row.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }
    let mut onehot_v = vec![0.0f32; batch * classes];
    for (i, &l) in labels.iter().enumerate() {
        onehot_v[i * classes + l as usize] = 1.0;
    }

    let ce = |logits: &Tensor, dev: Device| -> Result<Vec<Tensor>> {
        let shift = Tensor::from_slice(&row_max, [batch, 1], dev)?;
        let onehot = Tensor::from_slice(&onehot_v, [batch, classes], dev)?;
        let shifted = logits.sub(&shift)?;
        let e = shifted.exp()?;
        let denom = e.sum_dim(1, true)?;
        let log_denom = denom.log()?;
        let logp = shifted.sub(&log_denom)?;
        let per_sample = logp.mul(&onehot)?.sum_dim(1, false)?;
        let loss = per_sample.mean()?.neg()?;
        Ok(vec![shifted, e, denom, log_denom, logp, per_sample, loss])
    };
    let cpu_steps = ce(&lc, Device::Cpu)?;
    let gpu_steps = ce(&lg, gpu)?;
    let names = [
        "ce: shifted (sub [B,1])",
        "ce: exp",
        "ce: sum_dim keepdim [B,1]",
        "ce: log",
        "ce: logp (sub [B,1])",
        "ce: per-sample sum_dim",
        "ce: loss (mean+neg)",
    ];
    for ((name, c), g) in names.iter().zip(&cpu_steps).zip(&gpu_steps) {
        h.compare(name, c, g);
    }

    // Full training step at real scale: x[128,3072] @ w[3072,10] -> CE
    // -> backward. Compares loss and the weight gradient CPU vs GPU.
    {
        let wt_data: Vec<f32> = (0..big_k * classes)
            .map(|i| ((i % 383) as f32) * 0.0009 - 0.17)
            .collect();
        let step = |dev: Device| -> Result<(Tensor, Tensor)> {
            let x = Tensor::from_slice(&xb, [batch, big_k], dev)?;
            let w = Tensor::from_slice(&wt_data, [big_k, classes], dev)?
                .with_requires_grad(true);
            let onehot = Tensor::from_slice(&onehot_v, [batch, classes], dev)?;
            let mut loss_out = None;
            GradientTape::with_tape(|_t| -> Result<()> {
                let logits = x.matmul(&w)?;
                // Row max on host, like the real example does.
                let ld = logits.to_vec::<f32>()?;
                let mut rm = Vec::with_capacity(batch);
                for row in ld.chunks_exact(classes) {
                    rm.push(row.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
                }
                let shift = Tensor::from_slice(&rm, [batch, 1], dev)?;
                let shifted = logits.sub(&shift)?;
                let log_denom = shifted.exp()?.sum_dim(1, true)?.log()?;
                let logp = shifted.sub(&log_denom)?;
                let loss = logp.mul(&onehot)?.sum_dim(1, false)?.mean()?.neg()?;
                loss.backward()?;
                loss_out = Some(loss);
                Ok(())
            })?;
            let grad = w.grad().expect("weight gradient after backward");
            Ok((loss_out.unwrap(), grad))
        };
        let (loss_c, grad_c) = step(Device::Cpu)?;
        let (loss_g, grad_g) = step(gpu)?;
        println!(
            "training-step loss: cpu {:.6} vs gpu {:.6}",
            loss_c.item()?,
            loss_g.item()?
        );
        h.compare("training step: loss", &loss_c, &loss_g);
        h.compare("training step: weight grad", &grad_c, &grad_g);
    }

    println!();
    println!("{} passed, {} failed", h.passes, h.failures);
    if h.failures > 0 {
        std::process::exit(1);
    }
    Ok(())
}
