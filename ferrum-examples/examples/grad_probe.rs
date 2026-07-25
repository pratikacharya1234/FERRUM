//! Probe which autograd patterns propagate correct gradients.
//!
//! Checks, for each pattern, that after backward():
//! - the weight gradient exists and has the weight's shape
//! - the bias gradient exists and has the bias's shape
//! - an optimizer step actually changes the loss

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;
use ferrum_nn::{Linear, Module};
use ferrum_optim::{Optimizer, SGDConfig, SGD};

fn check_grad(name: &str, t: &Tensor) {
    match t.grad() {
        Some(g) => {
            let ok = g.shape() == t.shape();
            println!(
                "  {name}: grad shape {:?} vs param {:?} -> {}",
                g.shape(),
                t.shape(),
                if ok { "OK" } else { "SHAPE MISMATCH" }
            );
        }
        None => println!("  {name}: NO GRADIENT"),
    }
}

fn main() -> Result<()> {
    let x = Tensor::randn([4, 3], DType::F32, Device::Cpu);
    let y = Tensor::randn([4, 2], DType::F32, Device::Cpu);

    // Pattern A: x @ w + broadcast bias (weights stored [in, out])
    println!("Pattern A: x.matmul(w) + add(bias) broadcast");
    {
        let w = Tensor::randn([3, 2], DType::F32, Device::Cpu).with_requires_grad(true);
        let b = Tensor::zeros([2], DType::F32, Device::Cpu).with_requires_grad(true);
        GradientTape::with_tape(|_t| -> Result<()> {
            let out = x.matmul(&w)?.add(&b)?;
            let loss = out.sub(&y)?.pow(2.0)?.mean()?;
            loss.backward()?;
            Ok(())
        })?;
        check_grad("w [3,2]", &w);
        check_grad("b [2]", &b);
    }

    // Pattern B: bias via unsqueeze + expand (XOR example style)
    println!("Pattern B: bias via unsqueeze(0).expand()");
    {
        let w = Tensor::randn([3, 2], DType::F32, Device::Cpu).with_requires_grad(true);
        let b = Tensor::zeros([2], DType::F32, Device::Cpu).with_requires_grad(true);
        GradientTape::with_tape(|_t| -> Result<()> {
            let out = x.matmul(&w)?.add(&b.unsqueeze(0)?.expand([4, 2])?)?;
            let loss = out.sub(&y)?.pow(2.0)?.mean()?;
            loss.backward()?;
            Ok(())
        })?;
        check_grad("w [3,2]", &w);
        check_grad("b [2]", &b);
    }

    // Pattern C: ferrum-nn Linear (x @ w.t() + bias, weight stored [out, in])
    println!("Pattern C: ferrum_nn::Linear");
    {
        let lin = Linear::new(3, 2);
        GradientTape::with_tape(|_t| -> Result<()> {
            let out = lin.forward(&x)?;
            let loss = out.sub(&y)?.pow(2.0)?.mean()?;
            loss.backward()?;
            Ok(())
        })?;
        let params = lin.parameters();
        check_grad("weight [2,3]", &params[0]);
        check_grad("bias [2]", &params[1]);
    }

    // Pattern D: differentiable cross-entropy pieces: exp, sum_dim, div, log, mul, neg
    println!("Pattern D: log-softmax from primitives (exp/sum_dim/div/log)");
    {
        let w = Tensor::randn([3, 2], DType::F32, Device::Cpu).with_requires_grad(true);
        GradientTape::with_tape(|_t| -> Result<()> {
            let logits = x.matmul(&w)?;
            let e = logits.exp()?;
            let denom = e.sum_dim(1, true)?; // [4, 1]
            let probs = e.div(&denom)?; // broadcast [4,2] / [4,1]
            let logp = probs.log()?;
            // one-hot targets
            let onehot = Tensor::from_slice(
                &[1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                [4, 2],
                Device::Cpu,
            )?;
            let loss = logp.mul(&onehot)?.mean()?.neg()?;
            loss.backward()?;
            Ok(())
        })?;
        check_grad("w [3,2]", &w);
    }

    // Pattern E: does SGD.step() mutate the same storage the model sees?
    println!("Pattern E: optimizer step visibility (shared storage?)");
    {
        let w = Tensor::randn([3, 2], DType::F32, Device::Cpu).with_requires_grad(true);
        let before = w.to_vec::<f32>()?;
        let mut opt = SGD::new(vec![w.clone()], SGDConfig::new(0.1));
        GradientTape::with_tape(|_t| -> Result<()> {
            let loss = x.matmul(&w)?.pow(2.0)?.mean()?;
            loss.backward()?;
            Ok(())
        })?;
        opt.step()?;
        let after_local = w.to_vec::<f32>()?;
        let after_opt = opt.param_groups()[0][0].to_vec::<f32>()?;
        let local_changed = before
            .iter()
            .zip(&after_local)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        let opt_changed = before
            .iter()
            .zip(&after_opt)
            .any(|(a, b)| (a - b).abs() > 1e-9);
        println!("  original tensor changed after step: {local_changed}");
        println!("  optimizer's copy changed after step: {opt_changed}");
    }

    Ok(())
}
