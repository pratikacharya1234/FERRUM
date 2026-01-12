//! Training example: XOR problem with automatic differentiation.
//!
//! This demonstrates training a neural network on the XOR problem
//! using automatic backpropagation via GradientTape.

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;
use ferrum_optim::{Optimizer, SGDConfig, SGD};

pub fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║          FERRUM XOR Training with Autograd Example            ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // XOR dataset
    let x_data = vec![
        0.0f32, 0.0, // [0, 0] -> 0
        0.0, 1.0, // [0, 1] -> 1
        1.0, 0.0, // [1, 0] -> 1
        1.0, 1.0, // [1, 1] -> 0
    ];
    let y_data = vec![0.0f32, 1.0, 1.0, 0.0];

    let x = Tensor::from_slice(&x_data, [4, 2], Device::Cpu)?;
    let y = Tensor::from_slice(&y_data, [4, 1], Device::Cpu)?;

    println!("Dataset:");
    println!("  Input shape:  {:?}", x.shape());
    println!("  Target shape: {:?}", y.shape());
    println!();

    // Create network weights with gradient tracking
    // Network: 2 -> 4 -> 1 (smaller network)
    let mut w1 = Tensor::randn([2, 4], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?
        .with_requires_grad(true);
    let mut b1 = Tensor::zeros([4], DType::F32, Device::Cpu)
        .with_requires_grad(true);
    let mut w2 = Tensor::randn([4, 1], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?
        .with_requires_grad(true);
    let mut b2 = Tensor::zeros([1], DType::F32, Device::Cpu)
        .with_requires_grad(true);

    println!("Network architecture: 2 -> 4 (tanh) -> 1 (sigmoid)");
    println!("  W1: {:?}", w1.shape());
    println!("  b1: {:?}", b1.shape());
    println!("  W2: {:?}", w2.shape());
    println!("  b2: {:?}", b2.shape());
    println!();

    // Create optimizer
    let mut optimizer = SGD::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()],
        SGDConfig::new(0.5),
    );

    // Training parameters
    let num_epochs = 1000;
    let print_every = 100;

    println!("Training configuration:");
    println!("  Learning rate: {}", optimizer.learning_rate());
    println!("  Epochs: {}", num_epochs);
    println!();

    println!("Starting training with automatic differentiation...");
    println!("─────────────────────────────────────────────────────────────");

    for epoch in 1..=num_epochs {
        GradientTape::with_tape(|_tape| {
            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            let z1 = x.matmul(&w1)?.add(&b1.unsqueeze(0)?.expand([4, 4])?)?;
            let a1 = z1.tanh()?;
            let z2 = a1.matmul(&w2)?.add(&b2.unsqueeze(0)?.expand([4, 1])?)?;
            let a2 = z2.sigmoid()?;

            // Compute MSE loss
            let diff = a2.sub(&y)?;
            let loss = diff.pow(2.0)?.mean()?;
            let loss_val = loss.item()?;

            if epoch % print_every == 0 || epoch == 1 {
                println!("Epoch {:4} | Loss: {:.6}", epoch, loss_val);
            }

            // Backward pass (automatic!)
            loss.backward()?;

            Ok(())
        })?;

        // Update weights
        optimizer.step()?;

        // Get updated weights for next iteration
        let params = optimizer.param_groups()[0].clone();
        w1 = params[0].clone();
        b1 = params[1].clone();
        w2 = params[2].clone();
        b2 = params[3].clone();
    }

    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Evaluate final predictions
    println!("Final predictions:");
    let z1 = x.matmul(&w1)?.add(&b1.unsqueeze(0)?.expand([4, 4])?)?;
    let a1 = z1.tanh()?;
    let z2 = a1.matmul(&w2)?.add(&b2.unsqueeze(0)?.expand([4, 1])?)?;
    let final_output = z2.sigmoid()?;
    let predictions = final_output.to_vec::<f32>()?;

    let mut correct = 0;
    for i in 0..4 {
        let input = [x_data[i * 2], x_data[i * 2 + 1]];
        let target = y_data[i];
        let pred = predictions[i];
        let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
        let is_correct = (rounded - target).abs() < 0.1;
        if is_correct { correct += 1; }
        let check = if is_correct { "✓" } else { "✗" };
        println!(
            "  [{:.0}, {:.0}] -> Target: {:.0}, Predicted: {:.4} ({}) {}",
            input[0], input[1], target, pred, rounded, check
        );
    }

    println!();
    println!("Accuracy: {}/4 ({:.1}%)", correct, correct as f32 * 25.0);
    println!();
    println!("✓ Training completed successfully!");

    Ok(())
}
