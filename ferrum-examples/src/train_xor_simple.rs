//! Training example: XOR problem with automatic differentiation (simplified).
//!
//! This demonstrates training a neural network on the XOR problem
//! using automatic backpropagation via GradientTape with manual weight updates.

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;

pub fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       FERRUM XOR Training with Autograd (Simple) Example      ║");
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
    // Network: 2 -> 4 -> 1
    let mut w1 = Tensor::randn([2, 4], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?;
    let mut b1 = Tensor::zeros([1, 4], DType::F32, Device::Cpu);  // Shape [1, 4] for broadcasting
    let mut w2 = Tensor::randn([4, 1], DType::F32, Device::Cpu)
        .mul_scalar(0.5)?;
    let mut b2 = Tensor::zeros([1, 1], DType::F32, Device::Cpu);  // Shape [1, 1] for broadcasting

    println!("Network architecture: 2 -> 4 (tanh) -> 1 (sigmoid)");
    println!("  W1: {:?}", w1.shape());
    println!("  b1: {:?}", b1.shape());
    println!("  W2: {:?}", w2.shape());
    println!("  b2: {:?}", b2.shape());
    println!();

    // Training parameters
    let learning_rate = 0.5;
    let num_epochs = 2000;
    let print_every = 200;

    println!("Training configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", num_epochs);
    println!();

    println!("Starting training with automatic differentiation...");
    println!("─────────────────────────────────────────────────────────────");

    for epoch in 1..=num_epochs {
        let loss_val = GradientTape::with_tape(|_tape| {
            // Set requires_grad for this iteration
            w1.set_requires_grad(true);
            b1.set_requires_grad(true);
            w2.set_requires_grad(true);
            b2.set_requires_grad(true);

            // Zero gradients
            w1.zero_grad();
            b1.zero_grad();
            w2.zero_grad();
            b2.zero_grad();

            // Forward pass
            let z1 = x.matmul(&w1)?.add(&b1)?;
            let a1 = z1.tanh()?;
            let z2 = a1.matmul(&w2)?.add(&b2)?;
            let a2 = z2.sigmoid()?;

            // Compute MSE loss
            let diff = a2.sub(&y)?;
            let loss = diff.pow(2.0)?.mean()?;
            let loss_val = loss.item()? as f32;

            // Backward pass (automatic!)
            loss.backward()?;

            Ok(loss_val)
        })?;

        if epoch % print_every == 0 || epoch == 1 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, loss_val);
        }

        // Update weights manually (SGD)
        if let Some(grad_w1) = w1.grad() {
            w1 = w1.sub(&grad_w1.mul_scalar(learning_rate)?)?;
        }
        if let Some(grad_b1) = b1.grad() {
            b1 = b1.sub(&grad_b1.mul_scalar(learning_rate)?)?;
        }
        if let Some(grad_w2) = w2.grad() {
            w2 = w2.sub(&grad_w2.mul_scalar(learning_rate)?)?;
        }
        if let Some(grad_b2) = b2.grad() {
            b2 = b2.sub(&grad_b2.mul_scalar(learning_rate)?)?;
        }
    }

    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Evaluate final predictions
    println!("Final predictions:");
    let z1 = x.matmul(&w1)?.add(&b1)?;
    let a1 = z1.tanh()?;
    let z2 = a1.matmul(&w2)?.add(&b2)?;
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
