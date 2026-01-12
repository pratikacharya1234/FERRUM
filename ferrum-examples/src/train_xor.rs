//! Training example: XOR problem with manual gradient computation.
//!
//! This demonstrates training a neural network on the XOR problem
//! using manual gradient computation. Full autograd integration is
//! still being developed.

use ferrum::prelude::*;

pub fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║               FERRUM XOR Training Example                      ║");
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

    // Create network weights manually for direct gradient computation
    // Network: 2 -> 8 -> 1 with tanh activation
    let mut w1 = Tensor::randn([2, 8], DType::F32, Device::Cpu).mul_scalar(0.5)?;
    let mut b1 = Tensor::zeros([8], DType::F32, Device::Cpu);
    let mut w2 = Tensor::randn([8, 1], DType::F32, Device::Cpu).mul_scalar(0.5)?;
    let mut b2 = Tensor::zeros([1], DType::F32, Device::Cpu);

    println!("Network architecture: 2 -> 8 (tanh) -> 1 (sigmoid)");
    println!("  W1: {:?}", w1.shape());
    println!("  b1: {:?}", b1.shape());
    println!("  W2: {:?}", w2.shape());
    println!("  b2: {:?}", b2.shape());
    println!();

    // Training parameters
    let learning_rate = 1.0;
    let num_epochs = 2000;
    let print_every = 200;

    println!("Training configuration:");
    println!("  Learning rate: {}", learning_rate);
    println!("  Epochs: {}", num_epochs);
    println!();

    println!("Starting training with manual gradient computation...");
    println!("─────────────────────────────────────────────────────────────");

    for epoch in 1..=num_epochs {
        // Forward pass
        // z1 = x @ w1 + b1
        let z1 = x.matmul(&w1)?.add(&b1.unsqueeze(0)?.expand([4, 8])?)?;
        // a1 = tanh(z1)
        let a1 = z1.tanh()?;
        // z2 = a1 @ w2 + b2
        let z2 = a1.matmul(&w2)?.add(&b2.unsqueeze(0)?.expand([4, 1])?)?;
        // a2 = sigmoid(z2)
        let a2 = z2.sigmoid()?;
        
        // Compute MSE loss
        let diff = a2.sub(&y)?;
        let loss = diff.pow(2.0)?.mean()?;
        let loss_val = loss.item()?;

        if epoch % print_every == 0 || epoch == 1 {
            println!("Epoch {:4} | Loss: {:.6}", epoch, loss_val);
        }

        // Manual backward pass
        // d_loss/d_a2 = 2 * (a2 - y) / n
        let n = 4.0;
        let d_a2 = diff.mul_scalar(2.0 / n)?;
        
        // d_a2/d_z2 = a2 * (1 - a2) for sigmoid
        let sig_grad = a2.mul(&a2.neg()?.add_scalar(1.0)?)?;
        let d_z2 = d_a2.mul(&sig_grad)?;
        
        // d_z2/d_w2 = a1.T @ d_z2
        let d_w2 = a1.t()?.matmul(&d_z2)?;
        // d_z2/d_b2 = sum(d_z2, axis=0)
        let d_b2_vals = d_z2.to_vec::<f32>()?;
        let d_b2_sum: f32 = d_b2_vals.iter().sum();
        let d_b2 = Tensor::from_slice(&[d_b2_sum], [1], Device::Cpu)?;
        
        // d_z2/d_a1 = d_z2 @ w2.T
        let d_a1 = d_z2.matmul(&w2.t()?)?;
        
        // d_a1/d_z1 = 1 - tanh^2(z1) for tanh
        let tanh_grad = a1.pow(2.0)?.neg()?.add_scalar(1.0)?;
        let d_z1 = d_a1.mul(&tanh_grad)?;
        
        // d_z1/d_w1 = x.T @ d_z1
        let d_w1 = x.t()?.matmul(&d_z1)?;
        // d_z1/d_b1 = sum(d_z1, axis=0)
        let d_b1_data = d_z1.to_vec::<f32>()?;
        let mut d_b1_arr = vec![0.0f32; 8];
        for i in 0..4 {
            for j in 0..8 {
                d_b1_arr[j] += d_b1_data[i * 8 + j];
            }
        }
        let d_b1 = Tensor::from_slice(&d_b1_arr, [8], Device::Cpu)?;

        // Update weights (gradient descent)
        w1 = w1.sub(&d_w1.mul_scalar(learning_rate)?)?;
        b1 = b1.sub(&d_b1.mul_scalar(learning_rate)?)?;
        w2 = w2.sub(&d_w2.mul_scalar(learning_rate)?)?;
        b2 = b2.sub(&d_b2.mul_scalar(learning_rate)?)?;
    }

    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Evaluate final predictions
    println!("Final predictions:");
    let z1 = x.matmul(&w1)?.add(&b1.unsqueeze(0)?.expand([4, 8])?)?;
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
