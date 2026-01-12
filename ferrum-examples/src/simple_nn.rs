//! Simple neural network example.
//!
//! Demonstrates basic usage of FERRUM for training a small network.

use ferrum::prelude::*;

pub fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              FERRUM Simple Neural Network Example             ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Create a simple model
    let model = Sequential::new()
        .add(Linear::new(2, 4))
        .add(ReLU::new())
        .add(Linear::new(4, 1));

    println!("Model: {:?}", model);
    println!("Total parameters: {}", model.num_parameters());
    println!();

    // Create some dummy data (XOR problem)
    let x = Tensor::from_slice(
        &[0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [4, 2],
        Device::Cpu,
    )?;
    let y = Tensor::from_slice(&[0.0f32, 1.0, 1.0, 0.0], [4, 1], Device::Cpu)?;

    println!("Input shape: {:?}", x.shape());
    println!("Target shape: {:?}", y.shape());
    println!();

    // Forward pass
    let output = model.forward(&x)?;
    println!("Output: {}", output);

    // Compute loss (MSE)
    let diff = output.sub(&y)?;
    let loss = diff.pow(2.0)?.mean()?;
    println!("Initial MSE Loss: {:.6}", loss.item()?);

    println!();
    println!("✓ Example completed successfully!");

    Ok(())
}
