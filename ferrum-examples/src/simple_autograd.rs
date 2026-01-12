//! Simple autograd demonstration.
//!
//! This example shows the most basic usage of automatic differentiation
//! in Ferrum using GradientTape.

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;

pub fn main() -> Result<()> {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              FERRUM Simple Autograd Example                   ║");
    println!("╚═══════════════════════════════════════════════════════════════╝");
    println!();

    // Example 1: Simple scalar function f(x) = x^2
    println!("Example 1: f(x) = x^2");
    println!("─────────────────────────────────────────────────");

    GradientTape::with_tape(|_tape| {
        let x = Tensor::from_slice(&[3.0f32], [1], Device::Cpu)?
            .with_requires_grad(true);

        println!("x = {}", x);

        let y = x.pow(2.0)?;  // y = x^2
        println!("y = x^2 = {}", y);

        // Compute gradient: dy/dx = 2x
        y.backward()?;

        if let Some(grad) = x.grad() {
            println!("dy/dx = {}", grad);
            println!("Expected: 6.0 (2 * 3.0)");
        }

        Ok(())
    })?;

    println!();

    // Example 2: Two variables f(x, y) = x * y + x
    println!("Example 2: f(x, y) = x * y + x");
    println!("─────────────────────────────────────────────────");

    GradientTape::with_tape(|_tape| {
        let x = Tensor::from_slice(&[2.0f32], [1], Device::Cpu)?
            .with_requires_grad(true);
        let y = Tensor::from_slice(&[3.0f32], [1], Device::Cpu)?
            .with_requires_grad(true);

        println!("x = {}, y = {}", x, y);

        let z = x.mul(&y)?.add(&x)?;  // z = x*y + x
        println!("z = x*y + x = {}", z);

        // Compute gradients
        z.backward()?;

        if let Some(grad_x) = x.grad() {
            println!("dz/dx = {}", grad_x);
            println!("Expected: 4.0 (y + 1 = 3 + 1)");
        }

        if let Some(grad_y) = y.grad() {
            println!("dz/dy = {}", grad_y);
            println!("Expected: 2.0 (x = 2)");
        }

        Ok(())
    })?;

    println!();
    println!("✓ Autograd examples completed successfully!");

    Ok(())
}
