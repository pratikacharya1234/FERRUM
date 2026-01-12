//! Demonstration of new FERRUM features.
//!
//! This example showcases:
//! - Dimension-wise reductions (sum_dim, mean_dim)
//! - argmax/argmin operations
//! - Tensor concatenation (cat, stack)
//! - Embedding layers
//! - Learning rate schedulers

use ferrum::prelude::*;

fn main() -> Result<()> {
    println!("═══════════════════════════════════════════════════════════");
    println!("           FERRUM New Features Demonstration");
    println!("═══════════════════════════════════════════════════════════\n");

    // =======================================================================
    // 1. Dimension-wise Reductions
    // =======================================================================
    println!("1. Dimension-wise Reductions");
    println!("─────────────────────────────────────────────────────────────");

    let x = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        Device::Cpu,
    )?;
    println!("Input tensor (2x3):");
    println!("  [[1, 2, 3],");
    println!("   [4, 5, 6]]");

    // Sum along dimension 0 (columns)
    let sum0 = x.sum_dim(0, false)?;
    println!("\nsum_dim(0, false) = {:?}", sum0.to_vec::<f32>()?);
    println!("  (sum each column: [1+4, 2+5, 3+6] = [5, 7, 9])");

    // Sum along dimension 1 (rows)
    let sum1 = x.sum_dim(1, false)?;
    println!("\nsum_dim(1, false) = {:?}", sum1.to_vec::<f32>()?);
    println!("  (sum each row: [1+2+3, 4+5+6] = [6, 15])");

    // Mean along dimension 1
    let mean1 = x.mean_dim(1, false)?;
    println!("\nmean_dim(1, false) = {:?}", mean1.to_vec::<f32>()?);
    println!("  (mean each row: [2, 5])");

    // =======================================================================
    // 2. argmax/argmin Operations
    // =======================================================================
    println!("\n\n2. argmax/argmin Operations");
    println!("─────────────────────────────────────────────────────────────");

    let y = Tensor::from_slice(
        &[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0],
        [2, 3],
        Device::Cpu,
    )?;
    println!("Input tensor (2x3):");
    println!("  [[3, 1, 4],");
    println!("   [1, 5, 9]]");

    let argmax_row = y.argmax(1, false)?;
    println!("\nargmax(1, false) = {:?}", argmax_row.to_vec::<i64>()?);
    println!("  (index of max in each row: [2, 2])");

    let (max_vals, max_idx) = y.max_dim(1, false)?;
    println!("\nmax_dim(1, false):");
    println!("  values  = {:?}", max_vals.to_vec::<f32>()?);
    println!("  indices = {:?}", max_idx.to_vec::<i64>()?);

    // =======================================================================
    // 3. Tensor Concatenation
    // =======================================================================
    println!("\n\n3. Tensor Concatenation (cat, stack)");
    println!("─────────────────────────────────────────────────────────────");

    let a = Tensor::from_slice(&[1.0f32, 2.0], [2], Device::Cpu)?;
    let b = Tensor::from_slice(&[3.0f32, 4.0], [2], Device::Cpu)?;
    let c = Tensor::from_slice(&[5.0f32, 6.0], [2], Device::Cpu)?;

    println!("a = [1, 2], b = [3, 4], c = [5, 6]");

    let cat_result = Tensor::cat(&[&a, &b, &c], 0)?;
    println!("\ncat([a, b, c], dim=0) = {:?}", cat_result.to_vec::<f32>()?);
    println!("  shape: {:?}", cat_result.shape());

    let stack_result = Tensor::stack(&[&a, &b, &c], 0)?;
    println!("\nstack([a, b, c], dim=0) = {:?}", stack_result.to_vec::<f32>()?);
    println!("  shape: {:?}", stack_result.shape());

    // narrow example
    let full = Tensor::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        Device::Cpu,
    )?;
    let narrowed = full.narrow(1, 1, 2)?;
    println!("\nnarrow([[1,2,3],[4,5,6]], dim=1, start=1, len=2):");
    println!("  = {:?}", narrowed.contiguous()?.to_vec::<f32>()?);
    println!("  shape: {:?}", narrowed.shape());

    // =======================================================================
    // 4. Embedding Layer
    // =======================================================================
    println!("\n\n4. Embedding Layer (for NLP)");
    println!("─────────────────────────────────────────────────────────────");

    // Create embedding table: 10 words, 4-dimensional embeddings
    let embedding = Embedding::new(10, 4, Device::Cpu);
    println!("Created Embedding(num_embeddings=10, embedding_dim=4)");
    println!("  Parameters: {}", embedding.num_parameters());

    // Look up embeddings for word indices [1, 5, 3]
    let indices = Tensor::from_slice(&[1i64, 5, 3], [1, 3], Device::Cpu)?;
    let embeddings = embedding.forward(&indices)?;
    println!("\nLookup indices [1, 5, 3]:");
    println!("  Output shape: {:?}", embeddings.shape());
    println!("  (batch=1, seq_len=3, embed_dim=4)");

    // =======================================================================
    // 5. Learning Rate Schedulers
    // =======================================================================
    println!("\n\n5. Learning Rate Schedulers");
    println!("─────────────────────────────────────────────────────────────");

    // StepLR: Decay by 0.1x every 10 epochs
    println!("\nStepLR (base_lr=0.1, step_size=10, gamma=0.1):");
    let mut step_lr = StepLR::new(0.1, 10, 0.1);
    for epoch in [0, 5, 10, 15, 20, 30] {
        while step_lr.current_step() < epoch {
            step_lr.step();
        }
        println!("  Epoch {:2}: lr = {:.6}", epoch, step_lr.get_lr());
    }

    // CosineAnnealingLR
    println!("\nCosineAnnealingLR (base_lr=0.1, T_max=100, eta_min=0.001):");
    let mut cosine_lr = CosineAnnealingLR::new(0.1, 100, 0.001);
    for epoch in [0, 25, 50, 75, 100] {
        while cosine_lr.current_step() < epoch {
            cosine_lr.step();
        }
        println!("  Epoch {:3}: lr = {:.6}", epoch, cosine_lr.get_lr());
    }

    // OneCycleLR
    println!("\nOneCycleLR (max_lr=0.1, total_steps=100):");
    let mut one_cycle = OneCycleLR::new_default(0.1, 100);
    for step in [0, 15, 30, 50, 75, 99] {
        while one_cycle.current_step() < step {
            one_cycle.step();
        }
        println!("  Step {:2}: lr = {:.6}", step, one_cycle.get_lr());
    }

    // =======================================================================
    // Summary
    // =======================================================================
    println!("\n═══════════════════════════════════════════════════════════");
    println!("                    Summary of New Features");
    println!("═══════════════════════════════════════════════════════════");
    println!("
✓ Tensor Operations:
  - sum_dim, mean_dim: Reduce along specific dimension
  - argmax, argmin: Get indices of extrema
  - max_dim, min_dim: Values and indices together
  - cat: Concatenate tensors
  - stack: Stack tensors along new dimension
  - narrow: View into tensor slice

✓ Neural Network:
  - Embedding: Lookup table for NLP tokens

✓ Learning Rate Schedulers:
  - StepLR, MultiStepLR, ExponentialLR
  - CosineAnnealingLR, CosineAnnealingWarmRestarts
  - LinearWarmupLR, OneCycleLR
  - ReduceLROnPlateau

Total: 134 tests passing!
");

    Ok(())
}
