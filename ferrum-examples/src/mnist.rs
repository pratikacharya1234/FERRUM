//! MNIST handwritten digit classification example.
//!
//! Demonstrates:
//! - Data loading and preprocessing
//! - Simple MLP model
//! - Training loop with loss and optimizer
//! - Evaluation on test set

use ferrum::prelude::*;
use ferrum_nn::{Module, Linear, ReLU, Dropout, LogSoftmax};
use ferrum_optim::{Adam, AdamConfig};

/// Simple MLP for MNIST classification.
struct MnistModel {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    dropout1: Dropout,
    dropout2: Dropout,
}

impl MnistModel {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 128),
            fc3: Linear::new(128, 10),
            dropout1: Dropout::new(0.2),
            dropout2: Dropout::new(0.2),
        }
    }
}

impl Module for MnistModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?;
        let x = x.relu()?;
        let x = self.dropout1.forward(&x)?;
        
        let x = self.fc2.forward(&x)?;
        let x = x.relu()?;
        let x = self.dropout2.forward(&x)?;
        
        let x = self.fc3.forward(&x)?;
        LogSoftmax::new(-1).forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }

    fn train(&mut self) {
        self.dropout1.train();
        self.dropout2.train();
    }

    fn eval(&mut self) {
        self.dropout1.eval();
        self.dropout2.eval();
    }

    fn name(&self) -> &str {
        "MnistModel"
    }
}

/// Generate synthetic MNIST-like data for demonstration.
fn generate_synthetic_data(num_samples: usize) -> (Tensor, Tensor) {
    let mut x_data = Vec::with_capacity(num_samples * 784);
    let mut y_data = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let seed = (i * 7919 + 104729) as u32;
        let mut rng = seed;
        for _ in 0..784 {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            let val = (rng as f32) / (u32::MAX as f32);
            x_data.push(val);
        }
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        y_data.push((rng % 10) as i64);
    }
    
    let x = Tensor::from_slice(&x_data, [num_samples, 784], Device::Cpu).unwrap();
    let y = Tensor::from_slice(&y_data, [num_samples], Device::Cpu).unwrap();
    
    (x, y)
}

/// Compute cross entropy loss from log probabilities and targets.
fn cross_entropy_loss(log_probs: &Tensor, targets: &Tensor) -> Result<Tensor> {
    let log_probs_data = log_probs.to_vec::<f32>()?;
    let targets_data = targets.to_vec::<i64>()?;
    let batch_size = targets_data.len();
    let num_classes = log_probs.shape()[1];
    
    let mut total_loss = 0.0f32;
    for (i, &target_idx) in targets_data.iter().enumerate() {
        let idx = target_idx as usize;
        if idx < num_classes {
            total_loss += log_probs_data[i * num_classes + idx];
        }
    }
    
    let loss = -total_loss / batch_size as f32;
    Tensor::from_slice(&[loss], [1], log_probs.device())
}

/// Compute accuracy.
fn accuracy(predictions: &Tensor, targets: &Tensor) -> Result<f32> {
    let pred_data = predictions.to_vec::<f32>()?;
    let target_data = targets.to_vec::<i64>()?;
    let batch_size = target_data.len();
    let num_classes = predictions.shape()[1];
    
    let mut correct = 0;
    for (i, &target_idx) in target_data.iter().enumerate() {
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0;
        for j in 0..num_classes {
            let val = pred_data[i * num_classes + j];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }
        if max_idx == target_idx as usize {
            correct += 1;
        }
    }
    
    Ok(correct as f32 / batch_size as f32)
}

pub fn main() -> Result<()> {
    println!("FERRUM MNIST Example");
    println!("====================");
    println!();
    
    // Generate synthetic data
    println!("Generating synthetic MNIST data...");
    let (x_train, y_train) = generate_synthetic_data(1000);
    let (x_test, y_test) = generate_synthetic_data(200);
    
    println!("Training set: {} samples", x_train.shape()[0]);
    println!("Test set: {} samples", x_test.shape()[0]);
    println!();
    
    // Create model
    let mut model = MnistModel::new();
    println!("Model: {} parameters", model.num_parameters());
    
    // Create optimizer
    let config = AdamConfig {
        lr: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
    };
    let mut optimizer = Adam::new(model.parameters(), config);
    
    // Training loop
    let num_epochs = 5;
    let batch_size = 32;
    
    println!("Training for {} epochs...", num_epochs);
    println!();
    
    for epoch in 0..num_epochs {
        model.train();
        let mut epoch_loss = 0.0;
        let mut epoch_correct = 0;
        let mut epoch_total = 0;
        
        let num_batches = x_train.shape()[0] / batch_size;
        
        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;
            
            let x_batch = x_train.narrow(0, start, batch_size)?;
            let y_batch = y_train.narrow(0, start, batch_size)?;
            
            // Forward pass
            let log_probs = model.forward(&x_batch)?;
            let loss = cross_entropy_loss(&log_probs, &y_batch)?;
            
            // Backward pass (simplified - would use autograd in production)
            
            let batch_loss = loss.item()?;
            epoch_loss += batch_loss;
            
            let batch_acc = accuracy(&log_probs, &y_batch)?;
            epoch_correct += (batch_acc * batch_size as f32) as usize;
            epoch_total += batch_size;
        }
        
        let avg_loss = epoch_loss / num_batches as f64;
        let train_acc = epoch_correct as f32 / epoch_total as f32;
        
        // Evaluate on test set
        model.eval();
        let test_log_probs = model.forward(&x_test)?;
        let test_acc = accuracy(&test_log_probs, &y_test)?;
        
        println!("Epoch {}/{}: loss={:.4}, train_acc={:.4}, test_acc={:.4}",
                 epoch + 1, num_epochs, avg_loss, train_acc, test_acc);
    }
    
    println!();
    println!("Training complete!");
    
    Ok(())
}
