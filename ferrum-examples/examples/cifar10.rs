//! CIFAR-10 classification: end-to-end training with real data.
//!
//! Proves the full training stack works together: data loading, forward
//! pass through `ferrum_nn` layers, autograd backward, and Adam updates.
//! Success criterion: loss decreases and test accuracy climbs well above
//! the 10% random baseline.
//!
//! Expects the CIFAR-10 binary dataset. Download once with:
//! ```sh
//! curl -L -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
//! tar xzf cifar-10-binary.tar.gz -C data
//! ```
//! and point `CIFAR10_DIR` at the extracted `cifar-10-batches-bin`
//! directory (default: `./data/cifar-10-batches-bin`).
//!
//! Environment knobs (all optional):
//! - `CIFAR10_DIR`: dataset directory
//! - `CIFAR10_TRAIN`: training samples to use (default 5000)
//! - `CIFAR10_TEST`: test samples to use (default 1000)
//! - `CIFAR10_EPOCHS`: epochs (default 3)
//! - `CIFAR10_DEVICE`: `cpu` or `cuda` (default: auto-detect — cuda when
//!   an NVIDIA GPU is present, else cpu)

use std::io::Read;
use std::path::PathBuf;

use ferrum::prelude::*;
use ferrum_autograd::tape::GradientTape;
use ferrum_nn::{Linear, Module};
use ferrum_optim::{Adam, AdamConfig, Optimizer};

const IMAGE_BYTES: usize = 3072; // 32 * 32 * 3
const NUM_CLASSES: usize = 10;

struct Cifar10 {
    images: Vec<f32>, // [n, 3072], normalized
    labels: Vec<u8>,
}

impl Cifar10 {
    fn len(&self) -> usize {
        self.labels.len()
    }
}

/// Read CIFAR-10 binary files: each record is 1 label byte + 3072 pixel bytes.
fn load_cifar(dir: &PathBuf, files: &[&str], limit: usize) -> Result<Cifar10> {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for file in files {
        if labels.len() >= limit {
            break;
        }
        let path = dir.join(file);
        let mut bytes = Vec::new();
        std::fs::File::open(&path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| FerrumError::InternalError {
                message: format!(
                    "Cannot read {}: {e}. Download the CIFAR-10 binary dataset \
                     (see header of this example) and set CIFAR10_DIR.",
                    path.display()
                ),
            })?;

        for record in bytes.chunks_exact(1 + IMAGE_BYTES) {
            if labels.len() >= limit {
                break;
            }
            labels.push(record[0]);
            // Normalize to roughly zero mean, unit range.
            images.extend(record[1..].iter().map(|&p| p as f32 / 255.0 - 0.5));
        }
    }

    Ok(Cifar10 { images, labels })
}

struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new() -> Self {
        Self {
            fc1: Linear::new(IMAGE_BYTES, 256),
            fc2: Linear::new(256, NUM_CLASSES),
        }
    }

    /// Move all parameters to a device (before the optimizer is created).
    fn to_device(&mut self, device: Device) -> Result<()> {
        self.fc1.to_device(device)?;
        self.fc2.to_device(device)
    }
}

impl Module for Mlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(input)?.relu()?;
        self.fc2.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn name(&self) -> &str {
        "Cifar10Mlp"
    }
}

/// Differentiable cross-entropy from logits.
///
/// Built from primitive ops (`sub`, `exp`, `sum_dim`, `log`, `mul`, `mean`)
/// because those are the ops with verified backward implementations.
/// Logits are shifted by their row max (as a constant) for numerical
/// stability; the shift is gradient-transparent.
fn cross_entropy(logits: &Tensor, labels: &[u8]) -> Result<Tensor> {
    let batch = labels.len();
    let logits_data = logits.to_vec::<f32>()?;

    let mut row_max = Vec::with_capacity(batch);
    for row in logits_data.chunks_exact(NUM_CLASSES) {
        row_max.push(row.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    }
    let shift = Tensor::from_slice(&row_max, [batch, 1], logits.device())?;

    let mut onehot = vec![0.0f32; batch * NUM_CLASSES];
    for (i, &label) in labels.iter().enumerate() {
        onehot[i * NUM_CLASSES + label as usize] = 1.0;
    }
    let onehot = Tensor::from_slice(&onehot, [batch, NUM_CLASSES], logits.device())?;

    // log_softmax(z) = (z - max) - log(sum(exp(z - max)))
    let shifted = logits.sub(&shift)?;
    let log_denom = shifted.exp()?.sum_dim(1, true)?.log()?;
    let log_probs = shifted.sub(&log_denom)?;

    // Mean over batch of the negative target log-probability.
    log_probs.mul(&onehot)?.sum_dim(1, false)?.mean()?.neg()
}

fn accuracy(logits: &Tensor, labels: &[u8]) -> Result<f32> {
    let data = logits.to_vec::<f32>()?;
    let correct = data
        .chunks_exact(NUM_CLASSES)
        .zip(labels)
        .filter(|(row, &label)| {
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            argmax == label as usize
        })
        .count();
    Ok(correct as f32 / labels.len() as f32)
}

fn batch_tensor(data: &Cifar10, indices: &[usize], device: Device) -> Result<(Tensor, Vec<u8>)> {
    let mut x = Vec::with_capacity(indices.len() * IMAGE_BYTES);
    let mut y = Vec::with_capacity(indices.len());
    for &i in indices {
        x.extend_from_slice(&data.images[i * IMAGE_BYTES..(i + 1) * IMAGE_BYTES]);
        y.push(data.labels[i]);
    }
    let x = Tensor::from_slice(&x, [indices.len(), IMAGE_BYTES], device)?;
    Ok((x, y))
}

/// Pick the training device: env override, else CUDA when a GPU is present.
fn select_device() -> Device {
    match std::env::var("CIFAR10_DEVICE").as_deref() {
        Ok("cpu") => return Device::Cpu,
        Ok("cuda") => {
            ferrum_cuda::backend::register_gpu_backend();
            return Device::Cuda(0);
        }
        _ => {}
    }
    if ferrum_cuda::CudaDevice::new(0).is_ok() {
        ferrum_cuda::backend::register_gpu_backend();
        Device::Cuda(0)
    } else {
        Device::Cpu
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var("CIFAR10_DIR").unwrap_or_else(|_| "data/cifar-10-batches-bin".into()),
    );
    let n_train = env_usize("CIFAR10_TRAIN", 5000);
    let n_test = env_usize("CIFAR10_TEST", 1000);
    let epochs = env_usize("CIFAR10_EPOCHS", 3);
    let batch_size = 128;

    let device = select_device();

    println!("FERRUM CIFAR-10 Training");
    println!("========================");
    println!("Device: {device}");

    let train_files = [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin",
    ];
    let train = load_cifar(&dir, &train_files, n_train)?;
    let test = load_cifar(&dir, &["test_batch.bin"], n_test)?;
    println!("Train: {} samples, Test: {} samples", train.len(), test.len());

    let mut model = Mlp::new();
    model.to_device(device)?;
    println!("Model: 3072 -> 256 (relu) -> 10, {} parameters", model.num_parameters());

    let mut optimizer = Adam::new(model.parameters(), AdamConfig::default());

    // Deterministic LCG index shuffle, reseeded per epoch.
    let mut rng_state: u64 = 42;
    let mut next_rand = move || {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 33) as usize
    };

    for epoch in 1..=epochs {
        let mut indices: Vec<usize> = (0..train.len()).collect();
        for i in (1..indices.len()).rev() {
            indices.swap(i, next_rand() % (i + 1));
        }

        let mut epoch_loss = 0.0f64;
        let mut epoch_acc = 0.0f64;
        let num_batches = train.len() / batch_size;
        let start = std::time::Instant::now();

        for (batch_idx, chunk) in indices.chunks_exact(batch_size).enumerate() {
            let (x, y) = batch_tensor(&train, chunk, device)?;

            let mut batch_loss = 0.0f32;
            let mut batch_acc = 0.0f32;
            GradientTape::with_tape(|_tape| -> Result<()> {
                optimizer.zero_grad();
                let logits = model.forward(&x)?;
                let loss = cross_entropy(&logits, &y)?;
                batch_loss = loss.item()? as f32;
                batch_acc = accuracy(&logits, &y)?;
                loss.backward()?;
                Ok(())
            })?;
            optimizer.step()?;

            epoch_loss += batch_loss as f64;
            epoch_acc += batch_acc as f64;

            if (batch_idx + 1) % 10 == 0 {
                println!(
                    "  epoch {epoch} batch {}/{}: loss {batch_loss:.4}, acc {:.1}%",
                    batch_idx + 1,
                    num_batches,
                    batch_acc * 100.0
                );
            }
        }

        // Evaluate in test batches (keeps peak memory low).
        let mut test_acc = 0.0f64;
        let mut test_batches = 0;
        let test_indices: Vec<usize> = (0..test.len()).collect();
        for chunk in test_indices.chunks_exact(batch_size) {
            let (x, y) = batch_tensor(&test, chunk, device)?;
            let logits = model.forward(&x)?;
            test_acc += accuracy(&logits, &y)? as f64;
            test_batches += 1;
        }

        println!(
            "Epoch {epoch}/{epochs}: train loss {:.4}, train acc {:.1}%, test acc {:.1}% ({:.1}s)",
            epoch_loss / num_batches as f64,
            epoch_acc / num_batches as f64 * 100.0,
            test_acc / test_batches as f64 * 100.0,
            start.elapsed().as_secs_f64()
        );
    }

    println!();
    println!("Random baseline is 10%. If test accuracy is well above that and");
    println!("loss decreased across epochs, the training stack works end to end.");
    Ok(())
}
