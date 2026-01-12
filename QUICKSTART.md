# FERRUM Quick Start Guide# FERRUM Quick Start Guide# FERRUM Quick Start Guide# FERRUM Quick Start Guide# FERRUM Quick Start Guide# FERRUM Quick Start Guide



Get up and running with FERRUM in 5 minutes.



---Get up and running with FERRUM in 5 minutes.



## Installation



### From Git---Get up and running with FERRUM in 5 minutes.



```toml

[dependencies]

ferrum = { git = "https://github.com/pratikacharya1234/FERRUM" }## Installation

```



### From Source

### From Git---Get up and running with FERRUM in 5 minutes.

```bash

git clone https://github.com/pratikacharya1234/FERRUM.git

cd FERRUM

cargo build --release```toml

cargo test  # Verify 154 tests pass

```[dependencies]



---ferrum = { git = "https://github.com/pratikacharya1234/FERRUM" }## Installation



## Your First Tensor```



```rust

use ferrum::prelude::*;

### From Source

fn main() -> Result<()> {

    // Create tensors### From Git---Get up and running with FERRUM in 5 minutes.Get up and running with FERRUM in 5 minutes.

    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);

    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);```bash



    // Operationsgit clone https://github.com/pratikacharya1234/FERRUM.git

    let z = x.add(&y)?;

    let w = z.mul_scalar(2.0)?;cd FERRUM



    println!("Result shape: {:?}", w.shape());cargo build --release```toml

    println!("Result: {:?}", w.to_vec::<f32>());

cargo test  # Verify 134 tests pass

    Ok(())

}```[dependencies]

```



---

---ferrum = { git = "https://github.com/pratikacharya1234/FERRUM" }## Installation

## XOR Training Example



```rust

use ferrum::prelude::*;## Your First Tensor```



fn main() -> Result<()> {

    // XOR dataset

    let inputs = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 1., 1.], [4, 2], Device::Cpu);```rust

    let targets = Tensor::from_slice(&[0., 1., 1., 0.], [4, 1], Device::Cpu);

use ferrum::prelude::*;

    // Build model: 2 -> 8 -> 1

    let model = Sequential::new()### From Source

        .add(Linear::new(2, 8))

        .add(Tanh::new())fn main() -> Result<()> {

        .add(Linear::new(8, 1))

        .add(Sigmoid::new());    // Create tensors### From Git---## Installation



    // Train    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);

    let optimizer = SGD::new(model.parameters(), SGDConfig { lr: 1.0, ..Default::default() });

    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);```bash

    for epoch in 0..2000 {

        let output = model.forward(&inputs)?;

        let loss = bce_loss(&output, &targets)?;

    // Operationsgit clone https://github.com/pratikacharya1234/FERRUM.git

        backward(&loss)?;

        optimizer.step()?;    let z = x.add(&y)?;

        optimizer.zero_grad();

    let w = z.mul_scalar(2.0)?;cd FERRUM

        if epoch % 500 == 0 {

            println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>());

        }

    }    println!("Result shape: {:?}", w.shape());cargo build --release```toml



    // Test    println!("Result: {:?}", w.to_vec::<f32>());

    let predictions = model.forward(&inputs)?;

    println!("Predictions: {:?}", predictions.to_vec::<f32>());cargo test  # Verify 134 tests pass



    Ok(())    Ok(())

}

```}```[dependencies]



Run it:```

```bash

cargo run --example train_xor

```

---

---

---ferrum = { git = "https://github.com/pratikacharya1234/ferrum" }## InstallationAdd FERRUM to your `Cargo.toml`:

## Neural Network Layers

## XOR Training Example

```rust

use ferrum::prelude::*;



// Sequential model```rust

let model = Sequential::new()

    .add(Linear::new(784, 256))use ferrum::prelude::*;## Your First Tensor```

    .add(ReLU::new())

    .add(LayerNorm::new(vec![256]))

    .add(Linear::new(256, 10))

    .add(Softmax::new(-1));fn main() -> Result<()> {



let output = model.forward(&input)?;    // XOR dataset

```

    let inputs = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 1., 1.], [4, 2], Device::Cpu);```rust

### Available Layers

    let targets = Tensor::from_slice(&[0., 1., 1., 0.], [4, 1], Device::Cpu);

| Layer | Description |

|-------|-------------|use ferrum::prelude::*;

| Linear | Fully connected |

| ReLU, Sigmoid, Tanh | Basic activations |    // Build model: 2 -> 8 -> 1

| GELU, SiLU, ELU | Advanced activations |

| Softmax, LogSoftmax | Probability outputs |    let model = Sequential::new()### From Source

| LayerNorm, BatchNorm1d | Normalization |

| Dropout | Regularization |        .add(Linear::new(2, 8))

| Embedding | Token embeddings (NLP) |

| Conv1d, Conv2d | Convolution |        .add(Tanh::new())fn main() -> Result<()> {



---        .add(Linear::new(8, 1))



## Optimizers        .add(Sigmoid::new());    // Create tensors### From Git```toml



```rust

// SGD with momentum

let optimizer = SGD::new(params, SGDConfig {    // Train    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);

    lr: 0.01,

    momentum: 0.9,    let optimizer = SGD::new(model.parameters(), SGDConfig { lr: 1.0, ..Default::default() });

    weight_decay: 1e-4,

    ..Default::default()    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);```bash

});

    for epoch in 0..2000 {

// Adam

let optimizer = Adam::new(params, AdamConfig {        let output = model.forward(&inputs)?;

    lr: 0.001,

    betas: (0.9, 0.999),        let loss = bce_loss(&output, &targets)?;

    eps: 1e-8,

    weight_decay: 0.0,    // Operationsgit clone https://github.com/pratikacharya1234/ferrum.git[dependencies]

});

        backward(&loss)?;

// Training loop

optimizer.zero_grad();        optimizer.step()?;    let z = x.add(&y)?;

let loss = model.forward(&input)?.loss(&target)?;

backward(&loss)?;        optimizer.zero_grad();

optimizer.step()?;

```    let w = z.mul_scalar(2.0)?;cd ferrum



---        if epoch % 500 == 0 {



## Learning Rate Schedulers            println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>());



```rust        }

use ferrum::prelude::*;

    }    println!("Result shape: {:?}", w.shape());cargo build --release```tomlferrum = { path = "path/to/ferrum" }

// Step decay

let scheduler = StepLR::new(10, 0.1);  // Decay by 0.1 every 10 epochs



// Cosine annealing    // Test    println!("Result: {:?}", w.to_vec::<f32>());

let scheduler = CosineAnnealingLR::new(100, 1e-6);  // T_max=100, eta_min=1e-6

    let predictions = model.forward(&inputs)?;

// Use in training loop

for epoch in 0..100 {    println!("Predictions: {:?}", predictions.to_vec::<f32>());cargo test  # Verify 117 tests pass

    let current_lr = scheduler.get_lr(epoch, initial_lr);

    optimizer.set_lr(current_lr);

    // ... training ...

}    Ok(())    Ok(())

```

}

---

```}```[dependencies]```

## Data Loading



```rust

use ferrum::prelude::*;Run it:```



let dataset = TensorDataset::new(train_inputs, train_targets);```bash



let loader = DataLoader::new(dataset)cargo run --example train_xor

    .batch_size(32)

    .shuffle(true);```



for batch in &loader {---

    let (inputs, targets) = batch;

    // Training code---

}

```---ferrum = { git = "https://github.com/pratikacharya1234/ferrum" }



---## Neural Network Layers



## Loss Functions## XOR Training Example



```rust```rust

let mse = mse_loss(&output, &target)?;       // Regression

let bce = bce_loss(&output, &target)?;       // Binary classificationuse ferrum::prelude::*;

let ce = cross_entropy_loss(&logits, &labels)?;  // Multi-class

```



---// Sequential model```rust



## Save and Load Modelslet model = Sequential::new()



```rust    .add(Linear::new(784, 256))use ferrum::prelude::*;## Your First Tensor```Or clone the repository:

// Save

save_model(&model, "model.bin")?;    .add(ReLU::new())



// Load    .add(LayerNorm::new(vec![256]))

let model = load_model::<MyModel>("model.bin")?;

```    .add(Linear::new(256, 10))



---    .add(Softmax::new(-1));fn main() -> Result<()> {



## Running Tests



```bashlet output = model.forward(&input)?;    // XOR dataset

cargo test --workspace         # All 154 tests

cargo test -p ferrum-core      # Specific crate```

cargo test -- --nocapture      # With output

```    let inputs = Tensor::from_slice(&[0., 0., 0., 1., 1., 0., 1., 1.], [4, 2], Device::Cpu);```rust



---### Available Layers



## Next Steps    let targets = Tensor::from_slice(&[0., 1., 1., 0.], [4, 1], Device::Cpu);



- See [API Reference](docs/API_REFERENCE.md) for complete documentation| Layer | Description |

- See [PyTorch Comparison](docs/PYTORCH_COMPARISON.md) for migration guide

- Check out examples in `ferrum-examples/src/`|-------|-------------|use ferrum::prelude::*;



---| Linear | Fully connected |



## Repository| ReLU, Sigmoid, Tanh | Basic activations |    // Build model: 2 -> 8 -> 1



https://github.com/pratikacharya1234/FERRUM| GELU, SiLU, ELU | Advanced activations |


| Softmax, LogSoftmax | Probability outputs |    let model = Sequential::new()### From Source```bash

| LayerNorm, BatchNorm1d | Normalization |

| Dropout | Regularization |        .add(Linear::new(2, 8))

| Embedding | Token embeddings (NLP) |

| Conv1d, Conv2d | Convolution |        .add(Tanh::new())fn main() -> Result<()> {



---        .add(Linear::new(8, 1))



## Optimizers        .add(Sigmoid::new());    // Create tensorsgit clone https://github.com/your-org/ferrum.git



```rust

// SGD with momentum

let optimizer = SGD::new(params, SGDConfig {    // Train    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);

    lr: 0.01,

    momentum: 0.9,    let optimizer = SGD::new(model.parameters(), SGDConfig { lr: 1.0, ..Default::default() });

    weight_decay: 1e-4,

    ..Default::default()    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);```bashcd ferrum

});

    for epoch in 0..2000 {

// Adam

let optimizer = Adam::new(params, AdamConfig {        let output = model.forward(&inputs)?;

    lr: 0.001,

    betas: (0.9, 0.999),        let loss = bce_loss(&output, &targets)?;

    eps: 1e-8,

    weight_decay: 0.0,    // Operationsgit clone https://github.com/pratikacharya1234/ferrum.gitcargo build --release

});

        backward(&loss)?;

// Training loop

optimizer.zero_grad();        optimizer.step()?;    let z = x.add(&y)?;

let loss = model.forward(&input)?.loss(&target)?;

backward(&loss)?;        optimizer.zero_grad();

optimizer.step()?;

```    let w = z.mul_scalar(2.0)?;cd ferrum```



---        if epoch % 500 == 0 {



## Learning Rate Schedulers            println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>());



```rust        }

use ferrum::prelude::*;

    }    println!("Shape: {:?}", w.shape());cargo build --release

// Step decay

let scheduler = StepLR::new(10, 0.1);  // Decay by 0.1 every 10 epochs



// Cosine annealing    // Test    println!("Data: {:?}", w.to_vec::<f32>()?);

let scheduler = CosineAnnealingLR::new(100, 1e-6);  // T_max=100, eta_min=1e-6

    let predictions = model.forward(&inputs)?;

// Use in training loop

for epoch in 0..100 {    println!("Predictions: {:?}", predictions.to_vec::<f32>());cargo test  # Verify 117 tests pass---

    let current_lr = scheduler.get_lr(epoch, initial_lr);

    optimizer.set_lr(current_lr);

    // ... training ...

}    Ok(())    Ok(())

```

}

---

```}```

## Data Loading



```rust

use ferrum::prelude::*;Run it:```



let dataset = TensorDataset::new(train_inputs, train_targets);```bash



let loader = DataLoader::new(dataset)cargo run --example train_xor## Your First Tensor

    .batch_size(32)

    .shuffle(true);```



for batch in &loader {---

    let (inputs, targets) = batch;

    // Training code---

}

```---



---## Neural Network Layers



## Loss Functions## Your First Neural Network



```rust```rust

let mse = mse_loss(&output, &target)?;       // Regression

let bce = bce_loss(&output, &target)?;       // Binary classificationuse ferrum::prelude::*;```rust

let ce = cross_entropy_loss(&logits, &labels)?;  // Multi-class

```



---// Sequential model```rust



## Save and Load Modelslet model = Sequential::new()



```rust    .add(Linear::new(784, 256))use ferrum::prelude::*;## Your First Tensoruse ferrum::prelude::*;

// Save

save_model(&model, "model.bin")?;    .add(ReLU::new())



// Load    .add(LayerNorm::new(vec![256]))

let model = load_model::<MyModel>("model.bin")?;

```    .add(Linear::new(256, 10))



---    .add(Softmax::new(-1));fn main() -> Result<()> {



## Running Tests



```bashlet output = model.forward(&input)?;    // Build a 3-layer network

cargo test --workspace         # All 134 tests

cargo test -p ferrum-core      # Specific crate```

cargo test -- --nocapture      # With output

```    let model = Sequential::new()```rustfn main() -> Result<()> {



---### Available Layers



## Next Steps        .add(Linear::new(784, 256))



- See [API Reference](docs/API_REFERENCE.md) for complete documentation| Layer | Description |

- See [PyTorch Comparison](docs/PYTORCH_COMPARISON.md) for migration guide

- Check out examples in `ferrum-examples/src/`|-------|-------------|        .add(ReLU::new())use ferrum::prelude::*;    // Create tensors



---| Linear | Fully connected |



## Repository| ReLU, Sigmoid, Tanh | Basic activations |        .add(Linear::new(256, 10));



https://github.com/pratikacharya1234/FERRUM| GELU, SiLU, ELU | Advanced activations |


| Softmax, LogSoftmax | Probability outputs |    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);

| LayerNorm, BatchNorm1d | Normalization |

| Dropout | Regularization |    // Forward pass

| Embedding | Token embeddings (NLP) |

| Conv1d, Conv2d | Convolution |    let input = Tensor::randn([32, 784], DType::F32, Device::Cpu);fn main() -> Result<()> {    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);



---    let output = model.forward(&input)?;



## Optimizers    // Create tensors



```rust    println!("Output shape: {:?}", output.shape()); // [32, 10]

// SGD with momentum

let optimizer = SGD::new(params, SGDConfig {    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);    // Perform operations

    lr: 0.01,

    momentum: 0.9,    Ok(())

    weight_decay: 1e-4,

    ..Default::default()}    let y = Tensor::ones([2, 3], DType::F32, Device::Cpu);    let z = x.add(&y)?;

});

```

// Adam

let optimizer = Adam::new(params, AdamConfig {    println!("Result: {}", z);

    lr: 0.001,

    betas: (0.9, 0.999),---

    eps: 1e-8,

    weight_decay: 0.0,    // Operations

});

## Training a Model (XOR Example)

// Training loop

optimizer.zero_grad();    let z = x.add(&y)?;    Ok(())

let loss = model.forward(&input)?.loss(&target)?;

backward(&loss)?;This is a complete, working example:

optimizer.step()?;

```    let w = z.mul_scalar(2.0)?;}



---```rust



## Learning Rate Schedulersuse ferrum::prelude::*;    ```



```rust

use ferrum::prelude::*;

fn main() -> Result<()> {    println!("Shape: {:?}", w.shape());

// Step decay

let scheduler = StepLR::new(10, 0.1);  // Decay by 0.1 every 10 epochs    // XOR truth table



// Cosine annealing    let inputs = Tensor::from_slice(&[    println!("Data: {:?}", w.to_vec::<f32>()?);---

let scheduler = CosineAnnealingLR::new(100, 1e-6);  // T_max=100, eta_min=1e-6

        0.0, 0.0,  // -> 0

// Use in training loop

for epoch in 0..100 {        0.0, 1.0,  // -> 1

    let current_lr = scheduler.get_lr(epoch, initial_lr);

    optimizer.set_lr(current_lr);        1.0, 0.0,  // -> 1

    // ... training ...

}        1.0, 1.0,  // -> 0    Ok(())## Your First Neural Network

```

    ], [4, 2], Device::Cpu);

---

}

## Data Loading

    let targets = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], [4, 1], Device::Cpu);

```rust

use ferrum::prelude::*;``````rust



let dataset = TensorDataset::new(train_inputs, train_targets);    // Build model



let loader = DataLoader::new(dataset)    let model = Sequential::new()use ferrum::prelude::*;

    .batch_size(32)

    .shuffle(true);        .add(Linear::new(2, 8))



for batch in &loader {        .add(Tanh::new())---

    let (inputs, targets) = batch;

    // Training code        .add(Linear::new(8, 1))

}

```        .add(Sigmoid::new());fn main() -> Result<()> {



---



## Loss Functions    // Optimizer## Building Neural Networks    // Build a 3-layer network



```rust    let optimizer = SGD::new(model.parameters(), SGDConfig {

let mse = mse_loss(&output, &target)?;       // Regression

let bce = bce_loss(&output, &target)?;       // Binary classification        lr: 1.0,    let model = Sequential::new()

let ce = cross_entropy_loss(&logits, &labels)?;  // Multi-class

```        momentum: 0.0,



---        weight_decay: 0.0,```rust        .add(Linear::new(784, 256))



## Save and Load Models        dampening: 0.0,



```rust        nesterov: false,use ferrum::prelude::*;        .add(ReLU::new())

// Save

save_model(&model, "model.bin")?;    });



// Load        .add(Linear::new(256, 128))

let model = load_model::<MyModel>("model.bin")?;

```    // Training loop



---    for epoch in 0..2000 {fn main() -> Result<()> {        .add(ReLU::new())



## Running Tests        // Forward pass



```bash        let output = model.forward(&inputs)?;    // Sequential model        .add(Linear::new(128, 10));

cargo test --workspace         # All 134 tests

cargo test -p ferrum-core      # Specific crate

cargo test -- --nocapture      # With output

```        // Compute loss    let model = Sequential::new()



---        let loss = bce_loss(&output, &targets)?;



## Next Steps        .add(Linear::new(784, 256))    // Create input



- See [API Reference](docs/API_REFERENCE.md) for complete documentation        // Backward pass

- See [PyTorch Comparison](docs/PYTORCH_COMPARISON.md) for migration guide

- Check out examples in `ferrum-examples/src/`        backward(&loss)?;        .add(ReLU::new())    let input = Tensor::randn([32, 784], DType::F32, Device::Cpu);



---



## Repository        // Update weights        .add(Linear::new(256, 10));



https://github.com/pratikacharya1234/FERRUM        optimizer.step()?;


        optimizer.zero_grad();    // Forward pass



        if epoch % 500 == 0 {    // Forward pass    let output = model.forward(&input)?;

            println!("Epoch {} | Loss: {:.6}", epoch, loss.item()?);

        }    let input = Tensor::randn([32, 784], DType::F32, Device::Cpu);

    }

    let output = model.forward(&input)?;    println!("Output shape: {:?}", output.shape());

    // Test

    let predictions = model.forward(&inputs)?;    println!("Parameters: {}", model.num_parameters());

    let pred_data = predictions.to_vec::<f32>()?;

    println!("Output shape: {:?}", output.shape());  // [32, 10]

    println!("\nResults:");

    let test_cases = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)];    Ok(())    Ok(())

    for (i, (a, b, target)) in test_cases.iter().enumerate() {

        let pred = pred_data[i];}}

        let pred_class = if pred > 0.5 { 1 } else { 0 };

        let correct = if pred_class == *target { "OK" } else { "FAIL" };``````

        println!("[{}, {}] -> Target: {}, Predicted: {:.4} ({}) [{}]",

                 a, b, target, pred, pred_class, correct);

    }

------

    Ok(())

}

```

## Complete Training Example (XOR)## Computing Loss

Run it:



```bash

cargo run --example train_xorThis is a working example that trains to 100% accuracy:```rust

```

use ferrum::prelude::*;

Expected output:

```rust

```

Epoch 0 | Loss: 0.693147use ferrum::prelude::*;fn main() -> Result<()> {

Epoch 500 | Loss: 0.034521

Epoch 1000 | Loss: 0.008234    // Predictions from model

Epoch 1500 | Loss: 0.003912

Epoch 2000 | Loss: 0.002156fn main() -> Result<()> {    let predictions = Tensor::randn([10, 5], DType::F32, Device::Cpu);



Results:    // XOR dataset    let targets = Tensor::randn([10, 5], DType::F32, Device::Cpu);

[0, 0] -> Target: 0, Predicted: 0.0106 (0) [OK]

[0, 1] -> Target: 1, Predicted: 0.9842 (1) [OK]    let inputs = Tensor::from_slice(

[1, 0] -> Target: 1, Predicted: 0.9835 (1) [OK]

[1, 1] -> Target: 0, Predicted: 0.0194 (0) [OK]        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],    // Compute MSE loss

Accuracy: 4/4 (100.0%)

```        [4, 2],    let loss = mse_loss(&predictions, &targets)?;



---        Device::Cpu    println!("Loss: {:.4}", loss.item()?);



## Using DataLoader    );



```rust    let targets = Tensor::from_slice(    // Other loss functions

use ferrum::prelude::*;

        &[0.0, 1.0, 1.0, 0.0],    let binary_predictions = predictions.sigmoid()?;

fn main() -> Result<()> {

    // Create dataset        [4, 1],    let binary_targets = Tensor::uniform([10, 5], 0.0, 1.0, DType::F32, Device::Cpu);

    let inputs = Tensor::randn([1000, 784], DType::F32, Device::Cpu);

    let targets = Tensor::randn([1000, 10], DType::F32, Device::Cpu);        Device::Cpu    let bce = bce_loss(&binary_predictions, &binary_targets)?;

    let dataset = TensorDataset::new(inputs, targets);

    );    println!("BCE Loss: {:.4}", bce.item()?);

    // Create DataLoader

    let loader = DataLoader::new(dataset)

        .batch_size(32)

        .shuffle(true)    // Model: 2 -> 8 -> 1    Ok(())

        .num_workers(4);

    let w1 = Tensor::randn([2, 8], DType::F32, Device::Cpu);}

    // Training loop

    for batch in &loader {    let b1 = Tensor::zeros([8], DType::F32, Device::Cpu);```

        let (batch_inputs, batch_targets) = batch;

        // ... train on batch    let w2 = Tensor::randn([8, 1], DType::F32, Device::Cpu);

    }

    let b2 = Tensor::zeros([1], DType::F32, Device::Cpu);---

    Ok(())

}

```

    let lr = 1.0;## Training Loop Structure

---



## Layer Reference

    for epoch in 0..2000 {```rust

### Linear Layer

        // Forward passuse ferrum::prelude::*;

```rust

let layer = Linear::new(in_features, out_features);        let h = inputs.matmul(&w1)?.add(&b1)?.tanh()?;

let output = layer.forward(&input)?;

```        let out = h.matmul(&w2)?.add(&b2)?.sigmoid()?;fn main() -> Result<()> {



### Activation Functions    // Create model



```rust        // Loss    let model = Sequential::new()

let relu = ReLU::new();

let sigmoid = Sigmoid::new();        let diff = out.sub(&targets)?;        .add(Linear::new(2, 4))

let tanh = Tanh::new();

let gelu = GELU::new();        let loss = diff.mul(&diff)?.mean()?;        .add(ReLU::new())

let silu = SiLU::new();

let leaky_relu = LeakyReLU::new(0.01);  // negative_slope        .add(Linear::new(4, 1));

let elu = ELU::new(1.0);                // alpha

let softmax = Softmax::new(-1);         // dim        if epoch % 500 == 0 {

let log_softmax = LogSoftmax::new(-1);  // dim

```            println!("Epoch {}: Loss = {:.6}", epoch, loss.item::<f32>()?);    // Create data



### Normalization        }    let x_train = Tensor::randn([100, 2], DType::F32, Device::Cpu);



```rust    let y_train = Tensor::randn([100, 1], DType::F32, Device::Cpu);

let layer_norm = LayerNorm::new(vec![256]);

let batch_norm = BatchNorm1d::new(256);        // Backward pass (manual for clarity)

```

        // ... gradient computation ...    // Training parameters

### Dropout

        // In practice, use backward(&loss)?    let learning_rate = 0.01;

```rust

let dropout = Dropout::new(0.5);  // 50% dropout rate    }    let epochs = 100;

```



### Sequential Container

    Ok(())    // Training loop

```rust

let model = Sequential::new()}    for epoch in 0..epochs {

    .add(Linear::new(784, 256))

    .add(ReLU::new())```        // Forward pass

    .add(Dropout::new(0.5))

    .add(Linear::new(256, 10));        let predictions = model.forward(&x_train)?;

```

Run the full example:

---

```bash        // Compute loss

## Loss Functions

cargo run --example train_xor        let loss = mse_loss(&predictions, &y_train)?;

```rust

// Regression```

let loss = mse_loss(&predictions, &targets)?;

        // Print progress

// Binary classification

let loss = bce_loss(&predictions, &targets)?;---        if epoch % 10 == 0 {



// Multi-class classification            println!("Epoch {}: Loss = {:.4}", epoch, loss.item()?);

let loss = cross_entropy_loss(&logits, &class_indices)?;

## Using DataLoader        }

// Negative log likelihood (use with log_softmax)

let loss = nll_loss(&log_probs, &class_indices)?;



// L1 loss```rust        // TODO: Backward pass (requires autograd completion)

let loss = l1_loss(&predictions, &targets)?;

use ferrum::prelude::*;        // loss.backward()?;

// Smooth L1 (Huber) loss

let loss = smooth_l1_loss(&predictions, &targets)?;

```

fn main() -> Result<()> {        // TODO: Optimizer step (requires gradients)

---

    // Create dataset        // optimizer.step()?;

## Optimizers

    let inputs = Tensor::randn([1000, 10], DType::F32, Device::Cpu);    }

### SGD

    let targets = Tensor::randn([1000, 1], DType::F32, Device::Cpu);

```rust

let optimizer = SGD::new(model.parameters(), SGDConfig {    let dataset = TensorDataset::new(vec![inputs], vec![targets]);    Ok(())

    lr: 0.01,

    momentum: 0.9,}

    weight_decay: 1e-4,

    dampening: 0.0,    // Create DataLoader```

    nesterov: false,

});    let loader = DataLoader::new(dataset)

```

        .batch_size(32)---

### Adam

        .shuffle(true);

```rust

let optimizer = Adam::new(model.parameters(), AdamConfig {## Common Operations

    lr: 0.001,

    betas: (0.9, 0.999),    // Iterate

    eps: 1e-8,

    weight_decay: 0.0,    for batch in &loader {### Tensor Creation

});

```        println!("Batch size: {}", batch.len());



### Training Step    }```rust



```rust// Zeros and ones

optimizer.zero_grad();

let loss = compute_loss()?;    Ok(())let zeros = Tensor::zeros([3, 4], DType::F32, Device::Cpu);

backward(&loss)?;

optimizer.step()?;}let ones = Tensor::ones([3, 4], DType::F32, Device::Cpu);

```

```

---

// Random

## Save and Load Models

---let randn = Tensor::randn([3, 4], DType::F32, Device::Cpu);

```rust

use ferrum::serialize::{save, load};let uniform = Tensor::uniform([3, 4], 0.0, 1.0, DType::F32, Device::Cpu);



// Save model## Optimizers

save(&model.state_dict(), "model.ferrum")?;

// From data

// Load model

let state = load("model.ferrum")?;### SGDlet data = vec![1.0, 2.0, 3.0, 4.0];

model.load_state_dict(&state)?;

```let tensor = Tensor::from_slice(&data, [2, 2], Device::Cpu)?;



---```rust



## Common Errorslet optimizer = SGD::new(model.parameters(), SGDConfig {// Ranges



### "Tensor dimensions don't match"    lr: 0.01,let range = Tensor::arange(0.0, 10.0, 1.0, DType::F32, Device::Cpu);



Check that your tensor shapes are compatible for the operation. Use `tensor.shape()` to debug.    momentum: 0.9,let linspace = Tensor::linspace(0.0, 1.0, 100, DType::F32, Device::Cpu);



### "Gradient not found"    weight_decay: 1e-4,```



Make sure you called `backward(&loss)?` before `optimizer.step()`.    ..Default::default()



### "Cannot borrow as mutable"});### Arithmetic



Use `let mut` for tensors you want to modify, or clone if needed.```



---```rust



## Next Steps### Adamlet a = Tensor::ones([2, 3], DType::F32, Device::Cpu);



1. Read the [API Reference](docs/API_REFERENCE.md)let b = Tensor::full([3], 2.0, DType::F32, Device::Cpu);

2. Check the [examples](ferrum-examples/src/)

3. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for current limitations```rust


let optimizer = Adam::new(model.parameters(), AdamConfig {// Element-wise operations

    lr: 0.001,let sum = a.add(&b)?;         // Broadcasting

    betas: (0.9, 0.999),let product = a.mul(&b)?;

    eps: 1e-8,let difference = a.sub(&b)?;

    weight_decay: 0.0,let quotient = a.div(&b)?;

});

```// Scalar operations

let scaled = a.mul_scalar(2.5)?;

### Training Steplet shifted = a.add_scalar(1.0)?;

```

```rust

optimizer.zero_grad();### Matrix Operations

let output = model.forward(&input)?;

let loss = mse_loss(&output, &target)?;```rust

backward(&loss)?;let a = Tensor::randn([64, 128], DType::F32, Device::Cpu);

optimizer.step()?;let b = Tensor::randn([128, 256], DType::F32, Device::Cpu);

```

// Matrix multiplication

---let c = a.matmul(&b)?;  // Shape: [64, 256]



## Loss Functions// Transpose

let at = a.t()?;        // Shape: [128, 64]

```rust```

// Regression

let loss = mse_loss(&predictions, &targets)?;### Activations

let loss = l1_loss(&predictions, &targets)?;

```rust

// Binary classificationlet x = Tensor::randn([10, 10], DType::F32, Device::Cpu);

let loss = bce_loss(&sigmoid_output, &binary_targets)?;

let relu_out = x.relu()?;

// Multi-class classificationlet sigmoid_out = x.sigmoid()?;

let loss = cross_entropy_loss(&logits, &class_indices)?;let tanh_out = x.tanh()?;

let loss = nll_loss(&log_softmax_output, &class_indices)?;```

```

### Reductions

---

```rust

## Saving and Loadinglet tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [4], Device::Cpu)?;



```rustlet sum = tensor.sum()?;           // 10.0

use ferrum::serialize::{save, load};let mean = tensor.mean()?;         // 2.5

let sum_val = sum.item()?;         // Extract scalar

// Save model weights```

let state = model.state_dict();

save(&state, "model.ferrum")?;### Shape Manipulation



// Load model weights```rust

let loaded_state = load("model.ferrum")?;let x = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);

model.load_state_dict(&loaded_state)?;

```// Reshape

let reshaped = x.reshape([6, 4])?;

---

// Flatten

## Available Layerslet flat = x.flatten()?;



| Layer | Constructor |// Squeeze/Unsqueeze

|-------|-------------|let squeezed = x.squeeze(None)?;

| Fully Connected | `Linear::new(in_features, out_features)` |let unsqueezed = x.unsqueeze(0)?;

| ReLU | `ReLU::new()` |

| Sigmoid | `Sigmoid::new()` |// Transpose

| Tanh | `Tanh::new()` |let transposed = x.transpose(0, 2)?;

| GELU | `GELU::new()` |```

| SiLU/Swish | `SiLU::new()` |

| Leaky ReLU | `LeakyReLU::new(negative_slope)` |---

| ELU | `ELU::new(alpha)` |

| Softmax | `Softmax::new(dim)` |## Running Examples

| LogSoftmax | `LogSoftmax::new(dim)` |

| Layer Norm | `LayerNorm::new(normalized_shape)` |FERRUM includes several examples:

| Batch Norm | `BatchNorm1d::new(num_features)` |

| Dropout | `Dropout::new(p)` |```bash

| Sequential | `Sequential::new()` |# Simple neural network

cargo run --example simple_nn

---

# XOR training (forward pass only)

## Running Examplescargo run --example train_xor



```bash# MNIST (skeleton)

# XOR training (100% accuracy)cargo run --example mnist

cargo run --example train_xor```



# Simple neural network---

cargo run --example simple_nn

## Next Steps

# Run all tests

cargo test --workspace1. **Explore the API**: Check out [API_REFERENCE.md](docs/API_REFERENCE.md)

```2. **Read the source**: Browse `ferrum-core/src/tensor.rs` for tensor operations

3. **Check examples**: Look in `ferrum-examples/src/` for more code

---4. **Run tests**: `cargo test` to see everything in action

5. **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

## Troubleshooting

---

### Build Errors

## Common Pitfalls

```bash

# Clean and rebuild### 1. Shape Mismatches

cargo clean

cargo build --release```rust

```// ❌ Wrong

let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);

### Test Failureslet b = Tensor::ones([4, 5], DType::F32, Device::Cpu);

let c = a.add(&b)?;  // Error: shapes don't broadcast

```bash

# Run tests with output// ✅ Correct

cargo test -- --nocapturelet a = Tensor::ones([2, 3], DType::F32, Device::Cpu);

```let b = Tensor::ones([3], DType::F32, Device::Cpu);

let c = a.add(&b)?;  // OK: [2,3] + [3] -> [2,3]

### Import Issues```



Make sure you import the prelude:### 2. Device Mismatches

```rust

use ferrum::prelude::*;```rust

```// ❌ Wrong (when GPU support is added)

let a = Tensor::ones([2, 2], DType::F32, Device::Cpu);

---let b = Tensor::ones([2, 2], DType::F32, Device::Cuda(0));

let c = a.add(&b)?;  // Error: different devices

## Next Steps

// ✅ Correct

1. Read the [API Reference](docs/API_REFERENCE.md)let a = Tensor::ones([2, 2], DType::F32, Device::Cpu);

2. Check the [examples](ferrum-examples/src/)let b = Tensor::ones([2, 2], DType::F32, Device::Cpu);

3. See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for what's supportedlet c = a.add(&b)?;

```

### 3. Non-Contiguous Tensors

```rust
let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);
let y = x.t()?;  // Transpose creates non-contiguous view

// Some operations require contiguous memory
let contiguous = y.contiguous()?;
```

---

## Performance Tips

1. **Use appropriate dtypes**: F32 is usually faster than F64
2. **Batch operations**: Process multiple samples together
3. **Reuse memory**: Avoid unnecessary clones
4. **Profile first**: Use `cargo flamegraph` to find bottlenecks

---

## Getting Help

- **Issues**: Report bugs at GitHub Issues
- **Documentation**: See `docs/API_REFERENCE.md`
- **Examples**: Check `ferrum-examples/src/`
- **Tests**: Look at test files for usage patterns

---

## What's Next?

FERRUM is under active development. Upcoming features:

- ✅ Complete autograd integration
- ⏳ GPU support (CUDA, Metal)
- ⏳ Convolutional layers
- ⏳ Batch normalization
- ⏳ Data loading utilities
- ⏳ Pre-trained models

---

**Ready to build? Start coding with FERRUM! 🦀🔥**
