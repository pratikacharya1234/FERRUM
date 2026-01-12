# FERRUM API Reference# FERRUM API Reference# FERRUM API Reference



Complete API documentation for the FERRUM deep learning framework.



---Complete API documentation for the FERRUM deep learning framework.Complete API documentation for the FERRUM deep learning framework.



## Table of Contents



1. [Tensor Operations](#tensor-operations)---## Table of Contents

2. [Neural Network Layers](#neural-network-layers)

3. [Loss Functions](#loss-functions)

4. [Optimizers](#optimizers)

5. [Data Loading](#data-loading)## Table of Contents1. [Tensor Operations](#tensor-operations)

6. [Distributed Training](#distributed-training)

7. [Autograd](#autograd)2. [Neural Network Layers](#neural-network-layers)

8. [Serialization](#serialization)

1. [Tensor Operations](#tensor-operations)3. [Loss Functions](#loss-functions)

---

2. [Neural Network Layers](#neural-network-layers)4. [Optimizers](#optimizers)

## Tensor Operations

3. [Loss Functions](#loss-functions)5. [Autograd](#autograd)

### Creation

4. [Optimizers](#optimizers)6. [Serialization](#serialization)

| Function | Description |

|----------|-------------|5. [Data Loading](#data-loading)

| `Tensor::zeros(shape, dtype, device)` | Tensor filled with zeros |

| `Tensor::ones(shape, dtype, device)` | Tensor filled with ones |6. [Distributed Training](#distributed-training)---

| `Tensor::full(shape, value, dtype, device)` | Tensor filled with value |

| `Tensor::randn(shape, dtype, device)` | Standard normal distribution |7. [Autograd](#autograd)

| `Tensor::uniform(shape, low, high, dtype, device)` | Uniform distribution |

| `Tensor::eye(n, dtype, device)` | Identity matrix |8. [Serialization](#serialization)## Tensor Operations

| `Tensor::arange(start, end, step, dtype, device)` | Evenly spaced values |

| `Tensor::linspace(start, end, n, dtype, device)` | N evenly spaced values |

| `Tensor::from_slice(data, shape, device)` | From Rust slice |

---### Creation

```rust

let x = Tensor::randn([128, 256], DType::F32, Device::Cpu);

let y = Tensor::zeros([256], DType::F32, Device::Cpu);

let z = Tensor::from_slice(&[1.0, 2.0, 3.0], [3], Device::Cpu)?;## Tensor Operations#### `Tensor::zeros(shape, dtype, device)`

```

Create a tensor filled with zeros.

### Arithmetic

### Creation

| Operation | Description |

|-----------|-------------|```rust

| `add(other)` | Element-wise addition |

| `sub(other)` | Element-wise subtraction || Function | Description |let t = Tensor::zeros([2, 3], DType::F32, Device::Cpu);

| `mul(other)` | Element-wise multiplication |

| `div(other)` | Element-wise division ||----------|-------------|```

| `neg()` | Negation |

| `add_scalar(value)` | Add scalar || `Tensor::zeros(shape, dtype, device)` | Tensor filled with zeros |

| `mul_scalar(value)` | Multiply by scalar |

| `Tensor::ones(shape, dtype, device)` | Tensor filled with ones |#### `Tensor::ones(shape, dtype, device)`

```rust

let c = a.add(&b)?;| `Tensor::full(shape, value, dtype, device)` | Tensor filled with value |Create a tensor filled with ones.

let d = c.mul_scalar(2.0)?;

```| `Tensor::randn(shape, dtype, device)` | Standard normal distribution |



### Matrix Operations| `Tensor::uniform(shape, low, high, dtype, device)` | Uniform distribution |```rust



| Operation | Description || `Tensor::eye(n, dtype, device)` | Identity matrix |let t = Tensor::ones([2, 3], DType::F32, Device::Cpu);

|-----------|-------------|

| `matmul(other)` | Matrix multiplication || `Tensor::arange(start, end, step, dtype, device)` | Evenly spaced values |```

| `t()` | Transpose |

| `transpose(dim0, dim1)` | Swap dimensions || `Tensor::linspace(start, end, n, dtype, device)` | N evenly spaced values |



```rust| `Tensor::from_slice(data, shape, device)` | From Rust slice |#### `Tensor::full(shape, value, dtype, device)`

let result = x.matmul(&w)?;  // [batch, in] x [in, out] -> [batch, out]

let transposed = x.t()?;Create a tensor filled with a specific value.

```

```rust

### Reductions

let x = Tensor::randn([128, 256], DType::F32, Device::Cpu);```rust

| Operation | Description |

|-----------|-------------|let y = Tensor::zeros([256], DType::F32, Device::Cpu);let t = Tensor::full([2, 3], 5.0, DType::F32, Device::Cpu);

| `sum()` | Sum all elements |

| `sum_dim(dim, keepdim)` | Sum along dimension |let z = Tensor::from_slice(&[1.0, 2.0, 3.0], [3], Device::Cpu)?;```

| `mean()` | Mean of all elements |

| `mean_dim(dim, keepdim)` | Mean along dimension |```

| `max()` | Maximum value |

| `min()` | Minimum value |#### `Tensor::randn(shape, dtype, device)`



```rust### ArithmeticCreate a tensor with values from standard normal distribution N(0, 1).

let total = x.sum()?;

let row_means = x.mean_dim(1, true)?;

```

| Operation | Description |```rust

### Shape Operations

|-----------|-------------|let t = Tensor::randn([128, 256], DType::F32, Device::Cpu);

| Operation | Description |

|-----------|-------------|| `add(other)` | Element-wise addition |```

| `reshape(shape)` | Change shape |

| `view(shape)` | View with new shape || `sub(other)` | Element-wise subtraction |

| `squeeze(dim)` | Remove dimension of size 1 |

| `unsqueeze(dim)` | Add dimension of size 1 || `mul(other)` | Element-wise multiplication |#### `Tensor::uniform(shape, low, high, dtype, device)`

| `flatten(start, end)` | Flatten dimensions |

| `permute(dims)` | Reorder dimensions || `div(other)` | Element-wise division |Create a tensor with values uniformly distributed in [low, high).



```rust| `neg()` | Negation |

let flat = x.reshape([-1])?;

let batched = x.unsqueeze(0)?;| `add_scalar(value)` | Add scalar |```rust

```

| `mul_scalar(value)` | Multiply by scalar |let t = Tensor::uniform([100], 0.0, 1.0, DType::F32, Device::Cpu);

### Math Functions

```

| Operation | Description |

|-----------|-------------|```rust

| `exp()` | Exponential |

| `log()` | Natural logarithm |let c = a.add(&b)?;#### `Tensor::eye(n, dtype, device)`

| `pow(exponent)` | Power |

| `sqrt()` | Square root |let d = c.mul_scalar(2.0)?;Create an identity matrix of size n×n.

| `abs()` | Absolute value |

| `sin()` / `cos()` | Trigonometric |```



### Activations```rust



| Operation | Description |### Matrix Operationslet identity = Tensor::eye(5, DType::F32, Device::Cpu);

|-----------|-------------|

| `relu()` | ReLU activation |```

| `sigmoid()` | Sigmoid activation |

| `tanh()` | Tanh activation || Operation | Description |

| `gelu()` | GELU activation |

| `silu()` | SiLU/Swish activation ||-----------|-------------|#### `Tensor::arange(start, end, step, dtype, device)`



---| `matmul(other)` | Matrix multiplication |Create a 1D tensor with evenly spaced values.



## Neural Network Layers| `t()` | Transpose |



### Linear| `transpose(dim0, dim1)` | Swap dimensions |```rust



Fully connected layer: y = xW^T + blet t = Tensor::arange(0.0, 10.0, 1.0, DType::F32, Device::Cpu);



```rust```rust// [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

let layer = Linear::new(in_features, out_features);

let output = layer.forward(&input)?;let result = x.matmul(&w)?;  // [batch, in] x [in, out] -> [batch, out]```

```

let transposed = x.t()?;

### Activations

```#### `Tensor::linspace(start, end, n, dtype, device)`

| Layer | Constructor |

|-------|-------------|Create a 1D tensor with n evenly spaced values between start and end.

| ReLU | `ReLU::new()` |

| Sigmoid | `Sigmoid::new()` |### Reductions

| Tanh | `Tanh::new()` |

| GELU | `GELU::new()` |```rust

| SiLU | `SiLU::new()` |

| LeakyReLU | `LeakyReLU::new(negative_slope)` || Operation | Description |let t = Tensor::linspace(0.0, 1.0, 5, DType::F32, Device::Cpu);

| ELU | `ELU::new(alpha)` |

| Softmax | `Softmax::new(dim)` ||-----------|-------------|// [0.0, 0.25, 0.5, 0.75, 1.0]

| LogSoftmax | `LogSoftmax::new(dim)` |

| `sum()` | Sum all elements |```

```rust

let activation = ReLU::new();| `sum_dim(dim, keepdim)` | Sum along dimension |

let output = activation.forward(&input)?;

```| `mean()` | Mean of all elements |#### `Tensor::from_slice(data, shape, device)`



### Normalization| `mean_dim(dim, keepdim)` | Mean along dimension |Create a tensor from a Rust slice.



#### LayerNorm| `max()` | Maximum value |



```rust| `min()` | Minimum value |```rust

let norm = LayerNorm::new(normalized_shape);

let output = norm.forward(&input)?;let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

```

```rustlet t = Tensor::from_slice(&data, [2, 3], Device::Cpu)?;

#### BatchNorm1d

let total = x.sum()?;```

```rust

let norm = BatchNorm1d::new(num_features);let row_means = x.mean_dim(1, true)?;

let output = norm.forward(&input)?;

``````---



### Dropout



```rust### Shape Operations### Arithmetic Operations

let dropout = Dropout::new(p);  // p = dropout probability

let output = dropout.forward(&input)?;

```

| Operation | Description |#### `tensor.add(other)` / `tensor.sub(other)`

### Sequential

|-----------|-------------|Element-wise addition/subtraction with broadcasting.

Container for stacking layers.

| `reshape(shape)` | Change shape |

```rust

let model = Sequential::new()| `view(shape)` | View with new shape |```rust

    .add(Linear::new(784, 256))

    .add(ReLU::new())| `squeeze(dim)` | Remove dimension of size 1 |let a = Tensor::ones([2, 3], DType::F32, Device::Cpu);

    .add(Linear::new(256, 10));

| `unsqueeze(dim)` | Add dimension of size 1 |let b = Tensor::ones([3], DType::F32, Device::Cpu);

let output = model.forward(&input)?;

let params = model.parameters();| `flatten(start, end)` | Flatten dimensions |let c = a.add(&b)?;  // Broadcasting: [2,3] + [3] -> [2,3]

```

| `permute(dims)` | Reorder dimensions |```

---



## Loss Functions

```rust#### `tensor.mul(other)` / `tensor.div(other)`

### MSE Loss

let flat = x.reshape([-1])?;Element-wise multiplication/division with broadcasting.

Mean squared error for regression.

let batched = x.unsqueeze(0)?;

```rust

let loss = mse_loss(&predictions, &targets)?;``````rust

```

let a = Tensor::randn([2, 3], DType::F32, Device::Cpu);

### BCE Loss

### Math Functionslet b = Tensor::full([3], 2.0, DType::F32, Device::Cpu);

Binary cross-entropy for binary classification.

let c = a.mul(&b)?;  // Multiply each row by [2, 2, 2]

```rust

let loss = bce_loss(&predictions, &targets)?;| Operation | Description |```

// predictions should be in [0, 1] (use sigmoid)

```|-----------|-------------|



### Cross Entropy Loss| `exp()` | Exponential |#### `tensor.add_scalar(value)` / `tensor.mul_scalar(value)`



For multi-class classification.| `log()` | Natural logarithm |Add/multiply a scalar to all elements.



```rust| `pow(exponent)` | Power |

let loss = cross_entropy_loss(&logits, &class_indices)?;

// logits: [batch, num_classes]| `sqrt()` | Square root |```rust

// class_indices: [batch] with values in [0, num_classes)

```| `abs()` | Absolute value |let t = Tensor::ones([3, 3], DType::F32, Device::Cpu);



### NLL Loss| `sin()` / `cos()` | Trigonometric |let t2 = t.add_scalar(5.0)?;  // All elements become 6.0



Negative log likelihood (use with log_softmax).```



```rust### Activations

let log_probs = log_softmax(&logits, -1)?;

let loss = nll_loss(&log_probs, &class_indices)?;#### `tensor.neg()`

```

| Operation | Description |Negate all elements.

### L1 Loss

|-----------|-------------|

Mean absolute error.

| `relu()` | ReLU activation |```rust

```rust

let loss = l1_loss(&predictions, &targets)?;| `sigmoid()` | Sigmoid activation |let t = Tensor::from_slice(&[1.0, -2.0, 3.0], [3], Device::Cpu)?;

```

| `tanh()` | Tanh activation |let neg = t.neg()?;  // [-1.0, 2.0, -3.0]

### Smooth L1 Loss

| `gelu()` | GELU activation |```

Huber loss for robust regression.

| `silu()` | SiLU/Swish activation |

```rust

let loss = smooth_l1_loss(&predictions, &targets)?;---

```

---

---

### Matrix Operations

## Optimizers

## Neural Network Layers

### SGD

#### `tensor.matmul(other)`

Stochastic gradient descent with momentum.

### LinearMatrix multiplication.

```rust

let optimizer = SGD::new(model.parameters(), SGDConfig {

    lr: 0.01,

    momentum: 0.9,Fully connected layer: y = xW^T + b```rust

    weight_decay: 1e-4,

    dampening: 0.0,let a = Tensor::randn([64, 128], DType::F32, Device::Cpu);

    nesterov: false,

});```rustlet b = Tensor::randn([128, 256], DType::F32, Device::Cpu);

```

let layer = Linear::new(in_features, out_features);let c = a.matmul(&b)?;  // Shape: [64, 256]

### Adam

let output = layer.forward(&input)?;```

Adaptive moment estimation.

```

```rust

let optimizer = Adam::new(model.parameters(), AdamConfig {#### `tensor.t()`

    lr: 0.001,

    betas: (0.9, 0.999),### ActivationsTranspose a 2D tensor.

    eps: 1e-8,

    weight_decay: 0.0,

});

```| Layer | Constructor |```rust



### Training Step|-------|-------------|let a = Tensor::randn([3, 4], DType::F32, Device::Cpu);



```rust| ReLU | `ReLU::new()` |let b = a.t()?;  // Shape: [4, 3]

optimizer.zero_grad();

let output = model.forward(&input)?;| Sigmoid | `Sigmoid::new()` |```

let loss = loss_fn(&output, &target)?;

backward(&loss)?;| Tanh | `Tanh::new()` |

optimizer.step()?;

```| GELU | `GELU::new()` |#### `tensor.transpose(dim0, dim1)`



---| SiLU | `SiLU::new()` |Swap two dimensions.



## Data Loading| LeakyReLU | `LeakyReLU::new(negative_slope)` |



### Dataset Trait| ELU | `ELU::new(alpha)` |```rust



```rust| Softmax | `Softmax::new(dim)` |let a = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);

pub trait Dataset {

    type Item;| LogSoftmax | `LogSoftmax::new(dim)` |let b = a.transpose(0, 2)?;  // Shape: [4, 3, 2]

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Item>;```

}

``````rust



### TensorDatasetlet relu = ReLU::new();---



```rustlet output = relu.forward(&input)?;

let dataset = TensorDataset::new(inputs, targets);

``````### Activation Functions



### DataLoader



```rust### Normalization#### `tensor.relu()`

let loader = DataLoader::new(dataset)

    .batch_size(32)ReLU activation: max(0, x).

    .shuffle(true)

    .num_workers(4)| Layer | Constructor |

    .drop_last(false);

|-------|-------------|```rust

for batch in &loader {

    let (inputs, targets) = batch;| LayerNorm | `LayerNorm::new(normalized_shape)` |let x = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], [4], Device::Cpu)?;

    // training code

}| BatchNorm1d | `BatchNorm1d::new(num_features)` |let y = x.relu()?;  // [0.0, 0.0, 1.0, 2.0]

```

```

### Samplers

```rust

| Sampler | Description |

|---------|-------------|let norm = LayerNorm::new(vec![256]);#### `tensor.sigmoid()`

| `SequentialSampler` | In-order iteration |

| `RandomSampler` | Shuffled iteration |let output = norm.forward(&input)?;Sigmoid activation: 1 / (1 + exp(-x)).

| `WeightedRandomSampler` | Weighted sampling |

| `SubsetSampler` | Sample from indices |```

| `DistributedSampler` | Multi-GPU data splitting |

| `BatchSampler` | Batched indices |```rust



### Transforms### Regularizationlet x = Tensor::zeros([10], DType::F32, Device::Cpu);



```rustlet y = x.sigmoid()?;  // All 0.5

let transform = Compose::new(vec![

    Box::new(Normalize::new(mean, std)),```rust```

]);

let dropout = Dropout::new(0.5);  // 50% dropout

let normalized = transform.apply(&tensor)?;

```let output = dropout.forward(&input)?;#### `tensor.tanh()`



---```Hyperbolic tangent activation.



## Distributed Training



### ProcessGroup### Container```rust



```rustlet x = Tensor::randn([5, 5], DType::F32, Device::Cpu);

let pg = ProcessGroup::new(Backend::Gloo)?;

let rank = pg.rank();```rustlet y = x.tanh()?;

let world_size = pg.world_size();

```let model = Sequential::new()```



### Collectives    .add(Linear::new(784, 256))



```rust    .add(ReLU::new())---

// Broadcast from rank 0

broadcast(&mut tensor, 0, &pg)?;    .add(Linear::new(256, 10));



// All-reduce (sum)### Reduction Operations

all_reduce(&mut tensor, ReduceOp::Sum, &pg)?;

let output = model.forward(&input)?;

// Gather to rank 0

let gathered = gather(&tensor, 0, &pg)?;let params = model.parameters();#### `tensor.sum()`



// Scatter from rank 0```Sum all elements.

let received = scatter(&tensors, 0, &pg)?;



// Barrier

barrier(&pg)?;---```rust

```

let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [4], Device::Cpu)?;

### DistributedDataParallel

## Loss Functionslet s = t.sum()?;

```rust

let model = DistributedDataParallel::new(model, &pg)?;assert_eq!(s.item()?, 10.0);



// Training is same as single-GPU| Function | Use Case |```

let output = model.forward(&input)?;

let loss = loss_fn(&output, &target)?;|----------|----------|

backward(&loss)?;  // Gradients auto-synced

optimizer.step()?;| `mse_loss(pred, target)` | Regression |#### `tensor.mean()`

```

| `l1_loss(pred, target)` | Robust regression |Mean of all elements.

### DistributedSampler

| `smooth_l1_loss(pred, target)` | Object detection |

```rust

let sampler = DistributedSampler::new(dataset.len(), world_size, rank);| `bce_loss(pred, target)` | Binary classification |```rust

let loader = DataLoader::new(dataset).sampler(sampler);

```| `cross_entropy_loss(logits, targets)` | Multi-class (from logits) |let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], [4], Device::Cpu)?;



---| `nll_loss(log_probs, targets)` | Multi-class (from log_softmax) |let m = t.mean()?;



## Autogradassert_eq!(m.item()?, 2.5);



### Enable/Disable Gradients```rust```



```rust// Regression

// Tensors track gradients by default

let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);let loss = mse_loss(&predictions, &targets)?;---



// Disable gradient tracking

let _guard = no_grad();

let y = expensive_computation()?;  // No gradients recorded// Binary classification### Mathematical Functions

// guard dropped, gradients enabled again

```let loss = bce_loss(&sigmoid_output, &binary_targets)?;



### Backward Pass#### `tensor.exp()` / `tensor.log()`



```rust// Multi-class classificationElement-wise exponential/logarithm.

let loss = compute_loss()?;

backward(&loss)?;  // Compute gradientslet loss = cross_entropy_loss(&logits, &class_indices)?;



// Access gradients``````rust

let grad = tensor.grad()?;

```let x = Tensor::zeros([3], DType::F32, Device::Cpu);



### Gradient Checking---let exp_x = x.exp()?;  // All 1.0



```rust

use ferrum_autograd::gradcheck;

## Optimizerslet y = Tensor::from_slice(&[1.0, 2.7183, 7.389], [3], Device::Cpu)?;

let passed = gradcheck(

    |x| x.mul(&x)?.sum(),  // Function to checklet log_y = y.log()?;  // Approximately [0, 1, 2]

    &input,

    1e-5,  // epsilon### SGD```

    1e-3,  // tolerance

)?;

assert!(passed);

``````rust#### `tensor.pow(exponent)`



---let optimizer = SGD::new(parameters, SGDConfig {Element-wise power.



## Serialization    lr: 0.01,



### Save Tensor    momentum: 0.9,```rust



```rust    weight_decay: 1e-4,let x = Tensor::from_slice(&[2.0, 3.0], [2], Device::Cpu)?;

use ferrum::serialize::{save, load};

    dampening: 0.0,let squared = x.pow(2.0)?;  // [4.0, 9.0]

save(&tensor, "tensor.ferrum")?;

```    nesterov: false,```



### Load Tensor});



```rust```#### `tensor.sqrt()`

let tensor = load("tensor.ferrum")?;

```Element-wise square root.



### Save Model### Adam



```rust```rust

let state = model.state_dict();

save(&state, "model.ferrum")?;```rustlet x = Tensor::from_slice(&[4.0, 9.0, 16.0], [3], Device::Cpu)?;

```

let optimizer = Adam::new(parameters, AdamConfig {let y = x.sqrt()?;  // [2.0, 3.0, 4.0]

### Load Model

    lr: 0.001,```

```rust

let state = load("model.ferrum")?;    betas: (0.9, 0.999),

model.load_state_dict(&state)?;

```    eps: 1e-8,---



---    weight_decay: 0.0,



## CUDA (Simulated)});### Shape Manipulation



Note: CUDA support is currently simulated. No actual GPU execution.```



### Device Selection#### `tensor.reshape(new_shape)`



```rust### Common MethodsReshape tensor (view if possible, copies if needed).

let cpu_tensor = Tensor::randn([100, 100], DType::F32, Device::Cpu);

let cuda_tensor = Tensor::randn([100, 100], DType::F32, Device::Cuda(0));

```

```rust```rust

### Device Transfer

optimizer.zero_grad();      // Clear gradientslet t = Tensor::arange(0.0, 12.0, 1.0, DType::F32, Device::Cpu);

```rust

let gpu_tensor = cpu_tensor.to_device(Device::Cuda(0))?;optimizer.step()?;          // Update parameterslet reshaped = t.reshape([3, 4])?;  // Shape: [3, 4]

let back_to_cpu = gpu_tensor.to_device(Device::Cpu)?;

`````````



### CudaDevice



```rust---#### `tensor.flatten()`

use ferrum_cuda::CudaDevice;

Flatten to 1D tensor.

let device = CudaDevice::new(0)?;

println!("Device: {}", device.name());## Data Loading

println!("Memory: {} bytes", device.total_memory());

``````rust



---### Dataset Traitlet t = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);



## Error Handlinglet flat = t.flatten()?;  // Shape: [24]



All operations return `Result<T, Error>`.```rust```



```rustpub trait Dataset: Send + Sync {

use ferrum::error::Error;

    type Sample: Send;#### `tensor.squeeze(dim)` / `tensor.unsqueeze(dim)`

match tensor.matmul(&other) {

    Ok(result) => println!("Success: {:?}", result.shape()),    fn len(&self) -> usize;Remove/add dimensions of size 1.

    Err(Error::ShapeMismatch { expected, got }) => {

        eprintln!("Shape error: expected {:?}, got {:?}", expected, got);    fn get(&self, index: usize) -> Self::Sample;

    }

    Err(e) => eprintln!("Other error: {}", e),}```rust

}

``````let t = Tensor::randn([1, 3, 1, 5], DType::F32, Device::Cpu);



### Common Errorslet squeezed = t.squeeze(None)?;  // Shape: [3, 5]



| Error | Cause |### TensorDataset

|-------|-------|

| `ShapeMismatch` | Incompatible tensor shapes |let t2 = Tensor::randn([3, 5], DType::F32, Device::Cpu);

| `DTypeMismatch` | Incompatible data types |

| `DeviceMismatch` | Tensors on different devices |```rustlet unsqueezed = t2.unsqueeze(0)?;  // Shape: [1, 3, 5]

| `InvalidOperation` | Operation not supported |

| `GradientError` | Autograd-related error |let dataset = TensorDataset::new(inputs_vec, targets_vec);```



---let sample = dataset.get(0);



## Prelude```#### `tensor.permute(dims)`



Import common items:Permute dimensions.



```rust### DataLoader

use ferrum::prelude::*;

```rust

// Includes:

// - Tensor, DType, Device```rustlet t = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);

// - All layers (Linear, ReLU, etc.)

// - All loss functionslet loader = DataLoader::new(dataset)let permuted = t.permute(&[2, 0, 1])?;  // Shape: [4, 2, 3]

// - Optimizers (SGD, Adam)

// - backward(), no_grad()    .batch_size(32)```

// - Result, Error

```    .shuffle(true)


    .drop_last(false)---

    .num_workers(4);

### Tensor Properties

for batch in &loader {

    // Process batch#### `tensor.shape()` → `&[usize]`

}Get tensor shape.

```

```rust

### Samplerslet t = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);

assert_eq!(t.shape(), &[2, 3, 4]);

| Sampler | Description |```

|---------|-------------|

| `SequentialSampler::new(len)` | In-order indices |#### `tensor.ndim()` → `usize`

| `RandomSampler::new(len)` | Shuffled indices |Get number of dimensions.

| `RandomSampler::with_seed(len, seed)` | Reproducible shuffle |

| `WeightedRandomSampler::new(weights, n)` | Weighted sampling |```rust

| `SubsetRandomSampler::new(indices)` | Sample from subset |let t = Tensor::randn([2, 3], DType::F32, Device::Cpu);

| `DistributedSampler::new(len, world, rank, shuffle)` | Multi-GPU |assert_eq!(t.ndim(), 2);

| `BatchSampler::new(sampler, batch_size, drop_last)` | Batched indices |```



---#### `tensor.numel()` → `usize`

Get total number of elements.

## Distributed Training

```rust

### Initializationlet t = Tensor::randn([2, 3, 4], DType::F32, Device::Cpu);

assert_eq!(t.numel(), 24);

```rust```

use ferrum::distributed::*;

#### `tensor.dtype()` → `DType`

// From parametersGet data type.

let pg = init_process_group(Backend::Gloo, rank, world_size)?;

```rust

// From environment variables (RANK, WORLD_SIZE)let t = Tensor::zeros([3], DType::F32, Device::Cpu);

let pg = init_process_group_from_env()?;assert_eq!(t.dtype(), DType::F32);

``````



### Process Group#### `tensor.device()` → `Device`

Get device.

```rust

let rank = pg.rank();```rust

let world_size = pg.world_size();let t = Tensor::zeros([3], DType::F32, Device::Cpu);

let is_main = pg.is_main();assert_eq!(t.device(), Device::Cpu);

``````



### Collectives#### `tensor.is_contiguous()` → `bool`

Check if tensor is contiguous in memory.

```rust

pg.broadcast(&mut tensor, root)?;```rust

pg.all_reduce(&mut tensor, ReduceOp::Sum)?;let t = Tensor::randn([2, 3], DType::F32, Device::Cpu);

pg.reduce(&mut tensor, dst, ReduceOp::Average)?;assert!(t.is_contiguous());

pg.barrier()?;

```let transposed = t.t()?;

assert!(!transposed.is_contiguous());

### DistributedDataParallel```



```rust---

let ddp = DistributedDataParallel::new(model, process_group);

let output = ddp.module().forward(&input)?;### Gradient Operations

ddp.sync_gradients(&mut gradients)?;

```#### `tensor.requires_grad()` → `bool`

Check if tensor tracks gradients.

### Convenience Functions

```rust

```rustlet t = Tensor::randn([3, 3], DType::F32, Device::Cpu);

let rank = get_rank();assert!(!t.requires_grad());

let world_size = get_world_size();```

let is_main = is_main_process();

barrier()?;#### `tensor.with_requires_grad(bool)` → `Tensor`

destroy_process_group();Set gradient tracking.

```

```rust

---let t = Tensor::randn([3, 3], DType::F32, Device::Cpu)

    .with_requires_grad(true);

## Autogradassert!(t.requires_grad());

```

### Backward Pass

#### `tensor.grad()` → `Option<Tensor>`

```rustGet accumulated gradient.

use ferrum::autograd::backward;

```rust

let loss = model.forward(&input)?;// After backward pass

backward(&loss)?;  // Compute gradientsif let Some(grad) = tensor.grad() {

```    println!("Gradient: {:?}", grad);

}

### No Gradient Context```



```rust#### `tensor.zero_grad()`

use ferrum::prelude::no_grad;Clear accumulated gradient.



let _guard = no_grad();```rust

// Operations here don't track gradientstensor.zero_grad();

let output = model.forward(&input)?;assert!(tensor.grad().is_none());

``````



### Gradient Tape---



```rust## Neural Network Layers

use ferrum::autograd::GradientTape;

### Linear Layer

let tape = GradientTape::new();

// Tape automatically records operationsFully connected layer: y = xW^T + b

```

```rust

---use ferrum::prelude::*;



## Serializationlet layer = Linear::new(784, 256);  // 784 inputs, 256 outputs



### Savelet x = Tensor::randn([32, 784], DType::F32, Device::Cpu);

let y = layer.forward(&x)?;  // Shape: [32, 256]

```rust```

use ferrum::serialize::save;

### Activation Layers

let state = model.state_dict();

save(&state, "model.ferrum")?;#### ReLU

```

```rust

### Loadlet relu = ReLU::new();

let y = relu.forward(&x)?;

```rust```

use ferrum::serialize::load;

#### Sigmoid

let state = load("model.ferrum")?;

model.load_state_dict(&state)?;```rust

```let sigmoid = Sigmoid::new();

let y = sigmoid.forward(&x)?;

### Formats```



| Format | Extension |#### Tanh

|--------|-----------|

| Binary | `.ferrum` |```rust

| JSON | `.json` |let tanh = Tanh::new();

let y = tanh.forward(&x)?;

---```



## Data Types### Sequential Container



| DType | Description |Chain multiple layers together.

|-------|-------------|

| `DType::F32` | 32-bit float (default) |```rust

| `DType::F64` | 64-bit float |let model = Sequential::new()

| `DType::I32` | 32-bit integer |    .add(Linear::new(784, 256))

| `DType::I64` | 64-bit integer |    .add(ReLU::new())

| `DType::U8` | 8-bit unsigned |    .add(Linear::new(256, 128))

    .add(ReLU::new())

---    .add(Linear::new(128, 10));



## Deviceslet output = model.forward(&input)?;

println!("Parameters: {}", model.num_parameters());

| Device | Description |```

|--------|-------------|

| `Device::Cpu` | CPU computation |---

| `Device::Cuda(id)` | CUDA GPU (simulated) |

## Loss Functions

---

### Mean Squared Error

## Error Handling

```rust

All operations return `Result<T, FerrumError>`.use ferrum::prelude::*;



```rustlet predictions = model.forward(&x)?;

use ferrum::prelude::*;let loss = mse_loss(&predictions, &targets)?;

```

fn main() -> Result<()> {

    let x = Tensor::randn([2, 3], DType::F32, Device::Cpu);### Binary Cross Entropy

    let y = x.add(&x)?;  // Propagate errors with ?

    Ok(())```rust

}let predictions = model.forward(&x)?.sigmoid()?;

```let loss = bce_loss(&predictions, &targets)?;

```

Common error types:

- `ShapeMismatch` - Incompatible tensor shapes### L1 Loss (MAE)

- `DTypeMismatch` - Incompatible data types

- `InvalidOperation` - Unsupported operation```rust

- `InternalError` - Internal errorlet loss = l1_loss(&predictions, &targets)?;

```

---

### Cross Entropy

## Prelude

```rust

Import common types:let log_probs = log_softmax(&logits, -1)?;

let loss = cross_entropy_loss(&log_probs, &targets)?;

```rust```

use ferrum::prelude::*;

---

// Includes:

// - Tensor, DType, Device, Shape## Optimizers

// - Linear, ReLU, Sigmoid, Tanh, Sequential

// - GELU, SiLU, LeakyReLU, ELU### SGD

// - LayerNorm, BatchNorm1d, Dropout

// - Softmax, LogSoftmaxStochastic Gradient Descent.

// - SGD, Adam, Optimizer

// - mse_loss, bce_loss, cross_entropy_loss, etc.```rust

// - DataLoader, Dataset, TensorDatasetuse ferrum::prelude::*;

// - Samplers

// - backward, no_gradlet params = model.parameters();

// - save, loadlet mut optimizer = SGD::new(params, 0.01);

// - init_process_group, DistributedDataParallel

```// In training loop:

optimizer.zero_grad();
// loss.backward()?;  // When autograd is complete
optimizer.step()?;
```

### SGD with Momentum

```rust
let mut optimizer = SGDMomentum::new(params, 0.01, 0.9);
```

### Adam

```rust
let mut optimizer = Adam::new(params, 0.001);
```

### AdamW

Adam with decoupled weight decay.

```rust
let mut optimizer = AdamW::new(params, 0.001, 0.01);
```

---

## Autograd

### Gradient Checking

Verify backward implementations numerically.

```rust
use ferrum_autograd::gradcheck::*;

let compute_loss = |x: &Tensor| -> Result<Tensor> {
    x.pow(2.0)?.sum()
};

let x = Tensor::randn([3, 4], DType::F32, Device::Cpu);
let analytical_grad = ...; // From backward()

quick_gradcheck(&compute_loss, &x, &analytical_grad)?;
```

---

## Serialization

### Save Model Weights

```rust
use ferrum::serialize::*;

let tensors = model.state_dict();
save(&tensors, "model.ferrum")?;
```

### Load Model Weights

```rust
let loaded = load("model.ferrum")?;
model.load_state_dict(&loaded)?;
```

---

## Data Types

```rust
DType::F32    // 32-bit float (most common)
DType::F64    // 64-bit float
DType::I32    // 32-bit integer
DType::I64    // 64-bit integer
DType::U8     // 8-bit unsigned integer
```

## Devices

```rust
Device::Cpu           // CPU execution
// Future: Device::Cuda(0), Device::Metal(0)
```

---

For more examples, see the `ferrum-examples` crate.
