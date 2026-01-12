# Contributing to FERRUM

Thank you for your interest in contributing to FERRUM! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Areas We Need Help](#areas-we-need-help)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Be respectful, constructive, and professional in all interactions.

---

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ferrum.git
   cd ferrum
   ```
3. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

---

## Development Setup

### Prerequisites

- Rust 1.75+ (stable)
- Cargo
- (Optional) CMake for CUDA builds

### Build the Project

```bash
# Build all crates
cargo build

# Build with release optimizations
cargo build --release

# Run tests
cargo test

# Run examples
cargo run --example simple_nn
```

### Development Tools

We recommend:
- **rust-analyzer** for IDE support
- **rustfmt** for code formatting
- **clippy** for linting

```bash
# Format code
cargo fmt

# Run linter
cargo clippy -- -D warnings
```

---

## Project Structure

```
ferrum/
├── ferrum-core/       # Tensor, Device, DType, Shape
├── ferrum-autograd/   # Autograd engine
├── ferrum-ops/        # Optimized operations
├── ferrum-nn/         # Neural network layers
├── ferrum-optim/      # Optimizers
├── ferrum-serialize/  # Model I/O
├── ferrum-examples/   # Examples and tutorials
└── ferrum/            # Main facade crate
```

Each crate is self-contained with:
- `src/` - Source code
- `tests/` - Integration tests
- `Cargo.toml` - Dependencies and metadata

---

## How to Contribute

### Reporting Bugs

Open an issue with:
- Clear description of the problem
- Minimal reproducible example
- Expected vs actual behavior
- System information (OS, Rust version)

### Suggesting Features

Open an issue with:
- Use case and motivation
- Proposed API design
- Implementation considerations
- Comparison with PyTorch/other frameworks

### Contributing Code

1. Check existing issues for work in progress
2. Discuss major changes in an issue first
3. Follow the coding standards below
4. Write tests for new functionality
5. Update documentation

---

## Coding Standards

### Rust Style

- Follow standard Rust formatting (`cargo fmt`)
- Use meaningful variable names
- Add doc comments for public APIs
- Minimize `unsafe` code and justify usage

### Documentation

All public items must have documentation:

```rust
/// Computes the matrix multiplication of two tensors.
///
/// # Arguments
///
/// * `other` - The right-hand side tensor
///
/// # Returns
///
/// A new tensor containing the result
///
/// # Example
///
/// ```
/// let a = Tensor::randn([2, 3], DType::F32, Device::Cpu);
/// let b = Tensor::randn([3, 4], DType::F32, Device::Cpu);
/// let c = a.matmul(&b)?;
/// ```
pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
    // ...
}
```

### Error Handling

- Use `Result<T>` for fallible operations
- Provide descriptive error messages
- Create specific error variants when needed

```rust
if shape_a[1] != shape_b[0] {
    return Err(FerrumError::shape_mismatch(
        "matmul",
        format!("[M, K] @ [K, N]"),
        format!("{:?} @ {:?}", shape_a, shape_b),
    ));
}
```

### Performance

- Avoid unnecessary allocations
- Use `&[T]` instead of `Vec<T>` for function arguments
- Prefer iterators over explicit loops
- Profile before optimizing

---

## Testing

### Unit Tests

Place tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros([2, 3], DType::F32, Device::Cpu);
        assert_eq!(t.shape(), &[2, 3]);
    }
}
```

### Integration Tests

Place in `tests/` directory:

```rust
// tests/integration_test.rs
use ferrum::prelude::*;

#[test]
fn test_end_to_end_training() {
    // ...
}
```

### Running Tests

```bash
# All tests
cargo test

# Specific crate
cargo test -p ferrum-core

# Specific test
cargo test test_tensor_creation

# With output
cargo test -- --nocapture
```

---

## Pull Request Process

1. **Update documentation** for API changes
2. **Add tests** for new functionality
3. **Ensure all tests pass**: `cargo test`
4. **Format code**: `cargo fmt`
5. **Run linter**: `cargo clippy`
6. **Update CHANGELOG.md** with your changes
7. **Submit PR** with clear description

### PR Description Template

```markdown
## Summary
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Added X
- Modified Y
- Fixed Z

## Testing
How was this tested?

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code formatted
- [ ] Clippy warnings addressed
```

---

## Areas We Need Help

### High Priority

1. **Complete Autograd Integration**
   - Connect tape recording to tensor operations
   - Implement backward() method
   - Test gradient flow end-to-end

2. **Performance Optimization**
   - Optimize matmul (BLAS integration)
   - Add SIMD kernels
   - Parallelize operations with Rayon

3. **Additional Layers**
   - Conv2D, Conv1D
   - BatchNorm, LayerNorm
   - Dropout
   - Embedding

### Medium Priority

4. **Loss Functions**
   - CrossEntropyLoss with logits
   - Focal Loss
   - Custom loss examples

5. **Data Loading**
   - Dataset trait
   - DataLoader with batching
   - Common dataset implementations

6. **GPU Backend**
   - CUDA kernel integration
   - Device memory management
   - cuBLAS for operations

### Low Priority

7. **Documentation**
   - More examples
   - Tutorial notebooks
   - API reference improvements

8. **Tooling**
   - Benchmarking suite
   - Profiling utilities
   - Debugging helpers

---

## Questions?

- Open an issue with the `question` label
- Join our discussions (when available)
- Read the documentation in `docs/`

---

## License

By contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0.

---

**Thank you for contributing to FERRUM! 🦀**
