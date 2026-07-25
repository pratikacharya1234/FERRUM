#!/bin/bash
# FERRUM Colab Benchmark Runner
# Run this on Google Colab with GPU runtime enabled.
#
# Usage:
#   1. Upload this file and FERRUM to Colab
#   2. !bash run_colab.sh
#
# Or paste cells into a Colab notebook.

set -e

echo "=== FERRUM GPU Benchmark Setup ==="

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: No GPU detected. Go to Runtime → Change runtime type → GPU"
    exit 1
fi

echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# Install Rust if not present
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "--- Rust Version ---"
rustc --version
cargo --version
echo ""

# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# Check CUDA
echo "--- CUDA ---"
if command -v nvcc &> /dev/null; then
    nvcc --version | tail -1
else
    echo "nvcc not found, trying nvidia-smi..."
    nvidia-smi | head -5
fi
echo ""

# Navigate to FERRUM directory
cd /content/FERRUM 2>/dev/null || cd "$(dirname "$0")"

echo "=== Compiling FERRUM (release mode) ==="
cargo build --example bench --example gpu_smoke --example cifar10 --release 2>&1 | tail -5

echo ""
echo "=== 1/3 GPU dispatch smoke test (correctness gate) ==="
# Compares every GPU-dispatched op against its CPU result.
# If this fails, do NOT trust the training numbers below.
cargo run --example gpu_smoke --release 2>&1

echo ""
echo "=== 2/3 Raw kernel benchmarks ==="
FERRUM_PROFILE=1 cargo run --example bench --release 2>&1

echo ""
echo "=== 3/3 CIFAR-10 training on GPU ==="
# Auto-detects CUDA. Uses bundled data when present; otherwise downloads
# from the Toronto origin (resumable, with progress — can take a while).
if [ -d data/cifar-10-batches-bin ] && [ -f data/cifar-10-batches-bin/data_batch_1.bin ]; then
    echo "Using bundled CIFAR-10 data."
else
    echo "Downloading CIFAR-10 (~170 MB, resumable)..."
    mkdir -p data
    curl -L -C - --progress-bar -o cifar-10-binary.tar.gz \
        https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    tar xzf cifar-10-binary.tar.gz -C data
fi
cargo run --example cifar10 --release 2>&1

echo ""
echo "=== Done ==="
echo "Paste the above output for analysis."
