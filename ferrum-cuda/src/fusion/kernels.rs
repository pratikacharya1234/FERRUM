//! Pre-written Fused High-Performance Kernels.
//!
//! These kernels are hand-optimized for specific fusion patterns
//! that appear frequently in transformers and CNNs. They are compiled
//! at runtime via NVRTC.

/// Source code for fused LayerNorm + GELU + Linear kernel.
///
/// This kernel fuses three operations that appear after every attention
/// block in transformers. Currently: 3 kernel launches + 3 memory
/// round-trips. Fused: 1 kernel, 1 memory read, 1 memory write.
pub const FUSED_LAYERNORM_GELU_LINEAR: &str = r#"
extern "C" __global__ void fused_layernorm_gelu_linear(
    const float* __restrict__ input,      // [B, S, D]
    const float* __restrict__ gamma,      // [D]
    const float* __restrict__ beta,       // [D]
    const float* __restrict__ weight,     // [D, 4D] for FFN expansion
    const float* __restrict__ bias,       // [4D]
    float* __restrict__ output,           // [B, S, 4D]
    int B, int S, int D, int D4
) {
    extern __shared__ float smem[];
    float* row_norm = smem;       // [D]
    float* row_gelu = smem + D;   // [D]
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_size = D;
    
    if (row >= B * S) return;
    
    const float* row_in = input + row * D;
    
    // 1. Load row, compute mean/var
    float mean = 0.0f, var = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x) {
        float x = row_in[i];
        mean += x;
        var += x * x;
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);
        var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
    }
    
    // Block-level reduction via shared memory
    __shared__ float s_mean[32];
    __shared__ float s_var[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        s_mean[warp_id] = mean;
        s_var[warp_id] = var;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        mean = (lane_id < (blockDim.x + 31) / 32) ? s_mean[lane_id] : 0.0f;
        var = (lane_id < (blockDim.x + 31) / 32) ? s_var[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);
            var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
        }
    }
    __syncthreads();
    
    mean /= row_size;
    var = var / row_size - mean * mean;
    float inv_std = rsqrtf(var + 1e-5f);
    
    // 2. Normalize + GELU
    for (int i = tid; i < row_size; i += blockDim.x) {
        float x = row_in[i];
        float norm = (x - mean) * inv_std * gamma[i] + beta[i];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (norm + 0.044715f * norm * norm * norm)));
        row_gelu[i] = norm * cdf;
    }
    __syncthreads();
    
    // 3. Linear: output = gelu_row @ weight + bias
    for (int j = tid; j < D4; j += blockDim.x) {
        float sum = bias[j];
        for (int i = 0; i < row_size; i++) {
            sum += row_gelu[i] * weight[i * D4 + j];
        }
        output[row * D4 + j] = sum;
    }
}
"#;

/// Source code for fused Softmax kernel.
///
/// Uses warp-level reductions for better performance than PyTorch's
/// standard softmax implementation.
pub const FUSED_SOFTMAX: &str = r#"
extern "C" __global__ void fused_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols
) {
    int row = blockIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // 1. Find max (warp shuffle)
    float max_val = -INFINITY;
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        max_val = fmaxf(max_val, row_in[i]);
    }
    
    // Warp reduce max
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }
    
    // Block reduce max via shared memory
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp] = max_val;
    __syncthreads();
    
    if (warp == 0) {
        max_val = (lane < num_warps) ? s_max[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }
    }
    __syncthreads();
    
    // 2. Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        float exp_val = expf(row_in[i] - max_val);
        row_out[i] = exp_val;
        sum += exp_val;
    }
    
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Block reduce sum
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();
    
    if (warp == 0) {
        sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }
    }
    __syncthreads();
    
    // 3. Divide by sum
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        row_out[i] /= sum;
    }
}
"#;

/// Source code for fused Bias + GELU kernel.
pub const FUSED_BIAS_GELU: &str = r#"
extern "C" __global__ void fused_bias_gelu(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n, int bias_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int bias_idx = idx % bias_size;
    float x = input[idx] + bias[bias_idx];
    float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[idx] = x * cdf;
}
"#;

/// Source code for fused Bias + ReLU kernel.
pub const FUSED_BIAS_RELU: &str = r#"
extern "C" __global__ void fused_bias_relu(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int n, int bias_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int bias_idx = idx % bias_size;
    float x = input[idx] + bias[bias_idx];
    output[idx] = fmaxf(x, 0.0f);
}
"#;

/// Source code for fused MatMul + Bias kernel.
pub const FUSED_MATMUL_BIAS: &str = r#"
extern "C" __global__ void fused_matmul_bias(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = bias[col];
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
"#;

/// Source code for fused Scale + Softmax kernel (for attention).
pub const FUSED_SCALE_SOFTMAX: &str = r#"
extern "C" __global__ void fused_scale_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows, int cols, float scale
) {
    int row = blockIdx.x;
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;
    
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // 1. Find max (scale + max)
    float max_val = -INFINITY;
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        max_val = fmaxf(max_val, row_in[i] * scale);
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
    }
    
    __shared__ float s_max[32];
    if (lane == 0) s_max[warp] = max_val;
    __syncthreads();
    
    if (warp == 0) {
        max_val = (lane < num_warps) ? s_max[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset /= 2) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xFFFFFFFF, max_val, offset));
        }
    }
    __syncthreads();
    
    // 2. Compute exp(x*scale - max) and sum
    float sum = 0.0f;
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        float exp_val = expf(row_in[i] * scale - max_val);
        row_out[i] = exp_val;
        sum += exp_val;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
    }
    
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp] = sum;
    __syncthreads();
    
    if (warp == 0) {
        sum = (lane < num_warps) ? s_sum[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset);
        }
    }
    __syncthreads();
    
    // 3. Divide
    for (int i = lane + warp * 32; i < cols; i += num_warps * 32) {
        row_out[i] /= sum;
    }
}
"#;

/// Source code for fused LayerNorm kernel.
pub const FUSED_LAYERNORM: &str = r#"
extern "C" __global__ void fused_layernorm(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    int rows, int cols, float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row >= rows) return;
    
    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;
    
    // Compute mean
    float mean = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        mean += row_in[i];
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);
    }
    
    __shared__ float s_mean[32];
    int warp = tid / 32;
    int lane = tid % 32;
    if (lane == 0) s_mean[warp] = mean;
    __syncthreads();
    
    if (warp == 0) {
        mean = (lane < (blockDim.x + 31) / 32) ? s_mean[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            mean += __shfl_xor_sync(0xFFFFFFFF, mean, offset);
        }
    }
    __syncthreads();
    mean /= cols;
    
    // Compute variance
    float var = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float diff = row_in[i] - mean;
        var += diff * diff;
    }
    
    for (int offset = 16; offset > 0; offset /= 2) {
        var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
    }
    
    __shared__ float s_var[32];
    if (lane == 0) s_var[warp] = var;
    __syncthreads();
    
    if (warp == 0) {
        var = (lane < (blockDim.x + 31) / 32) ? s_var[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            var += __shfl_xor_sync(0xFFFFFFFF, var, offset);
        }
    }
    __syncthreads();
    var /= cols;
    
    float inv_std = rsqrtf(var + eps);
    
    // Normalize
    for (int i = tid; i < cols; i += blockDim.x) {
        row_out[i] = (row_in[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}
"#;

/// Get all blessed kernel sources as (name, source) pairs.
pub fn blessed_kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("fused_layernorm_gelu_linear", FUSED_LAYERNORM_GELU_LINEAR),
        ("fused_softmax", FUSED_SOFTMAX),
        ("fused_bias_gelu", FUSED_BIAS_GELU),
        ("fused_bias_relu", FUSED_BIAS_RELU),
        ("fused_matmul_bias", FUSED_MATMUL_BIAS),
        ("fused_scale_softmax", FUSED_SCALE_SOFTMAX),
        ("fused_layernorm", FUSED_LAYERNORM),
    ]
}
