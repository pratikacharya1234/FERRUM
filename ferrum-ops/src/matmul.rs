//! Matrix multiplication kernels with optional BLAS acceleration.

use ferrum_core::{Result, Tensor, FerrumError};
use rayon::prelude::*;

/// General matrix multiplication: C = A @ B
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.matmul(b)
}

/// Optimized matrix multiplication with tiling for cache efficiency.
/// Falls back to BLAS when the `openblas` feature is enabled.
pub fn matmul_optimized(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "openblas")]
    {
        return matmul_blas(a, b);
    }
    
    #[cfg(not(feature = "openblas"))]
    {
        return matmul_tiled(a, b);
    }
}

/// Tiled matrix multiplication for better cache utilization.
/// Uses blocking to improve L1/L2 cache hits.
pub fn matmul_tiled(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return a.matmul(b); // Fall back to standard matmul for batched
    }
    
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    
    if k != b_shape[0] {
        return Err(FerrumError::ShapeMismatch {
            operation: "matmul_tiled",
            expected: format!("[{}, {}]", m, k),
            actual: format!("[{}, {}]", b_shape[0], b_shape[1]),
        });
    }
    
    let a_data = a.to_vec::<f32>()?;
    let b_data = b.to_vec::<f32>()?;
    let mut c_data = vec![0.0f32; m * n];
    
    // Tile size tuned for typical L1 cache (32KB)
    const TILE_SIZE: usize = 64;
    
    // Blocked matrix multiplication
    for i0 in (0..m).step_by(TILE_SIZE) {
        for j0 in (0..n).step_by(TILE_SIZE) {
            for k0 in (0..k).step_by(TILE_SIZE) {
                let i_end = (i0 + TILE_SIZE).min(m);
                let j_end = (j0 + TILE_SIZE).min(n);
                let k_end = (k0 + TILE_SIZE).min(k);
                
                for i in i0..i_end {
                    for kk in k0..k_end {
                        let a_val = a_data[i * k + kk];
                        for j in j0..j_end {
                            c_data[i * n + j] += a_val * b_data[kk * n + j];
                        }
                    }
                }
            }
        }
    }
    
    Tensor::from_slice(&c_data, [m, n], a.device())
}

/// Parallel tiled matrix multiplication using rayon.
pub fn matmul_parallel(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return a.matmul(b);
    }
    
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    
    if k != b_shape[0] {
        return Err(FerrumError::ShapeMismatch {
            operation: "matmul_parallel",
            expected: format!("[{}, {}]", m, k),
            actual: format!("[{}, {}]", b_shape[0], b_shape[1]),
        });
    }
    
    let a_data = a.to_vec::<f32>()?;
    let b_data = b.to_vec::<f32>()?;
    
    // Parallel over rows
    let c_data: Vec<f32> = (0..m)
        .into_par_iter()
        .flat_map(|i| {
            let mut row = vec![0.0f32; n];
            for kk in 0..k {
                let a_val = a_data[i * k + kk];
                for j in 0..n {
                    row[j] += a_val * b_data[kk * n + j];
                }
            }
            row
        })
        .collect();
    
    Tensor::from_slice(&c_data, [m, n], a.device())
}

/// BLAS-accelerated matrix multiplication (when openblas feature is enabled).
#[cfg(feature = "openblas")]
pub fn matmul_blas(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    use blas::sgemm;
    
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return a.matmul(b);
    }
    
    let m = a_shape[0] as i32;
    let k = a_shape[1] as i32;
    let n = b_shape[1] as i32;
    
    if k != b_shape[0] as i32 {
        return Err(FerrumError::ShapeMismatch {
            operation: "matmul_blas",
            expected: format!("[{}, {}]", m, k),
            actual: format!("[{}, {}]", b_shape[0], b_shape[1]),
        });
    }
    
    let a_data = a.to_vec::<f32>()?;
    let b_data = b.to_vec::<f32>()?;
    let mut c_data = vec![0.0f32; (m * n) as usize];
    
    // sgemm: C = alpha * A * B + beta * C
    unsafe {
        sgemm(
            b'N',      // No transpose A
            b'N',      // No transpose B
            n,         // Columns of C
            m,         // Rows of C  
            k,         // Inner dimension
            1.0,       // alpha
            &b_data,   // B matrix (column-major, so we swap)
            n,         // Leading dimension of B
            &a_data,   // A matrix
            k,         // Leading dimension of A
            0.0,       // beta
            &mut c_data, // C matrix
            n,         // Leading dimension of C
        );
    }
    
    Tensor::from_slice(&c_data, [m as usize, n as usize], a.device())
}

#[cfg(not(feature = "openblas"))]
pub fn matmul_blas(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Fallback when BLAS is not available
    matmul_tiled(a, b)
}

/// Batched matrix multiplication.
pub fn bmm(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.matmul(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_core::{Device, DType};
    
    #[test]
    fn test_matmul_tiled() {
        let a = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            Device::Cpu,
        ).unwrap();
        let b = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            Device::Cpu,
        ).unwrap();
        
        let c = matmul_tiled(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        
        let data = c.to_vec::<f32>().unwrap();
        assert!((data[0] - 22.0).abs() < 1e-5);
        assert!((data[1] - 28.0).abs() < 1e-5);
        assert!((data[2] - 49.0).abs() < 1e-5);
        assert!((data[3] - 64.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_matmul_parallel() {
        let a = Tensor::randn([64, 128], DType::F32, Device::Cpu);
        let b = Tensor::randn([128, 64], DType::F32, Device::Cpu);
        
        let c1 = matmul(&a, &b).unwrap();
        let c2 = matmul_parallel(&a, &b).unwrap();
        
        let d1 = c1.to_vec::<f32>().unwrap();
        let d2 = c2.to_vec::<f32>().unwrap();
        
        for (v1, v2) in d1.iter().zip(d2.iter()) {
            assert!((v1 - v2).abs() < 1e-4);
        }
    }
}
