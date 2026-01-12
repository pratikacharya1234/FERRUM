//! Adam optimizer.

use ferrum_core::{Result, Tensor};

use crate::optimizer::Optimizer;

/// Adam configuration.
#[derive(Debug, Clone)]
pub struct AdamConfig {
    /// Learning rate.
    pub lr: f64,
    /// Exponential decay rate for first moment.
    pub beta1: f64,
    /// Exponential decay rate for second moment.
    pub beta2: f64,
    /// Term added for numerical stability.
    pub eps: f64,
    /// Weight decay (decoupled).
    pub weight_decay: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

impl AdamConfig {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

/// Adam optimizer (Adaptive Moment Estimation).
///
/// Implements Adam with optional decoupled weight decay (AdamW):
///
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// param = param - lr * m_hat / (sqrt(v_hat) + eps)
///
/// Reference: Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014)
pub struct Adam {
    params: Vec<Tensor>,
    config: AdamConfig,
    /// First moment estimates.
    m: Vec<Option<Tensor>>,
    /// Second moment estimates.
    v: Vec<Option<Tensor>>,
    /// Step counter for bias correction.
    step_count: u64,
}

impl Adam {
    /// Create new Adam optimizer.
    pub fn new(params: Vec<Tensor>, config: AdamConfig) -> Self {
        let n = params.len();
        Self {
            params,
            config,
            m: vec![None; n],
            v: vec![None; n],
            step_count: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as f64;

        let lr = self.config.lr;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let weight_decay = self.config.weight_decay;

        // Bias correction terms
        let bias_correction1 = 1.0 - beta1.powi(t as i32);
        let bias_correction2 = 1.0 - beta2.powi(t as i32);

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // Decoupled weight decay (AdamW style)
                if weight_decay != 0.0 {
                    let decay = param.mul_scalar(lr * weight_decay)?;
                    param.sub_inplace(&decay)?;
                }

                // Update first moment: m = beta1 * m + (1 - beta1) * grad
                let m = if let Some(ref prev_m) = self.m[i] {
                    prev_m
                        .mul_scalar(beta1)?
                        .add(&grad.mul_scalar(1.0 - beta1)?)?
                } else {
                    grad.mul_scalar(1.0 - beta1)?
                };
                self.m[i] = Some(m.clone());

                // Update second moment: v = beta2 * v + (1 - beta2) * grad^2
                let grad_sq = grad.pow(2.0)?;
                let v = if let Some(ref prev_v) = self.v[i] {
                    prev_v
                        .mul_scalar(beta2)?
                        .add(&grad_sq.mul_scalar(1.0 - beta2)?)?
                } else {
                    grad_sq.mul_scalar(1.0 - beta2)?
                };
                self.v[i] = Some(v.clone());

                // Bias-corrected estimates
                let m_hat = m.mul_scalar(1.0 / bias_correction1)?;
                let v_hat = v.mul_scalar(1.0 / bias_correction2)?;

                // Compute update: lr * m_hat / (sqrt(v_hat) + eps)
                let update = m_hat.div(&v_hat.sqrt()?.add_scalar(eps)?)?;
                let scaled_update = update.mul_scalar(lr)?;
                param.sub_inplace(&scaled_update)?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            param.zero_grad();
        }
    }

    fn learning_rate(&self) -> f64 {
        self.config.lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.config.lr = lr;
    }

    fn param_groups(&self) -> &[Vec<Tensor>] {
        std::slice::from_ref(&self.params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adam_config() {
        let config = AdamConfig::new(0.001)
            .with_betas(0.9, 0.999)
            .with_weight_decay(1e-4);

        assert_eq!(config.lr, 0.001);
        assert_eq!(config.beta1, 0.9);
        assert_eq!(config.beta2, 0.999);
        assert_eq!(config.weight_decay, 1e-4);
    }
}
