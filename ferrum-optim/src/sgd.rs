//! Stochastic Gradient Descent optimizer.

use ferrum_core::{Result, Tensor};

use crate::optimizer::Optimizer;

/// SGD configuration.
#[derive(Debug, Clone)]
pub struct SGDConfig {
    /// Learning rate.
    pub lr: f64,
    /// Momentum factor.
    pub momentum: f64,
    /// Weight decay (L2 penalty).
    pub weight_decay: f64,
    /// Enable Nesterov momentum.
    pub nesterov: bool,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }
}

impl SGDConfig {
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            ..Default::default()
        }
    }

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

/// Stochastic Gradient Descent optimizer.
///
/// Implements SGD with optional momentum and weight decay:
///
/// v = momentum * v + grad + weight_decay * param
/// param = param - lr * v
pub struct SGD {
    params: Vec<Tensor>,
    config: SGDConfig,
    /// Velocity buffers for momentum.
    velocities: Vec<Option<Tensor>>,
}

impl SGD {
    /// Create new SGD optimizer.
    pub fn new(params: Vec<Tensor>, config: SGDConfig) -> Self {
        let n = params.len();
        Self {
            params,
            config,
            velocities: vec![None; n],
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        let lr = self.config.lr;
        let momentum = self.config.momentum;
        let weight_decay = self.config.weight_decay;

        for (i, param) in self.params.iter().enumerate() {
            if let Some(grad) = param.grad() {
                // Apply weight decay
                let grad = if weight_decay != 0.0 {
                    grad.add(&param.mul_scalar(weight_decay)?)?
                } else {
                    grad
                };

                // Apply momentum
                let update = if momentum != 0.0 {
                    let velocity = if let Some(ref v) = self.velocities[i] {
                        v.mul_scalar(momentum)?.add(&grad)?
                    } else {
                        grad.clone()
                    };
                    self.velocities[i] = Some(velocity.clone());

                    if self.config.nesterov {
                        grad.add(&velocity.mul_scalar(momentum)?)?
                    } else {
                        velocity
                    }
                } else {
                    grad
                };

                // Update parameter: param = param - lr * update
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
    fn test_sgd_config() {
        let config = SGDConfig::new(0.1)
            .with_momentum(0.9)
            .with_weight_decay(1e-4);

        assert_eq!(config.lr, 0.1);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.weight_decay, 1e-4);
    }
}
