//! Learning rate schedulers.
//!
//! Schedulers adjust the learning rate during training to improve convergence.

use std::f64::consts::PI;

/// Trait for learning rate schedulers.
pub trait LRScheduler: Send + Sync {
    /// Get the current learning rate.
    fn get_lr(&self) -> f64;
    
    /// Update the scheduler state (typically called after each epoch or step).
    fn step(&mut self);
    
    /// Get the current step/epoch count.
    fn current_step(&self) -> usize;
    
    /// Reset the scheduler to initial state.
    fn reset(&mut self);
}

/// Step learning rate decay.
/// 
/// Decays the learning rate by `gamma` every `step_size` epochs.
/// 
/// # Example
/// ```ignore
/// let scheduler = StepLR::new(0.1, 30, 0.1);  // LR 0.1, decay by 0.1x every 30 epochs
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    base_lr: f64,
    step_size: usize,
    gamma: f64,
    current_epoch: usize,
}

impl StepLR {
    /// Create a new StepLR scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `step_size` - Period of learning rate decay (epochs)
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self) -> f64 {
        let num_decays = self.current_epoch / self.step_size;
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
    
    fn step(&mut self) {
        self.current_epoch += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch
    }
    
    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Multi-step learning rate decay.
/// 
/// Decays the learning rate by `gamma` once the number of epochs reaches one
/// of the milestones.
/// 
/// # Example
/// ```ignore
/// let scheduler = MultiStepLR::new(0.1, vec![30, 80], 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    base_lr: f64,
    milestones: Vec<usize>,
    gamma: f64,
    current_epoch: usize,
}

impl MultiStepLR {
    /// Create a new MultiStepLR scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `milestones` - List of epoch indices where LR is decayed
    /// * `gamma` - Multiplicative factor of learning rate decay
    pub fn new(base_lr: f64, milestones: Vec<usize>, gamma: f64) -> Self {
        let mut milestones = milestones;
        milestones.sort();
        
        Self {
            base_lr,
            milestones,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for MultiStepLR {
    fn get_lr(&self) -> f64 {
        let num_decays = self.milestones.iter()
            .filter(|&&m| m <= self.current_epoch)
            .count();
        self.base_lr * self.gamma.powi(num_decays as i32)
    }
    
    fn step(&mut self) {
        self.current_epoch += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch
    }
    
    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Exponential learning rate decay.
/// 
/// Decays the learning rate by `gamma` every epoch.
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    base_lr: f64,
    gamma: f64,
    current_epoch: usize,
}

impl ExponentialLR {
    /// Create a new ExponentialLR scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `gamma` - Multiplicative factor of learning rate decay per epoch
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self) -> f64 {
        self.base_lr * self.gamma.powi(self.current_epoch as i32)
    }
    
    fn step(&mut self) {
        self.current_epoch += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch
    }
    
    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Cosine annealing learning rate scheduler.
/// 
/// Sets the learning rate using a cosine annealing schedule.
/// The learning rate decreases from `base_lr` to `eta_min` over `T_max` steps.
/// 
/// # Example
/// ```ignore
/// let scheduler = CosineAnnealingLR::new(0.1, 100, 1e-6);  // 100 epochs to anneal
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    base_lr: f64,
    t_max: usize,
    eta_min: f64,
    current_epoch: usize,
}

impl CosineAnnealingLR {
    /// Create a new CosineAnnealingLR scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `t_max` - Maximum number of iterations/epochs
    /// * `eta_min` - Minimum learning rate
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_max,
            eta_min,
            current_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self) -> f64 {
        // η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T_max))
        let t = self.current_epoch.min(self.t_max) as f64;
        let t_max = self.t_max as f64;
        
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * t / t_max).cos())
    }
    
    fn step(&mut self) {
        self.current_epoch += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch
    }
    
    fn reset(&mut self) {
        self.current_epoch = 0;
    }
}

/// Cosine annealing with warm restarts.
/// 
/// The learning rate is reset to the initial value periodically.
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    base_lr: f64,
    t_0: usize,
    t_mult: usize,
    eta_min: f64,
    current_epoch: usize,
    current_cycle: usize,
    current_t_i: usize,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new CosineAnnealingWarmRestarts scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `t_0` - Number of iterations for the first restart
    /// * `t_mult` - Factor to increase T_i after each restart
    /// * `eta_min` - Minimum learning rate
    pub fn new(base_lr: f64, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_0,
            t_mult,
            eta_min,
            current_epoch: 0,
            current_cycle: 0,
            current_t_i: t_0,
        }
    }
}

impl LRScheduler for CosineAnnealingWarmRestarts {
    fn get_lr(&self) -> f64 {
        // Calculate position within current cycle
        let t_cur = self.current_epoch % self.current_t_i;
        let t_i = self.current_t_i as f64;
        let t = t_cur as f64;
        
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + (PI * t / t_i).cos())
    }
    
    fn step(&mut self) {
        self.current_epoch += 1;
        
        // Check if we should restart
        if self.current_epoch >= self.current_t_i {
            self.current_epoch = 0;
            self.current_cycle += 1;
            if self.t_mult > 1 {
                self.current_t_i *= self.t_mult;
            }
        }
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch + self.current_cycle * self.t_0
    }
    
    fn reset(&mut self) {
        self.current_epoch = 0;
        self.current_cycle = 0;
        self.current_t_i = self.t_0;
    }
}

/// Linear learning rate warmup followed by decay.
/// 
/// Gradually increases learning rate from 0 to base_lr during warmup,
/// then decays linearly to 0.
#[derive(Debug, Clone)]
pub struct LinearWarmupLR {
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LinearWarmupLR {
    /// Create a new LinearWarmupLR scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Peak learning rate after warmup
    /// * `warmup_steps` - Number of warmup steps
    /// * `total_steps` - Total number of training steps
    pub fn new(base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }
}

impl LRScheduler for LinearWarmupLR {
    fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Linear decay
            let remaining = self.total_steps - self.current_step;
            let decay_steps = self.total_steps - self.warmup_steps;
            self.base_lr * (remaining as f64 / decay_steps as f64).max(0.0)
        }
    }
    
    fn step(&mut self) {
        self.current_step += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_step
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// One Cycle learning rate policy.
/// 
/// Uses a triangular learning rate schedule with optional annealing.
/// Popular for training with super-convergence.
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    max_lr: f64,
    total_steps: usize,
    pct_start: f64,
    div_factor: f64,
    final_div_factor: f64,
    current_step: usize,
}

impl OneCycleLR {
    /// Create a new OneCycleLR scheduler.
    /// 
    /// # Arguments
    /// * `max_lr` - Upper learning rate boundary
    /// * `total_steps` - Total number of training steps
    /// * `pct_start` - Percentage of steps spent increasing LR (default 0.3)
    /// * `div_factor` - Initial LR = max_lr / div_factor (default 25)
    /// * `final_div_factor` - Final LR = initial_lr / final_div_factor (default 1e4)
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        div_factor: f64,
        final_div_factor: f64,
    ) -> Self {
        Self {
            max_lr,
            total_steps,
            pct_start,
            div_factor,
            final_div_factor,
            current_step: 0,
        }
    }
    
    /// Create with default parameters.
    pub fn new_default(max_lr: f64, total_steps: usize) -> Self {
        Self::new(max_lr, total_steps, 0.3, 25.0, 1e4)
    }
}

impl LRScheduler for OneCycleLR {
    fn get_lr(&self) -> f64 {
        let initial_lr = self.max_lr / self.div_factor;
        let final_lr = initial_lr / self.final_div_factor;
        
        let pct = self.current_step as f64 / self.total_steps as f64;
        let up_steps = (self.total_steps as f64 * self.pct_start) as usize;
        
        if self.current_step < up_steps {
            // Warmup phase: linear increase
            let up_pct = self.current_step as f64 / up_steps as f64;
            initial_lr + (self.max_lr - initial_lr) * up_pct
        } else {
            // Annealing phase: cosine decay
            let down_pct = (self.current_step - up_steps) as f64 / 
                           (self.total_steps - up_steps) as f64;
            let cosine_val = (1.0 + (PI * down_pct).cos()) / 2.0;
            final_lr + (self.max_lr - final_lr) * cosine_val
        }
    }
    
    fn step(&mut self) {
        self.current_step = (self.current_step + 1).min(self.total_steps);
    }
    
    fn current_step(&self) -> usize {
        self.current_step
    }
    
    fn reset(&mut self) {
        self.current_step = 0;
    }
}

/// Reduce learning rate when a metric has stopped improving.
/// 
/// Similar to PyTorch's ReduceLROnPlateau.
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    base_lr: f64,
    current_lr: f64,
    factor: f64,
    patience: usize,
    min_lr: f64,
    mode: PlateauMode,
    threshold: f64,
    best: f64,
    num_bad_epochs: usize,
    current_epoch: usize,
}

/// Mode for ReduceLROnPlateau.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlateauMode {
    /// Reduce when metric stops decreasing
    Min,
    /// Reduce when metric stops increasing
    Max,
}

impl ReduceLROnPlateau {
    /// Create a new ReduceLROnPlateau scheduler.
    /// 
    /// # Arguments
    /// * `base_lr` - Initial learning rate
    /// * `mode` - 'min' or 'max' mode
    /// * `factor` - Factor by which to reduce LR (default 0.1)
    /// * `patience` - Number of epochs to wait before reducing (default 10)
    /// * `min_lr` - Minimum learning rate (default 0)
    pub fn new(
        base_lr: f64,
        mode: PlateauMode,
        factor: f64,
        patience: usize,
        min_lr: f64,
    ) -> Self {
        let best = match mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        
        Self {
            base_lr,
            current_lr: base_lr,
            factor,
            patience,
            min_lr,
            mode,
            threshold: 1e-4,
            best,
            num_bad_epochs: 0,
            current_epoch: 0,
        }
    }
    
    /// Update based on a metric value.
    pub fn step_with_metric(&mut self, metric: f64) {
        self.current_epoch += 1;
        
        let is_better = match self.mode {
            PlateauMode::Min => metric < self.best * (1.0 - self.threshold),
            PlateauMode::Max => metric > self.best * (1.0 + self.threshold),
        };
        
        if is_better {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
            
            if self.num_bad_epochs >= self.patience {
                self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                self.num_bad_epochs = 0;
            }
        }
    }
}

impl LRScheduler for ReduceLROnPlateau {
    fn get_lr(&self) -> f64 {
        self.current_lr
    }
    
    fn step(&mut self) {
        // For this scheduler, use step_with_metric instead
        self.current_epoch += 1;
    }
    
    fn current_step(&self) -> usize {
        self.current_epoch
    }
    
    fn reset(&mut self) {
        self.current_lr = self.base_lr;
        self.best = match self.mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        self.num_bad_epochs = 0;
        self.current_epoch = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_step_lr() {
        let mut scheduler = StepLR::new(0.1, 10, 0.1);
        
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
        
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-10);
    }
    
    #[test]
    fn test_multi_step_lr() {
        let mut scheduler = MultiStepLR::new(0.1, vec![5, 15], 0.1);
        
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
        
        for _ in 0..10 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.001).abs() < 1e-10);
    }
    
    #[test]
    fn test_cosine_annealing() {
        let mut scheduler = CosineAnnealingLR::new(0.1, 100, 0.0);
        
        // At start, should be at max LR
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        
        // At T_max/2, should be at (max + min) / 2
        for _ in 0..50 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-10);
        
        // At T_max, should be at min LR
        for _ in 0..50 {
            scheduler.step();
        }
        assert!(scheduler.get_lr() < 0.001);
    }
    
    #[test]
    fn test_linear_warmup() {
        let mut scheduler = LinearWarmupLR::new(0.1, 10, 100);
        
        // At start, should be 0
        assert!(scheduler.get_lr() < 1e-10);
        
        // Halfway through warmup
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.05).abs() < 1e-10);
        
        // After warmup
        for _ in 0..5 {
            scheduler.step();
        }
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
    }
    
    #[test]
    fn test_one_cycle() {
        let scheduler = OneCycleLR::new_default(0.1, 100);
        
        // Initial LR should be max_lr / div_factor
        assert!((scheduler.get_lr() - 0.004).abs() < 0.001);
    }
    
    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = ReduceLROnPlateau::new(0.1, PlateauMode::Min, 0.1, 2, 1e-6);
        
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        
        // Improve metric
        scheduler.step_with_metric(1.0);
        assert!((scheduler.get_lr() - 0.1).abs() < 1e-10);
        
        // No improvement for patience epochs
        scheduler.step_with_metric(1.0);
        scheduler.step_with_metric(1.0);
        
        // Should have reduced
        assert!((scheduler.get_lr() - 0.01).abs() < 1e-10);
    }
}
