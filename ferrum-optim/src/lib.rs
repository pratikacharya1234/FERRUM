//! # FERRUM Optim
//!
//! Optimization algorithms for training neural networks.
//!
//! ```rust,ignore
//! use ferrum_optim::{SGD, Adam, Optimizer};
//!
//! let optimizer = Adam::new(model.parameters(), AdamConfig::default());
//!
//! // Training loop
//! optimizer.zero_grad();
//! let loss = model.forward(&input)?.sum()?;
//! loss.backward()?;
//! optimizer.step()?;
//! ```

pub mod adam;
pub mod amp;
pub mod optimizer;
pub mod scheduler;
pub mod sgd;

pub mod prelude {
    pub use crate::adam::{Adam, AdamConfig};
    pub use crate::amp::{GradScaler, Autocast, to_half, to_bfloat16, to_float, has_inf_or_nan};
    pub use crate::optimizer::Optimizer;
    pub use crate::scheduler::{
        LRScheduler, StepLR, MultiStepLR, ExponentialLR, 
        CosineAnnealingLR, CosineAnnealingWarmRestarts,
        LinearWarmupLR, OneCycleLR, ReduceLROnPlateau, PlateauMode,
    };
    pub use crate::sgd::{SGDConfig, SGD};
}

pub use prelude::*;
