//! # FERRUM Serialize
//!
//! Model serialization and deserialization using safetensors-compatible format.
//!
//! ```rust,ignore
//! use ferrum_serialize::{save, load};
//!
//! // Save model weights
//! save(&model.parameters(), "model.ferrum")?;
//!
//! // Load model weights
//! let weights = load("model.ferrum")?;
//! ```

pub mod format;
pub mod io;

pub use io::{load, save};
