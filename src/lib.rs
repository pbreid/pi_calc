//! High-performance, verifiable arbitrary-precision computation of π.
//!
//! This crate implements the Chudnovsky formula with binary splitting using
//! GMP big integers (`rug`), parallelised across all available cores, with a
//! subquadratic binary→decimal converter and two independent verification
//! layers (BBP hex digit checks and externally-sourced decimal checkpoints).

pub mod bbp;
pub mod chudnovsky;
pub mod convert;
pub mod output;
pub mod verify;

// Re-export a small public API surface for library consumers and tests.
pub use chudnovsky::{compute_pi, PiConfig, PiResult};
