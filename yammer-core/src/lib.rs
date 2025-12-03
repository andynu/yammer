//! yammer-core: shared types, configuration, and model management
//!
//! This crate provides the foundational types and utilities used across
//! the yammer application.

pub mod config;
pub mod error;
pub mod model;

pub use config::Config;
pub use error::{Error, Result};
