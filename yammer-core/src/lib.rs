//! yammer-core: shared types, configuration, and model management
//!
//! This crate provides the foundational types and utilities used across
//! the yammer application.

pub mod config;
pub mod download;
pub mod error;
pub mod model;

pub use config::Config;
pub use download::{format_bytes, DownloadManager, VerifiedHashes};
pub use error::{Error, Result};
pub use model::{get_default_models, get_model_registry, ModelInfo, ModelStatus, ModelType};
