//! Error types for yammer-core

use thiserror::Error;

/// Core error type
#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Model error: {0}")]
    Model(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
