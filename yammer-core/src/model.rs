//! Model management types

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Information about an available model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Unique identifier for the model
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Model type (whisper, llm)
    pub model_type: ModelType,
    /// URL to download from
    pub url: String,
    /// Expected file size in bytes
    pub size_bytes: u64,
    /// SHA256 hash for verification
    pub sha256: Option<String>,
}

/// Type of model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Whisper,
    Llm,
}

/// Status of a local model
#[derive(Debug, Clone)]
pub enum ModelStatus {
    /// Model is not downloaded
    NotDownloaded,
    /// Model is currently downloading
    Downloading { progress: f32 },
    /// Model is downloaded and ready
    Ready { path: PathBuf },
    /// Model download or verification failed
    Failed { error: String },
}
