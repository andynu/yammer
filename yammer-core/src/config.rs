//! Configuration management for yammer

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Directory where models are stored
    pub model_dir: PathBuf,
    /// Path to whisper model
    pub whisper_model: Option<PathBuf>,
    /// Path to LLM model
    pub llm_model: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        let model_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("yammer")
            .join("models");

        Self {
            model_dir,
            whisper_model: None,
            llm_model: None,
        }
    }
}
