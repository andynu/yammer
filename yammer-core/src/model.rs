//! Model management types and registry

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
    /// Filename to save as
    pub filename: String,
}

/// Type of model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelType {
    Whisper,
    Llm,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Whisper => write!(f, "whisper"),
            ModelType::Llm => write!(f, "llm"),
        }
    }
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

/// Built-in model registry with known models
pub fn get_model_registry() -> Vec<ModelInfo> {
    vec![
        // Whisper models (GGML format from HuggingFace)
        ModelInfo {
            id: "whisper-base.en".to_string(),
            name: "Whisper Base (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".to_string(),
            size_bytes: 147_951_465, // ~141MB
            sha256: Some("a03779c86df3323075f5e796b3f6f1fe7d8abbca".to_string()),
            filename: "ggml-base.en.bin".to_string(),
        },
        ModelInfo {
            id: "whisper-small.en".to_string(),
            name: "Whisper Small (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".to_string(),
            size_bytes: 487_601_929, // ~465MB
            sha256: Some("6a4b2e8696cd4d0662e4c2c47b07ef3f4ef0c2f9".to_string()),
            filename: "ggml-small.en.bin".to_string(),
        },
        // LLM models (GGUF format)
        ModelInfo {
            id: "qwen2-1.5b".to_string(),
            name: "Qwen2 1.5B (Grammar/Punctuation)".to_string(),
            model_type: ModelType::Llm,
            url: "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf".to_string(),
            size_bytes: 1_019_348_992, // ~972MB
            sha256: None, // Will verify after first download
            filename: "qwen2-1_5b-instruct-q4_k_m.gguf".to_string(),
        },
    ]
}

/// Get default models for a fresh installation
pub fn get_default_models() -> Vec<&'static str> {
    vec!["whisper-base.en", "qwen2-1.5b"]
}
