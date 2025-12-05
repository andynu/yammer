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
            id: "whisper-tiny.en".to_string(),
            name: "Whisper Tiny (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin".to_string(),
            size_bytes: 77_704_715, // ~74MB
            sha256: None, // Will log actual SHA256 on first download
            filename: "ggml-tiny.en.bin".to_string(),
        },
        ModelInfo {
            id: "whisper-base.en".to_string(),
            name: "Whisper Base (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin".to_string(),
            size_bytes: 147_951_465, // ~141MB
            sha256: None, // Will log actual SHA256 on first download
            filename: "ggml-base.en.bin".to_string(),
        },
        ModelInfo {
            id: "whisper-small.en".to_string(),
            name: "Whisper Small (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin".to_string(),
            size_bytes: 487_601_929, // ~465MB
            sha256: None, // TODO: Fetch from server or verify after download
            filename: "ggml-small.en.bin".to_string(),
        },
        // LLM models (GGUF format)
        ModelInfo {
            id: "tinyllama-1.1b".to_string(),
            name: "TinyLlama 1.1B Chat (Grammar/Punctuation)".to_string(),
            model_type: ModelType::Llm,
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
            size_bytes: 668_788_096, // ~638MB (actual size from HuggingFace)
            sha256: None,
            filename: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf".to_string(),
        },
        // NOTE: These models are incompatible with llama_cpp Rust 0.3:
        // - Phi-3: unknown architecture 'phi3'
        // - Gemma2: unknown architecture 'gemma2'
        // - Qwen2: tied embeddings issue (missing output.weight tensor)
        // ModelInfo {
        //     id: "phi-3-mini-4k".to_string(),
        //     name: "Phi-3-Mini 4K (Grammar/Punctuation)".to_string(),
        //     model_type: ModelType::Llm,
        //     url: "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf".to_string(),
        //     size_bytes: 2_393_231_072, // ~2.3GB (actual size from HuggingFace)
        //     sha256: None,
        //     filename: "Phi-3-mini-4k-instruct-q4.gguf".to_string(),
        // },
        // ModelInfo {
        //     id: "gemma-2-2b".to_string(),
        //     name: "Gemma 2 2B (Grammar/Punctuation)".to_string(),
        //     model_type: ModelType::Llm,
        //     url: "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf".to_string(),
        //     size_bytes: 1_708_582_752, // ~1.6GB (actual size from HuggingFace)
        //     sha256: None,
        //     filename: "gemma-2-2b-it-Q4_K_M.gguf".to_string(),
        // },
        // NOTE: Qwen2 models currently incompatible with llama.cpp 0.3 (missing output.weight tensor)
        // See: https://github.com/QwenLM/Qwen2.5/issues/255
        // ModelInfo {
        //     id: "qwen2-1.5b".to_string(),
        //     name: "Qwen2 1.5B (Grammar/Punctuation)".to_string(),
        //     model_type: ModelType::Llm,
        //     url: "https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf".to_string(),
        //     size_bytes: 986_045_824, // ~941MB (actual size from HuggingFace)
        //     sha256: None,
        //     filename: "qwen2-1_5b-instruct-q4_k_m.gguf".to_string(),
        // },
    ]
}

/// Get default models for a fresh installation
pub fn get_default_models() -> Vec<&'static str> {
    vec!["whisper-base.en", "tinyllama-1.1b"]
}
