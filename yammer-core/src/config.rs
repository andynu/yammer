//! Configuration management for yammer

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Hotkey configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HotkeyConfig {
    /// Key modifiers (Control, Alt, Super, Shift)
    pub modifiers: Vec<String>,
    /// The key to press
    pub key: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            modifiers: vec!["Control".to_string(), "Alt".to_string()],
            key: "D".to_string(),
        }
    }
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelsConfig {
    /// Directory where models are stored
    pub model_dir: PathBuf,
    /// Whisper model name or path (tiny.en, base.en, small.en, medium.en)
    pub whisper: String,
    /// LLM model name or path, or "none" to disable
    pub llm: String,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        let model_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("yammer")
            .join("models");

        Self {
            model_dir,
            whisper: "base.en".to_string(),
            llm: "tinyllama-1.1b".to_string(),
        }
    }
}

/// Audio/VAD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// RMS threshold for speech detection
    pub vad_threshold: f64,
    /// Number of consecutive frames to confirm speech start
    pub vad_speech_frames: usize,
    /// Number of consecutive frames to confirm speech end
    pub vad_silence_frames: usize,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            vad_threshold: 0.01,
            vad_speech_frames: 3,
            vad_silence_frames: 15,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Output method: "type" for xdotool type, "clipboard" for clipboard paste
    pub method: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            method: "type".to_string(),
        }
    }
}

/// LLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LlmConfig {
    /// Custom correction prompt template. Use {text} as placeholder for transcribed text.
    /// If omitted, uses built-in default few-shot prompt.
    /// If {text} placeholder is missing, text is appended at end.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub correction_prompt: Option<String>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            correction_prompt: None,
        }
    }
}

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Hotkey settings
    pub hotkey: HotkeyConfig,
    /// Model settings
    pub models: ModelsConfig,
    /// Audio/VAD settings
    pub audio: AudioConfig,
    /// Output settings
    pub output: OutputConfig,
    /// LLM settings
    pub llm: LlmConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: HotkeyConfig::default(),
            models: ModelsConfig::default(),
            audio: AudioConfig::default(),
            output: OutputConfig::default(),
            llm: LlmConfig::default(),
        }
    }
}

impl Config {
    /// Get the default config file path
    pub fn default_path() -> PathBuf {
        dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("yammer")
            .join("config.toml")
    }

    /// Load configuration from the default path, or return defaults
    pub fn load() -> Self {
        Self::load_from(&Self::default_path())
    }

    /// Load configuration from a specific path, or return defaults
    pub fn load_from(path: &PathBuf) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => {
                    info!("Loaded config from {:?}", path);
                    config
                }
                Err(e) => {
                    warn!("Failed to parse config file {:?}: {}", path, e);
                    warn!("Using default configuration");
                    Self::default()
                }
            },
            Err(e) => {
                debug!("Config file not found at {:?}: {}", path, e);
                debug!("Using default configuration");
                Self::default()
            }
        }
    }

    /// Save configuration to the default path
    pub fn save(&self) -> Result<(), String> {
        self.save_to(&Self::default_path())
    }

    /// Save configuration to a specific path
    pub fn save_to(&self, path: &PathBuf) -> Result<(), String> {
        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create config directory: {}", e))?;
        }

        let contents = toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(path, contents)
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        info!("Saved config to {:?}", path);
        Ok(())
    }

    /// Create default config file if it doesn't exist
    pub fn ensure_default_exists() -> Result<(), String> {
        let path = Self::default_path();
        if !path.exists() {
            info!("Creating default config at {:?}", path);
            Self::default().save_to(&path)?;
        }
        Ok(())
    }

    /// Get the path to the whisper model
    pub fn whisper_model_path(&self) -> PathBuf {
        // Check if it's an absolute path
        let whisper = &self.models.whisper;
        let path = PathBuf::from(whisper);
        if path.is_absolute() && path.exists() {
            return path;
        }

        // Try to resolve as a model name
        let filename = match whisper.as_str() {
            "tiny.en" | "whisper-tiny.en" => "ggml-tiny.en.bin",
            "base.en" | "whisper-base.en" => "ggml-base.en.bin",
            "small.en" | "whisper-small.en" => "ggml-small.en.bin",
            "medium.en" | "whisper-medium.en" => "ggml-medium.en.bin",
            "large" | "whisper-large" => "ggml-large.bin",
            _ => whisper, // Assume it's already a filename
        };

        self.models.model_dir.join(filename)
    }

    /// Get the path to the LLM model, or None if disabled
    pub fn llm_model_path(&self) -> Option<PathBuf> {
        let llm = &self.models.llm;
        if llm == "none" || llm.is_empty() {
            return None;
        }

        // Check if it's an absolute path
        let path = PathBuf::from(llm);
        if path.is_absolute() && path.exists() {
            return Some(path);
        }

        // Try to resolve as a model name
        let filename = match llm.as_str() {
            "tinyllama-1.1b" => "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "phi-3-mini" => "Phi-3-mini-4k-instruct-q4.gguf",
            "gemma-2-2b" => "gemma-2-2b-it-Q4_K_M.gguf",
            _ => llm, // Assume it's already a filename
        };

        Some(self.models.model_dir.join(filename))
    }

    /// Check if LLM correction is enabled
    pub fn llm_enabled(&self) -> bool {
        self.models.llm != "none" && !self.models.llm.is_empty()
    }

    /// Format config as TOML string
    pub fn to_toml(&self) -> Result<String, String> {
        toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.hotkey.key, "D");
        assert_eq!(config.hotkey.modifiers, vec!["Control", "Alt"]);
        assert_eq!(config.models.whisper, "base.en");
        assert_eq!(config.audio.vad_threshold, 0.01);
        assert_eq!(config.output.method, "type");
    }

    #[test]
    fn test_whisper_model_path() {
        let config = Config::default();
        let path = config.whisper_model_path();
        assert!(path.to_string_lossy().contains("ggml-base.en.bin"));
    }

    #[test]
    fn test_llm_disabled() {
        let mut config = Config::default();
        config.models.llm = "none".to_string();
        assert!(!config.llm_enabled());
        assert!(config.llm_model_path().is_none());
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = Config::default();
        let toml_str = config.to_toml().unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        assert_eq!(config.hotkey.key, parsed.hotkey.key);
        assert_eq!(config.models.whisper, parsed.models.whisper);
    }
}
