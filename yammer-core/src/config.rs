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
    /// Input device name (use "default" or omit for system default)
    /// Run `yammer list-devices` to see available devices
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
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
            device: None,
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
    /// Delay between keystrokes in milliseconds (for "type" method)
    /// Lower values = faster typing, higher values = more compatible with slow apps
    /// Default is 0 (xdotool's default is 12ms; 0 means no delay)
    pub typing_delay_ms: u32,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            method: "type".to_string(),
            typing_delay_ms: 0,
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

/// GUI window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GuiConfig {
    /// Window X position (if None, uses center or default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_x: Option<i32>,
    /// Window Y position (if None, uses center or default)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_y: Option<i32>,
}

impl Default for GuiConfig {
    fn default() -> Self {
        Self {
            window_x: None,
            window_y: None,
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
    /// GUI settings
    pub gui: GuiConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hotkey: HotkeyConfig::default(),
            models: ModelsConfig::default(),
            audio: AudioConfig::default(),
            output: OutputConfig::default(),
            llm: LlmConfig::default(),
            gui: GuiConfig::default(),
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

    /// Get saved window position, or None if not set
    pub fn window_position(&self) -> Option<(i32, i32)> {
        match (self.gui.window_x, self.gui.window_y) {
            (Some(x), Some(y)) => Some((x, y)),
            _ => None,
        }
    }

    /// Set window position and save config
    pub fn set_window_position(&mut self, x: i32, y: i32) -> Result<(), String> {
        self.gui.window_x = Some(x);
        self.gui.window_y = Some(y);
        self.save()
    }

    /// Validate and clamp window position to screen bounds
    /// Returns the clamped position, or None if window should use default centering
    pub fn validated_window_position(
        &self,
        screen_width: u32,
        screen_height: u32,
        window_width: u32,
        _window_height: u32,
    ) -> Option<(i32, i32)> {
        let (x, y) = self.window_position()?;

        // Calculate max valid positions (window must be at least partially visible)
        let max_x = screen_width as i32 - 50; // At least 50px visible
        let max_y = screen_height as i32 - 50;
        let min_x = -(window_width as i32) + 50;
        let min_y = 0; // Don't allow above screen top

        // Clamp to valid range
        let clamped_x = x.clamp(min_x, max_x);
        let clamped_y = y.clamp(min_y, max_y);

        Some((clamped_x, clamped_y))
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

    #[test]
    fn test_window_position_none_when_not_set() {
        let config = Config::default();
        assert!(config.window_position().is_none());
    }

    #[test]
    fn test_window_position_set() {
        let mut config = Config::default();
        config.gui.window_x = Some(100);
        config.gui.window_y = Some(200);
        assert_eq!(config.window_position(), Some((100, 200)));
    }

    #[test]
    fn test_window_position_partial_set() {
        let mut config = Config::default();
        config.gui.window_x = Some(100);
        // y not set
        assert!(config.window_position().is_none());
    }

    #[test]
    fn test_validated_window_position_within_bounds() {
        let mut config = Config::default();
        config.gui.window_x = Some(100);
        config.gui.window_y = Some(100);
        // Screen 1920x1080, window 200x100
        let result = config.validated_window_position(1920, 1080, 200, 100);
        assert_eq!(result, Some((100, 100)));
    }

    #[test]
    fn test_validated_window_position_clamps_x_max() {
        let mut config = Config::default();
        config.gui.window_x = Some(2000); // Beyond screen width
        config.gui.window_y = Some(100);
        // Screen 1920x1080, window 200x100
        // max_x = 1920 - 50 = 1870
        let result = config.validated_window_position(1920, 1080, 200, 100);
        assert_eq!(result, Some((1870, 100)));
    }

    #[test]
    fn test_validated_window_position_clamps_x_min() {
        let mut config = Config::default();
        config.gui.window_x = Some(-300); // Too far left
        config.gui.window_y = Some(100);
        // Screen 1920x1080, window 200x100
        // min_x = -200 + 50 = -150
        let result = config.validated_window_position(1920, 1080, 200, 100);
        assert_eq!(result, Some((-150, 100)));
    }

    #[test]
    fn test_validated_window_position_clamps_y_min() {
        let mut config = Config::default();
        config.gui.window_x = Some(100);
        config.gui.window_y = Some(-50); // Above screen top
        // min_y = 0
        let result = config.validated_window_position(1920, 1080, 200, 100);
        assert_eq!(result, Some((100, 0)));
    }

    #[test]
    fn test_validated_window_position_clamps_y_max() {
        let mut config = Config::default();
        config.gui.window_x = Some(100);
        config.gui.window_y = Some(1100); // Below screen bottom
        // max_y = 1080 - 50 = 1030
        let result = config.validated_window_position(1920, 1080, 200, 100);
        assert_eq!(result, Some((100, 1030)));
    }

    #[test]
    fn test_validated_window_position_returns_none_when_not_set() {
        let config = Config::default();
        assert!(config.validated_window_position(1920, 1080, 200, 100).is_none());
    }
}
