# yammer-core Onboarding Guide

Welcome to yammer-core, the foundational crate for the Yammer dictation application. This guide covers configuration management, model handling, and shared utilities.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### Role of a Core Crate

In a Rust workspace with multiple crates, a "core" crate provides:

- **Shared types**: Structures used across multiple crates
- **Configuration**: Centralized settings management
- **Utilities**: Common functions (download, path resolution)
- **No circular dependencies**: Core depends on nothing internal

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ yammer-audio │   │  yammer-stt  │   │  yammer-llm  │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                    ┌─────▼─────┐
                    │yammer-core│
                    └───────────┘
```

### XDG Base Directory Specification

Linux applications should follow the [XDG Base Directory Spec](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html) for file locations:

| Directory | Purpose | Environment Variable | Default |
|-----------|---------|---------------------|---------|
| **Config** | User configuration | `$XDG_CONFIG_HOME` | `~/.config` |
| **Data** | User data (models, logs) | `$XDG_DATA_HOME` | `~/.local/share` |
| **Cache** | Cached data (can be deleted) | `$XDG_CACHE_HOME` | `~/.cache` |

Yammer uses:
- `~/.config/yammer/config.toml` - Configuration
- `~/.local/share/yammer/models/` - Downloaded models
- `~/.cache/yammer/verified_hashes.json` - SHA256 hash cache

### Configuration File Formats

We chose TOML for configuration:

| Format | Pros | Cons |
|--------|------|------|
| **TOML** | Human-readable, native Rust support, comments | Less common outside Rust |
| **YAML** | Widely used, flexible | Whitespace-sensitive, security issues |
| **JSON** | Universal, strict | No comments, verbose |
| **INI** | Simple | Limited nesting |

Example config.toml:
```toml
[hotkey]
modifiers = ["Control", "Alt"]
key = "D"

[models]
whisper = "base.en"
llm = "tinyllama-1.1b"

[audio]
vad_threshold = 0.01
```

### Model Management

ML applications need to:
1. **Discover models**: Know what's available
2. **Download models**: Get from remote sources
3. **Verify integrity**: SHA256 hash checking
4. **Resolve paths**: Name → filename → full path

We maintain a registry of known models with URLs, sizes, and hashes.

---

## Library Choices

### Why TOML?

We use [toml](https://crates.io/crates/toml) for configuration:

| Alternative | Why Not |
|-------------|---------|
| **serde_yaml** | YAML has quirks (Norway problem), security concerns |
| **serde_json** | No comments, verbose for config files |
| **config** | Adds complexity for simple use case |

TOML advantages:
- First-class Rust ecosystem support
- Human-readable and writable
- Supports comments
- Strict syntax (catches errors early)

### Why dirs Crate?

[dirs](https://crates.io/crates/dirs) provides cross-platform directory paths:

```rust
dirs::config_dir()  // ~/.config on Linux, ~/Library/Preferences on macOS
dirs::data_local_dir()  // ~/.local/share on Linux
dirs::cache_dir()  // ~/.cache on Linux
```

| Alternative | Why Not |
|-------------|---------|
| **directories** | More features, but more complex |
| **Hardcoded paths** | Not cross-platform |
| **Environment vars only** | Tedious, error-prone |

### Why reqwest + sha2?

For downloading and verifying models:

- **reqwest**: Async HTTP client, mature, well-maintained
- **sha2**: Pure Rust SHA-256 implementation

| Alternative | Why Not |
|-------------|---------|
| **ureq** | Blocking only, we use async |
| **hyper** | Lower-level, more boilerplate |
| **ring** for crypto | C dependency, overkill for hashing |

### Why serde_json for Hashes?

The verified hashes file uses JSON because:
- Simpler than TOML for key-value map
- Cache file, not user-edited
- Easy to debug/inspect

---

## Architecture Overview

### Module Structure

```
yammer-core/
├── src/
│   ├── lib.rs       # Public re-exports
│   ├── config.rs    # Configuration structs and loading
│   ├── model.rs     # Model registry and types
│   ├── download.rs  # Download manager with verification
│   └── error.rs     # Error types
```

### Configuration Hierarchy

```
┌──────────────────────────────────────────────────────────────┐
│                          Config                              │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │HotkeyConfig │  │ModelsConfig │  │    AudioConfig      │  │
│  │             │  │             │  │                     │  │
│  │• modifiers  │  │• model_dir  │  │• device             │  │
│  │• key        │  │• whisper    │  │• vad_threshold      │  │
│  │             │  │• llm        │  │• vad_speech_frames  │  │
│  │             │  │             │  │• vad_silence_frames │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │OutputConfig │  │  LlmConfig  │  │     GuiConfig       │  │
│  │             │  │             │  │                     │  │
│  │• method     │  │• correction │  │• window_x           │  │
│  │• typing_    │  │  _prompt    │  │• window_y           │  │
│  │  delay_ms   │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Model Flow

```
┌─────────────┐    ┌────────────┐    ┌──────────────┐
│   Config    │───▶│  Model ID  │───▶│   Filename   │
│             │    │            │    │              │
│ whisper:    │    │ "base.en"  │    │ ggml-base.en │
│ "base.en"   │    │            │    │   .bin       │
└─────────────┘    └────────────┘    └──────────────┘
                                            │
                                            ▼
┌─────────────┐    ┌────────────┐    ┌──────────────┐
│   Model     │◀───│  Download  │◀───│   Full Path  │
│   Ready     │    │  Manager   │    │              │
│             │    │            │    │ ~/.local/    │
│             │    │ • download │    │ share/yammer/│
│             │    │ • verify   │    │ models/      │
└─────────────┘    └────────────┘    └──────────────┘
```

### Key Types

```rust
// Main configuration
pub struct Config {
    pub hotkey: HotkeyConfig,
    pub models: ModelsConfig,
    pub audio: AudioConfig,
    pub output: OutputConfig,
    pub llm: LlmConfig,
    pub gui: GuiConfig,
}

// Model metadata
pub struct ModelInfo {
    pub id: String,           // "whisper-base.en"
    pub name: String,         // "Whisper Base (English)"
    pub model_type: ModelType, // Whisper or Llm
    pub url: String,          // Download URL
    pub size_bytes: u64,      // Expected size
    pub sha256: Option<String>, // Hash for verification
    pub filename: String,     // "ggml-base.en.bin"
}

// Model state tracking
pub enum ModelStatus {
    NotDownloaded,
    Downloading { progress: f32 },
    Ready { path: PathBuf },
    Failed { error: String },
}

// Download and verification
pub struct DownloadManager {
    model_dir: PathBuf,
    client: reqwest::Client,
    verified_hashes: VerifiedHashes,
}
```

---

## Code Walkthrough

### Config Loading (config.rs:183-207)

```rust
impl Config {
    pub fn load() -> Self {
        Self::load_from(&Self::default_path())
    }

    pub fn load_from(path: &PathBuf) -> Self {
        match fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(config) => {
                    info!("Loaded config from {:?}", path);
                    config
                }
                Err(e) => {
                    warn!("Failed to parse config file: {}", e);
                    Self::default()
                }
            },
            Err(e) => {
                debug!("Config file not found: {}", e);
                Self::default()
            }
        }
    }
}
```

Key design decisions:
- **Graceful fallback**: Missing or invalid config → use defaults
- **No panic**: Bad config shouldn't crash the app
- **Logging**: Inform user about config issues

### Model Name Resolution (config.rs:243-262)

```rust
pub fn whisper_model_path(&self) -> PathBuf {
    let whisper = &self.models.whisper;

    // Check if absolute path
    let path = PathBuf::from(whisper);
    if path.is_absolute() && path.exists() {
        return path;
    }

    // Map friendly name to filename
    let filename = match whisper.as_str() {
        "tiny.en" | "whisper-tiny.en" => "ggml-tiny.en.bin",
        "base.en" | "whisper-base.en" => "ggml-base.en.bin",
        "small.en" | "whisper-small.en" => "ggml-small.en.bin",
        "medium.en" | "whisper-medium.en" => "ggml-medium.en.bin",
        "large" | "whisper-large" => "ggml-large.bin",
        _ => whisper, // Assume already a filename
    };

    self.models.model_dir.join(filename)
}
```

Resolution priority:
1. Absolute path (if exists) → use directly
2. Friendly name → map to standard filename
3. Unknown → assume it's the filename

### Download with Verification (download.rs:183-273)

```rust
pub async fn download(
    &mut self,
    model: &ModelInfo,
    progress: Option<ProgressCallback>,
) -> Result<PathBuf> {
    self.ensure_model_dir().await?;

    let dest_path = self.model_path(model);
    let temp_path = dest_path.with_extension("download");

    // Start download
    let response = self.client.get(&model.url).send().await?;
    let total_size = response.content_length().unwrap_or(model.size_bytes);

    // Stream to temp file while computing hash
    let mut file = fs::File::create(&temp_path).await?;
    let mut hasher = Sha256::new();
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        hasher.update(&chunk);
        // Report progress
        if let Some(ref cb) = progress { cb(downloaded, total_size); }
    }

    let actual_sha = format!("{:x}", hasher.finalize());

    // Verify hash
    let expected_sha = model.sha256.as_ref()
        .or_else(|| self.verified_hashes.get(&model.id));

    if let Some(expected) = expected_sha {
        if actual_sha != *expected {
            fs::remove_file(&temp_path).await?;
            return Err(Error::Model("Checksum mismatch".into()));
        }
    } else {
        // First download - save hash for future
        self.verified_hashes.set(model.id.clone(), actual_sha);
        self.verified_hashes.save()?;
    }

    // Atomic move to final location
    fs::rename(&temp_path, &dest_path).await?;
    Ok(dest_path)
}
```

Key patterns:
- **Temp file**: Write to `.download` extension, rename on success
- **Streaming**: Don't load entire file into memory
- **Hash-as-you-go**: Compute SHA256 during download
- **Trust on first download**: Save hash for future verification

### Model Registry (model.rs:56-130)

```rust
pub fn get_model_registry() -> Vec<ModelInfo> {
    vec![
        ModelInfo {
            id: "whisper-base.en".to_string(),
            name: "Whisper Base (English)".to_string(),
            model_type: ModelType::Whisper,
            url: "https://huggingface.co/ggerganov/whisper.cpp/...".to_string(),
            size_bytes: 147_951_465,
            sha256: None,  // Computed on first download
            filename: "ggml-base.en.bin".to_string(),
        },
        // ... more models
    ]
}
```

The registry is hardcoded because:
- No network request needed to list models
- Models rarely change
- Easier to update via code review

---

## Common Tasks

### Load Configuration

```rust
use yammer_core::Config;

fn load_config() -> Config {
    // Load from default path (~/.config/yammer/config.toml)
    let config = Config::load();

    println!("Hotkey: {:?} + {}",
        config.hotkey.modifiers,
        config.hotkey.key
    );
    println!("Whisper model: {}", config.models.whisper);
    println!("LLM enabled: {}", config.llm_enabled());

    config
}
```

### Save Configuration

```rust
use yammer_core::Config;

fn save_config() -> Result<(), String> {
    let mut config = Config::load();

    // Modify settings
    config.audio.vad_threshold = 0.02;
    config.hotkey.key = "S".to_string();

    // Save to default path
    config.save()?;

    Ok(())
}
```

### Get Model Path

```rust
use yammer_core::Config;

fn get_model_paths() {
    let config = Config::load();

    // Whisper model path (always exists)
    let whisper_path = config.whisper_model_path();
    println!("Whisper: {:?}", whisper_path);

    // LLM model path (None if disabled)
    if let Some(llm_path) = config.llm_model_path() {
        println!("LLM: {:?}", llm_path);
    } else {
        println!("LLM correction disabled");
    }
}
```

### Download a Model

```rust
use yammer_core::{DownloadManager, get_model_registry, format_bytes};

async fn download_model(model_id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = DownloadManager::default_model_dir();
    let mut manager = DownloadManager::new(model_dir);

    // Find model in registry
    let registry = get_model_registry();
    let model = registry.iter()
        .find(|m| m.id == model_id)
        .ok_or("Model not found")?;

    // Download with progress
    let progress = Box::new(|downloaded, total| {
        println!("Progress: {} / {}",
            format_bytes(downloaded),
            format_bytes(total)
        );
    });

    let path = manager.download(model, Some(progress)).await?;
    println!("Downloaded to: {:?}", path);

    Ok(())
}
```

### Check Model Status

```rust
use yammer_core::{DownloadManager, get_model_registry, ModelStatus};

async fn check_models() {
    let model_dir = DownloadManager::default_model_dir();
    let manager = DownloadManager::new(model_dir);
    let registry = get_model_registry();

    for model in &registry {
        let status = manager.check_status(model).await;
        match status {
            ModelStatus::Ready { path } => {
                println!("{}: Ready at {:?}", model.id, path);
            }
            ModelStatus::NotDownloaded => {
                println!("{}: Not downloaded", model.id);
            }
            ModelStatus::Failed { error } => {
                println!("{}: Failed - {}", model.id, error);
            }
            ModelStatus::Downloading { progress } => {
                println!("{}: Downloading {:.0}%", model.id, progress * 100.0);
            }
        }
    }
}
```

### Add a New Config Option

1. Add field to appropriate sub-config struct:

```rust
// In config.rs
pub struct AudioConfig {
    // ... existing fields
    /// New option description
    pub new_option: String,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults
            new_option: "default_value".to_string(),
        }
    }
}
```

2. The field automatically works with load/save due to serde.

3. Add tests:

```rust
#[test]
fn test_new_option_default() {
    let config = Config::default();
    assert_eq!(config.audio.new_option, "default_value");
}
```

### Add a New Model

1. Add to registry in `model.rs`:

```rust
ModelInfo {
    id: "new-model-id".to_string(),
    name: "New Model Display Name".to_string(),
    model_type: ModelType::Whisper, // or Llm
    url: "https://example.com/model.bin".to_string(),
    size_bytes: 123_456_789,
    sha256: None, // Will be computed on first download
    filename: "model-file.bin".to_string(),
},
```

2. Add name mapping in `whisper_model_path()` or `llm_model_path()`:

```rust
"new-model" | "new-model-full" => "model-file.bin",
```

### Persist Window Position

```rust
use yammer_core::Config;

fn save_window_position(x: i32, y: i32) -> Result<(), String> {
    let mut config = Config::load();
    config.set_window_position(x, y)?;
    Ok(())
}

fn restore_window_position(
    screen_width: u32,
    screen_height: u32,
    window_width: u32,
    window_height: u32,
) -> Option<(i32, i32)> {
    let config = Config::load();
    config.validated_window_position(
        screen_width, screen_height,
        window_width, window_height
    )
}
```

### Custom Config Path

```rust
use yammer_core::Config;
use std::path::PathBuf;

fn load_custom_config(path: &str) -> Config {
    let path = PathBuf::from(path);
    Config::load_from(&path)
}
```

---

## Configuration Reference

### Default config.toml

```toml
[hotkey]
modifiers = ["Control", "Alt"]
key = "D"

[models]
model_dir = "~/.local/share/yammer/models"
whisper = "base.en"
llm = "tinyllama-1.1b"

[audio]
# device = "default"  # Omit for system default
vad_threshold = 0.01
vad_speech_frames = 3
vad_silence_frames = 15

[output]
method = "type"
typing_delay_ms = 0

[llm]
# correction_prompt = "Custom prompt with {text} placeholder"

[gui]
# window_x = 100  # Omit for auto-center
# window_y = 100
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `XDG_CONFIG_HOME` | Override config directory |
| `XDG_DATA_HOME` | Override data directory |
| `XDG_CACHE_HOME` | Override cache directory |

---

## Summary

yammer-core provides foundational infrastructure:

1. **Config**: Hierarchical TOML configuration with defaults
2. **Model Registry**: Known models with metadata
3. **Download Manager**: Async download with SHA256 verification
4. **Path Resolution**: Friendly names → actual file paths

Key files:
- `src/config.rs`: Configuration structs, load/save, path resolution
- `src/model.rs`: Model registry and types
- `src/download.rs`: Download manager with progress and verification
- `src/error.rs`: Shared error types

Design principles:
- Graceful degradation (bad config → defaults)
- XDG compliance (proper Linux paths)
- Trust on first download (hash computed and saved)
- Atomic operations (temp file → rename)
