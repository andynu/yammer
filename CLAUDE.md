# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Yammer

Local, privacy-first speech-to-text dictation for Linux, macOS, and Windows. Converts speech to polished text using platform-native STT (Apple SpeechAnalyzer on macOS, Kyutai Moshi on Linux/Windows) + TinyLlama LLM correction, all running locally. Tauri v2 desktop app with floating overlay UI.

## Build & Run Commands

```bash
# Build workspace (all crates)
cargo build --workspace
cargo build --release --workspace
cargo build --release --features cuda   # With NVIDIA GPU acceleration

# Check (fast compile check)
cargo check --workspace

# Lint & format
cargo clippy --workspace
cargo fmt --all
cargo fmt --all -- --check

# Tests
cargo test --workspace

# GUI app (Tauri)
cd yammer-app && npm install && npm run tauri dev    # Dev with hot reload
cd yammer-app && npm run tauri build                 # Production build

# CLI (component testing)
cargo run -p yammer-cli -- download-models
cargo run -p yammer-cli -- list-devices
cargo run -p yammer-cli -- dictate
cargo run -p yammer-cli -- record -d 5 -o test.wav
cargo run -p yammer-cli -- transcribe test.wav
cargo run -p yammer-cli -- correct "some text"
cargo run -p yammer-cli -- vad-test -d 30
```

## System Dependencies

**Linux**: `sudo apt install clang libclang-dev libasound2-dev cmake xdotool libayatana-appindicator3-dev libwebkit2gtk-4.1-dev nodejs npm`
**macOS**: `brew install cmake node`
**Windows**: `choco install llvm nodejs`

## Architecture

7-crate Cargo workspace, each single-responsibility:

| Crate | Role |
|---|---|
| `yammer-core` | Config (TOML/serde), model registry, model downloads |
| `yammer-audio` | cpal audio capture, energy-based VAD, resampling (rubato) |
| `yammer-stt` | STT with `SttBackend` trait: Apple SpeechAnalyzer (macOS, Swift FFI), Kyutai Moshi (Linux/Win, candle) |
| `yammer-llm` | TinyLlama text correction (llama_cpp) |
| `yammer-output` | Text injection: xdotool (Linux), enigo (macOS/Win), clipboard fallback |
| `yammer-cli` | CLI interface wrapping all above |
| `yammer-app` | Tauri v2 GUI: hotkey listener, DictationPipeline orchestration, TypeScript/Vite frontend |

**Dictation pipeline flow**: Hotkey → Audio capture + VAD → STT (Apple SpeechAnalyzer on macOS, Kyutai on Linux/Win) → TinyLlama correction → Text output to target window.

**Frontend-backend**: Tauri events (JSON). Backend emits `StateChanged`, `AudioLevel`, `Transcript`, `Error` to frontend.

## Key Patterns

- **Error handling**: `thiserror` enums in library crates, `anyhow::Result` in CLI/app
- **Platform abstraction**: `#[cfg(target_os = "...")]` gates throughout `yammer-stt`, `yammer-output`, and `yammer-app`. `yammer-stt` uses a `Transcriber` type alias that resolves to platform-specific backends at compile time
- **Hotkey**: `rdev` on Linux/Windows, CGEventTap on macOS. Captures active window ID at press, refocuses before output
- **Async**: tokio runtime, `mpsc` channels for pipeline→UI events, `Arc<AtomicBool>` for cancellation
- **Config**: `~/.config/yammer/config.toml` (Linux), `~/Library/Application Support/yammer/config.toml` (macOS), serde with `#[serde(default)]`
- **Models**: Downloaded from HuggingFace Hub on first use, cached in `~/.cache/huggingface/`
- **Logging**: `tracing` crate, enable with `RUST_LOG=debug`

## Pipeline State Machine

`Idle → Listening → Correcting → Done` (or `→ Error` / `→ Discarded` at any point). Defined in `yammer-app/src-tauri/src/pipeline.rs`.
