# Yammer - Linux Dictation App

Local speech-to-text with LLM correction for Linux (X11). Speak naturally, get polished text typed into any application.

## Features

- 🎤 **Voice Activity Detection** - Automatically detects when you start/stop speaking
- 🗣️ **Speech-to-Text** - Local Whisper model for accurate transcription
- ✨ **LLM Text Correction** - Fixes homophones, adds punctuation, removes filler words
- 🪟 **Floating Overlay UI** - Transparent window with real-time waveform visualization
- ⌨️ **Universal Input** - Types into any application (via xdotool)
- 🔒 **100% Local** - No cloud services, all processing on-device
- 🚀 **GPU Accelerated** - Optional CUDA support for faster transcription

## System Requirements

- **OS**: Linux (X11 session required - not Wayland)
- **Architecture**: x86_64
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: ~2GB for models
- **Audio**: Working microphone and ALSA

Check your session type:
```bash
echo $XDG_SESSION_TYPE  # Should output: x11
```

## Prerequisites

### Required Dependencies

```bash
# Ubuntu/Debian
sudo apt install clang libclang-dev libasound2-dev xdotool libayatana-appindicator3-dev

# Arch Linux
sudo pacman -S clang alsa-lib xdotool libayatana-appindicator

# Fedora
sudo dnf install clang clang-devel alsa-lib-devel xdotool libayatana-appindicator-gtk3-devel
```

**Why each dependency:**
- `clang` + `libclang-dev` - Required for llama.cpp Rust bindings (LLM support)
- `libasound2-dev` - Required for audio capture via ALSA
- `xdotool` - Required for typing text into applications (Phase 6)
- `libayatana-appindicator3-dev` - Required for system tray icon (Tauri UI)

### Rust Toolchain

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Ensure you have a recent version
rustup update
```

Requires Rust 1.70 or later.

### Node.js (for Tauri UI)

```bash
# Ubuntu/Debian
sudo apt install nodejs npm

# Or use nvm for latest version
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install node
```

## Installation

### 1. Clone and Build

```bash
git clone <repository-url>
cd yammer

# Build the workspace
cargo build --release
```

**Note**: First build will take 10-15 minutes as it compiles Whisper and llama.cpp.

### 2. Download AI Models

```bash
# Download default models (Whisper Base + TinyLlama)
cargo run --release --bin yammer download-models

# Or download only Whisper (for testing)
cargo run --release --bin yammer download-models --model whisper-base-en
```

Models are downloaded to: `~/.cache/yammer/models/`

**Model sizes:**
- Whisper Base English: ~141MB (good balance of speed and accuracy)
- Phi-2 Q4 (LLM): ~1.6GB (text correction)

### 3. Test Your Setup

```bash
# List audio devices
cargo run --release --bin yammer list-devices

# Test live dictation
cargo run --release --bin yammer dictate
```

If dictation works, you're ready to go!

## Quick Start

### CLI Mode (for testing)

```bash
# Live dictation in terminal
cargo run --release --bin yammer dictate

# Record audio
cargo run --release --bin yammer record --duration 5 --output test.wav --resample

# Transcribe a file
cargo run --release --bin yammer transcribe test.wav

# Test voice activity detection
cargo run --release --bin yammer vad-test --duration 30

# Correct text with LLM
cargo run --release --bin yammer correct "this is some transcribed text"
```

### GUI Mode (Tauri overlay)

```bash
cd yammer-app
npm install
npm run tauri dev
```

A floating transparent window appears with:
- Real-time waveform visualization
- Status indicators (listening, processing, etc.)
- Transcription display

**Test the UI:**
- Open browser dev tools (Right-click → Inspect)
- Console: `window.testStates()` - cycles through all states
- Console: `window.testWaveform()` - tests waveform animation

## Project Structure

```
yammer/
├── yammer-core/          # Shared types, config, model management
├── yammer-audio/         # Audio capture, VAD, resampling
├── yammer-stt/           # Whisper speech-to-text
├── yammer-llm/           # LLM text correction (Phi-2)
├── yammer-cli/           # CLI for testing components
└── yammer-app/           # Tauri UI (floating overlay)
    ├── src/              # Frontend (HTML/CSS/JS)
    └── src-tauri/        # Backend (Rust)
```

## Documentation

- **[AUDIO_TESTING.md](./AUDIO_TESTING.md)** - Complete guide to testing audio capture, VAD, and transcription
- **[yammer-app/TESTING.md](./yammer-app/TESTING.md)** - Guide to testing the Tauri UI components

## Development Status

**Completed Phases:**

- ✅ **Phase 1**: Model management and downloading
- ✅ **Phase 2**: Audio pipeline (capture, VAD, resampling)
- ✅ **Phase 3**: Speech-to-text (Whisper integration)
- ✅ **Phase 5**: Tauri UI (overlay window, waveform, status indicators)

**In Progress:**

- 🚧 **Phase 4**: LLM text correction (functional, needs optimization)
- 🚧 **Phase 6**: Integration (global hotkey, xdotool output, full pipeline)

**Current Status:** Core dictation engine (audio → transcription) is fully functional. UI is complete but not yet wired to audio pipeline. LLM correction works but needs GPU optimization.

## Troubleshooting

### Build fails with "stdbool.h not found"

You need to install clang:
```bash
sudo apt install clang libclang-dev
```

### "No audio devices found"

Install ALSA development libraries:
```bash
sudo apt install libasound2-dev
```

Then rebuild:
```bash
cargo clean
cargo build --release
```

### Window not transparent / not draggable

- **Transparency**: Requires X11 (not Wayland) and a compositor (usually automatic on GNOME)
- **Dragging**: Known issue, tracked in issue yam-f8a

### Transcription is slow

- Use `--release` builds (10x faster than debug)
- Enable GPU acceleration (requires CUDA)
- Whisper Tiny model is fastest but less accurate

### VAD too sensitive / not sensitive enough

Adjust threshold:
```bash
# More sensitive (picks up quiet speech)
cargo run --bin yammer dictate --threshold 0.005

# Less sensitive (only loud/clear speech)
cargo run --bin yammer dictate --threshold 0.02
```

Default 0.01 works for most quiet environments.

## Performance

**Whisper Tiny English (CPU):**
- ~100-300ms per second of audio
- 5 second utterance → ~500ms-1.5s processing time

**Whisper Tiny English (CUDA):**
- ~50-150ms per second of audio
- 5 second utterance → ~250ms-750ms processing time

**LLM Correction (Phi-2 Q4):**
- ~100-500ms for typical sentence
- Varies based on text length and system

**VAD Latency:**
- Speech start detection: ~200ms
- Speech end detection: ~400ms (includes silence confirmation)

## Roadmap

**Short term:**
- [ ] Wire Tauri UI to audio pipeline
- [ ] Implement global hotkey (Super+D)
- [ ] Add xdotool text output
- [ ] Optimize LLM VRAM usage
- [ ] Configuration UI

**Medium term:**
- [ ] Support for larger Whisper models (better accuracy)
- [ ] Custom vocabulary/corrections
- [ ] Multiple hotkey actions (dictate vs command mode)
- [ ] System tray integration

**Long term:**
- [ ] Wayland support (requires different approach)
- [ ] Plugin system for custom processing
- [ ] Multi-language support
- [ ] Voice commands (not just dictation)

## Architecture

```
User Speech
    ↓
[Microphone] → [cpal Audio Capture]
    ↓
[VAD] → Detects speech segments
    ↓
[Resampler] → 16kHz mono
    ↓
[Whisper] → Raw transcription
    ↓
[LLM] → Corrected text
    ↓
[xdotool] → Types into active window
```

**Parallel UI:**
```
[Audio Samples] ──→ [Tauri UI]
[State Changes] ──→ [Status Display]
[Transcription] ──→ [Text Display]
```

## Contributing

This project uses Beads for issue tracking:

```bash
# View available work
bd ready

# View all issues
bd list

# Show issue details
bd show <issue-id>
```

See `.beads/` directory for issue tracking.

## License

[License information here]

## Credits

Built with:
- [whisper.rs](https://github.com/tazz4843/whisper-rs) - Rust bindings for Whisper
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference
- [Tauri](https://tauri.app/) - Desktop app framework
- [cpal](https://github.com/RustAudio/cpal) - Cross-platform audio I/O

## Support

For issues and questions:
- Check documentation: AUDIO_TESTING.md, yammer-app/TESTING.md
- Review open issues: `bd list`
- Create new issue: `bd create "Issue description"`
