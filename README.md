# Yammer

Local speech-to-text with LLM correction for Linux. Speak naturally, get polished text typed into any application.

![Yammer UI](docs/screenshot.png)

## Features

- **Voice Activity Detection** - Automatically detects when you start/stop speaking
- **Speech-to-Text** - Local Kyutai `stt-1b-en_fr` model, streaming words as you speak (English and French)
- **LLM Text Correction** - Fixes homophones and adds punctuation
- **Floating Overlay** - Transparent window with real-time waveform visualization
- **Universal Input** - Types into any application via xdotool
- **100% Local** - No cloud services, all processing on-device
- **GPU Accelerated** - Speech-to-text runs on CUDA through candle

## Requirements

- Linux with X11 (Wayland not yet supported)
- NVIDIA GPU and the CUDA toolkit (`nvcc` on `PATH`). The default build enables CUDA, and while the STT model falls back to CPU when no GPU is found, CPU inference is too slow for live dictation
- 4GB RAM minimum, 8GB recommended
- ~3GB disk for models

```bash
# Check your session type
echo $XDG_SESSION_TYPE  # Should output: x11
```

## Quick Start

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt install clang libclang-dev libasound2-dev xdotool libayatana-appindicator3-dev nodejs npm

# Arch Linux
sudo pacman -S clang alsa-lib xdotool libayatana-appindicator nodejs npm

# Fedora
sudo dnf install clang clang-devel alsa-lib-devel xdotool libayatana-appindicator-gtk3-devel nodejs npm
```

### 2. Build

```bash
git clone https://github.com/andynu/yammer.git
cd yammer
cargo build --release
```

First build takes 10-15 minutes (compiles candle's CUDA kernels and llama.cpp).

### 3. Download Models

```bash
cargo run --release --bin yammer download-models
```

Downloads the TinyLlama 1.1B correction model (~670MB) to `~/.local/share/yammer/models/`.

The Kyutai STT model (`kyutai/stt-1b-en_fr-candle`, ~2.4GB) is not part of this step. It is fetched from Hugging Face the first time Yammer loads it and cached in `~/.cache/huggingface/hub/`.

### 4. Run

```bash
# GUI mode
cd yammer-app && npm install && npm run tauri dev

# CLI mode (for testing)
cargo run --release --bin yammer dictate
```

## Usage

### GUI Mode

The floating overlay shows:
- Microphone status and state indicator
- Real-time waveform visualization
- Transcribed and corrected text

Press the global hotkey (configurable) to start/stop dictation. Text is automatically typed into the focused application.

### CLI Commands

```bash
# Live dictation
yammer dictate

# List audio devices
yammer list-devices

# Record audio
yammer record --duration 5 --output test.wav

# Transcribe a file
yammer transcribe test.wav

# Test voice activity detection
yammer vad-test --duration 30

# Correct text with LLM
yammer correct "your transcribed text here"
```

## Project Structure

```
yammer/
├── yammer-core/     # Shared types, config, model management
├── yammer-audio/    # Audio capture, VAD, resampling
├── yammer-stt/      # Kyutai streaming speech-to-text (moshi + candle)
├── yammer-llm/      # LLM text correction
├── yammer-output/   # Text output via xdotool
├── yammer-cli/      # CLI interface
└── yammer-app/      # Tauri desktop app
```

## Architecture

```
Microphone → Audio Capture → VAD → Resampler (24kHz)
                                       ↓
                              Kyutai STT (80ms chunks) → LLM Correction → xdotool → Active Window
                                       ↓
                               Tauri UI (waveform, status, live text)
```

## Configuration

VAD sensitivity can be adjusted:

```bash
# More sensitive (quiet speech)
yammer dictate --threshold 0.005

# Less sensitive (loud/clear speech only)
yammer dictate --threshold 0.02
```

## Performance

| Component | Time |
|-----------|------|
| Kyutai STT model load (GPU, cached weights) | ~2.5s, once at startup |
| Kyutai STT, 6.7s clip transcribed as a file (RTX A4000) | ~1.4s |
| Kyutai STT, final words after speech ends | ~0.5s (the model's audio delay) |
| LLM correction | 100-500ms |
| VAD latency | ~200ms start, ~400ms end |

During live dictation the STT model consumes audio in 80ms chunks as it arrives, so transcription runs alongside speech instead of after it. The GPU figures above are from one machine; the app logs a per-dictation stage timing breakdown at `info` level for measuring your own.

## Troubleshooting

**Build fails with "stdbool.h not found"**
```bash
sudo apt install clang libclang-dev
```

**"No audio devices found"**
```bash
sudo apt install libasound2-dev
cargo clean && cargo build --release
```

**Window not transparent**
Requires X11 with a compositor (standard on GNOME/KDE).

**Transcription slow**
Use `--release` builds. If the log shows `CUDA unavailable ... falling back to CPU`, the STT model is running on CPU, which cannot keep up with live audio. Check the NVIDIA driver and that the build had the `cuda` feature (on by default).

## Documentation

- [Audio Testing Guide](./AUDIO_TESTING.md)
- [UI Testing Guide](./yammer-app/TESTING.md)
- [Architecture & Onboarding](./docs/ONBOARDING.md)

## License

MIT - See [LICENSE](./LICENSE) and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md) for details.

## Credits

- [Kyutai STT](https://github.com/kyutai-labs/delayed-streams-modeling) and [moshi](https://github.com/kyutai-labs/moshi) - Speech recognition
- [candle](https://github.com/huggingface/candle) - ML inference for the STT model
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference
- [Tauri](https://tauri.app/) - Desktop app framework
- [cpal](https://github.com/RustAudio/cpal) - Audio I/O
