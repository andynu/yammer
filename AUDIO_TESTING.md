# Yammer Audio & Transcription Testing Guide

This guide walks you through testing the core audio capture, voice activity detection, and speech-to-text functionality that was built in Phases 1-3.

## Prerequisites

```bash
# Check Whisper model is downloaded
cargo run --bin yammer-cli list-models
```

If no Whisper model is downloaded:
```bash
cargo run --bin yammer-cli download-models
```

This will download:
- Whisper Tiny English model (~75MB) - for speech-to-text
- Phi-2 Q4 model (~1.6GB) - for LLM text correction (blocked by clang dependency)

## Phase 1: Model Management

### List Available Models
```bash
cargo run --bin yammer-cli list-models
```

Shows:
- Available models in the registry
- Which models are downloaded
- Download status and file locations

### Download Specific Model
```bash
cargo run --bin yammer-cli download-models --model whisper-tiny-en
```

### Download All Models
```bash
cargo run --bin yammer-cli download-models --all
```

**Note**: LLM model download will work, but LLM inference requires clang to be installed first.

## Phase 2: Audio Pipeline

### List Audio Devices
```bash
cargo run --bin yammer-cli list-devices
```

Shows:
- All available audio input devices
- Sample rates and channel configurations
- Which device is the default

### Record Audio (5 seconds)
```bash
cargo run --bin yammer-cli record --duration 5 --output test.wav
```

Options:
- `--duration N` - Record N seconds (default: 5)
- `--output FILE` - Output WAV file path
- `--device "Device Name"` - Specific device (optional)
- `--resample` - Resample to 16kHz for Whisper compatibility

### Record with Resampling
```bash
cargo run --bin yammer-cli record --duration 5 --output test.wav --resample
```

This captures at native sample rate, then resamples to 16kHz mono (required for Whisper).

### Play Back Recording
```bash
aplay test.wav
```

### Voice Activity Detection (VAD) Test

Test VAD in real-time:
```bash
cargo run --bin yammer-cli vad-test --duration 30 --threshold 0.01
```

What to watch for:
- **QUIET** - Background silence detected
- **maybe speech...** - Audio level rising
- **>>> SPEECH <<<** - Active speech detected
- **maybe quiet...** - Audio level falling
- **QUIET** - Back to silence (shows speech duration)

The VAD uses hysteresis to avoid false triggers:
- Silence → Speech requires sustained audio above threshold
- Speech → Silence requires sustained quiet below threshold

**Adjust sensitivity:**
- Lower threshold (e.g., 0.005) = more sensitive (picks up quieter speech)
- Higher threshold (e.g., 0.02) = less sensitive (only loud/clear speech)

```bash
# Very sensitive
cargo run --bin yammer-cli vad-test --threshold 0.005

# Less sensitive
cargo run --bin yammer-cli vad-test --threshold 0.02
```

## Phase 3: Speech-to-Text

### Transcribe a WAV File

First, record something:
```bash
cargo run --bin yammer-cli record --duration 10 --output speech.wav --resample
```

Then transcribe it:
```bash
cargo run --bin yammer-cli transcribe speech.wav
```

Output shows the transcribed text.

### Transcribe with Timestamps
```bash
cargo run --bin yammer-cli transcribe speech.wav --timestamps
```

Shows segment-by-segment transcription with time ranges.

### Transcribe with Specific Model
```bash
cargo run --bin yammer-cli transcribe speech.wav --model /path/to/model.bin
```

By default, it auto-detects the downloaded Whisper model.

### Live Dictation Mode

**This is the real-time transcription engine:**

```bash
cargo run --bin yammer-cli dictate
```

What happens:
1. Starts capturing audio continuously
2. VAD detects when you start/stop speaking
3. Each speech segment is transcribed with Whisper
4. Text appears as you speak
5. Press Ctrl+C to stop

**With custom threshold:**
```bash
cargo run --bin yammer-cli dictate --threshold 0.015
```

**With time limit (auto-stop after 60 seconds):**
```bash
cargo run --bin yammer-cli dictate --duration 60
```

**With specific audio device:**
```bash
cargo run --bin yammer-cli dictate --device "Blue Yeti"
```

### How Dictation Mode Works

The dictation flow:
1. **Audio Capture**: Continuous 50ms audio chunks via cpal
2. **VAD Processing**: Each chunk analyzed for speech vs silence
3. **Segment Collection**: Speech chunks accumulated until silence detected
4. **Resampling**: Native sample rate → 16kHz mono for Whisper
5. **Transcription**: Whisper transcribes the audio segment
6. **Output**: Text printed to console in real-time

**Minimum segment duration**: 200ms (shorter segments are ignored as noise)

## Phase 4: LLM Text Correction (BLOCKED)

**Status**: Requires clang/libclang-dev installation

Once installed:
```bash
sudo apt install clang libclang-dev
```

Then you'll be able to test:
```bash
cargo run --bin yammer-cli correct "this is some transcribed text with errors"
```

The LLM will:
- Fix homophones (their/there/they're)
- Add punctuation
- Fix capitalization
- Remove filler words (um, uh, like)

## Testing Workflow: Full Pipeline

### 1. Check Your Setup
```bash
# Verify X11
echo $XDG_SESSION_TYPE  # Should show: x11

# List audio devices
cargo run --bin yammer-cli list-devices

# Check models
cargo run --bin yammer-cli list-models
```

### 2. Test Audio Capture
```bash
# Record 5 seconds
cargo run --bin yammer-cli record --duration 5 --output test.wav --resample

# Play it back
aplay test.wav
```

### 3. Test VAD
```bash
# Run VAD test, speak and watch state changes
cargo run --bin yammer-cli vad-test --duration 30

# Try different thresholds to find what works for your mic/environment
cargo run --bin yammer-cli vad-test --duration 20 --threshold 0.015
```

### 4. Test Transcription
```bash
# Record a longer sample
cargo run --bin yammer-cli record --duration 10 --output speech.wav --resample

# Transcribe it
cargo run --bin yammer-cli transcribe speech.wav
```

### 5. Test Live Dictation
```bash
# Start live dictation
cargo run --bin yammer-cli dictate

# Speak clearly: "The quick brown fox jumps over the lazy dog."
# Wait for silence
# Should see text appear

# Press Ctrl+C to stop
```

## Understanding Performance

### Whisper Processing Time

The Tiny English model is optimized for speed:
- ~100-300ms per second of audio (CPU)
- ~50-150ms per second with GPU acceleration

**To test processing speed:**
```bash
# Time a 10-second transcription
time cargo run --release --bin yammer-cli transcribe speech.wav
```

### VAD Latency

VAD is near-instant:
- Processes 50ms chunks in <1ms
- Hysteresis adds ~150-300ms smoothing
- Total speech start detection: ~200ms
- Total speech end detection: ~400ms (waits to ensure silence)

## Architecture Overview

### Audio Pipeline (Phase 2)
```
Microphone → cpal → AudioCapture → VadProcessor → Speech Segments
                         ↓
                    Resampler (48kHz → 16kHz)
```

**Key Components:**
- `yammer-audio/src/capture.rs` - cpal-based audio capture
- `yammer-audio/src/vad.rs` - Voice Activity Detection with hysteresis
- `yammer-audio/src/resample.rs` - Resampling to 16kHz mono

### STT Pipeline (Phase 3)
```
Audio Samples (16kHz mono) → Whisper Model → Transcript Segments → Text
```

**Key Components:**
- `yammer-stt/src/lib.rs` - whisper-rs integration
- Supports both file and streaming transcription
- Handles Whisper context management

### Model Management (Phase 1)
```
Model Registry → DownloadManager → Local Cache (~/.cache/yammer/models/)
```

**Key Components:**
- `yammer-core/src/models.rs` - Model registry and metadata
- `yammer-core/src/download.rs` - HTTP downloads with progress

## Common Issues

### No Audio Devices Found
```bash
# Install ALSA development libraries
sudo apt install libasound2-dev
```

### Transcription is Slow
- Make sure you're using `--release` build:
  ```bash
  cargo build --release
  ./target/release/yammer-cli dictate
  ```
- Check if GPU acceleration is working (requires CUDA)
- Consider using Whisper Tiny model (fastest)

### VAD Too Sensitive / Not Sensitive Enough
- Adjust threshold: `--threshold 0.005` (more sensitive) or `--threshold 0.03` (less sensitive)
- Check background noise level with `vad-test`
- Default 0.01 works for most quiet environments

### "Model not found" Error
```bash
# Download the default models
cargo run --bin yammer-cli download-models
```

### Audio Clicks/Pops in Recording
- This is usually a sample rate mismatch
- Always use `--resample` when recording for Whisper
- Check your microphone's native sample rate with `list-devices`

## Next Steps

Once clang is installed, the full pipeline will work:
1. **Speak** → VAD detects speech
2. **Audio captured** → Resampled to 16kHz
3. **Whisper transcribes** → Raw text
4. **LLM corrects** → Polished text
5. **Output** → Typed into active window (Phase 6)

For now, you can test steps 1-3 in isolation using the CLI commands above!

## File Locations

**Models:**
```bash
~/.cache/yammer/models/
```

**Test recordings:**
```bash
# Wherever you specify with --output
./test.wav
./speech.wav
```

**Source code:**
```
yammer-audio/    # Audio capture, VAD, resampling
yammer-stt/      # Whisper transcription
yammer-llm/      # LLM correction (blocked)
yammer-core/     # Models, config, shared types
yammer-cli/      # CLI commands for testing
```
