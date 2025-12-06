# Yammer Onboarding Guide

Yammer is a **local, on-device dictation application for Linux** that converts speech to text using AI. The entire pipeline runs locally without cloud services - your voice data never leaves your machine.

## What Does Yammer Do?

Press a hotkey (Ctrl+Alt+D) and speak. Yammer will:
1. Record your voice until you stop speaking
2. Transcribe it using Whisper (local AI model)
3. Correct grammar and punctuation using TinyLlama (local LLM)
4. Type the corrected text into whatever application is focused

---

## Architecture Overview

Yammer is a **Rust workspace** with 7 crates, each handling a specific responsibility:

| Crate | Purpose | Key Dependencies |
|-------|---------|------------------|
| `yammer-core` | Configuration, model management, downloads | tokio, reqwest, serde |
| `yammer-audio` | Microphone capture, VAD, resampling | cpal, rubato |
| `yammer-stt` | Whisper speech-to-text | whisper-rs (CUDA) |
| `yammer-llm` | LLM text correction | llama_cpp |
| `yammer-output` | Text injection via xdotool/clipboard | - |
| `yammer-cli` | Command-line interface for testing | clap |
| `yammer-app` | Tauri GUI with waveform visualization | tauri v2 |

---

## Component Diagram

![Component Architecture](components.svg)

<details>
<summary>D2 Source (components.d2)</summary>

```d2
# Yammer Component Architecture

direction: right

user: User {
  shape: person
}

# External Dependencies
external: External {
  mic: Microphone
  xdotool: xdotool
  xclip: xclip
  app: Target App
}

# Rust Workspace Crates
workspace: yammer workspace {

  core: yammer-core {
    config: Config (TOML)
    model: Model Registry
    download: Download Manager
  }

  audio: yammer-audio {
    capture: Audio Capture (cpal)
    vad: Voice Activity Detection
    resample: Resampler (16kHz)
  }

  stt: yammer-stt {
    whisper: Whisper Transcriber
  }

  llm: yammer-llm {
    corrector: LLM Corrector
  }

  output: yammer-output {
    textout: Text Output
  }

  cli: yammer-cli {
    commands: CLI Commands
  }

  app: yammer-app (Tauri) {
    backend: Rust Backend
    frontend: Web Frontend
    pipeline: Dictation Pipeline
  }
}

# Models on disk
models: AI Models {
  whisper_model: Whisper GGML
  llm_model: TinyLlama GGUF
}

# Dependencies between crates
workspace.audio -> workspace.core: uses config
workspace.stt -> workspace.core: uses config
workspace.llm -> workspace.core: uses config
workspace.output -> workspace.core: uses config

workspace.cli -> workspace.audio: audio testing
workspace.cli -> workspace.stt: transcribe cmd
workspace.cli -> workspace.llm: correct cmd
workspace.cli -> workspace.output: dictate output
workspace.cli -> workspace.core: config cmds

workspace.app.pipeline -> workspace.audio: capture
workspace.app.pipeline -> workspace.stt: transcribe
workspace.app.pipeline -> workspace.llm: correct
workspace.app.pipeline -> workspace.output: output

# External connections
user -> external.mic: speaks
external.mic -> workspace.audio.capture: audio stream
workspace.output.textout -> external.xdotool: type command
workspace.output.textout -> external.xclip: clipboard
external.xdotool -> external.app: keystrokes
external.xclip -> external.app: paste

# Model loading
workspace.core.download -> models: downloads
workspace.stt.whisper -> models.whisper_model: loads
workspace.llm.corrector -> models.llm_model: loads
```

</details>

---

## Call Flow Diagram

![Dictation Call Flow](callflow.svg)

<details>
<summary>D2 Source (callflow.d2)</summary>

```d2
# Yammer Dictation Call Flow

direction: down

# Trigger
hotkey: User presses Ctrl+Alt+D

# Phase 1: Audio Capture
phase1: 1. Audio Capture {
  start: Start Recording
  capture: cpal AudioCapture
  chunks: Audio Chunks (50ms)
  vad: VAD Processor
  detect: Speech Detection
  resample: Resample to 16kHz
  samples: Audio Samples

  start -> capture
  capture -> chunks
  chunks -> vad
  vad -> detect
  detect -> resample: speech end
  resample -> samples
}

# Phase 2: Transcription
phase2: 2. Transcription {
  load: Load Whisper Model
  infer: Whisper Inference
  filter: Filter Hallucinations
  transcript: Raw Transcript

  load -> infer
  infer -> filter
  filter -> transcript
}

# Phase 3: LLM Correction
phase3: 3. LLM Correction {
  prompt: Build Correction Prompt
  llm: TinyLlama Inference
  clean: Clean Output
  corrected: Corrected Text

  prompt -> llm
  llm -> clean
  clean -> corrected
}

# Phase 4: Output
phase4: 4. Text Output {
  method: Check Output Method
  type: xdotool type
  clipboard: xclip + Ctrl+V
  done: Text in Target App

  method -> type: type method
  method -> clipboard: clipboard method
  type -> done
  clipboard -> done
}

# Main flow between phases
hotkey -> phase1.start: trigger
phase1.samples -> phase2.load: 16kHz audio
phase2.transcript -> phase3.prompt: raw text
phase3.corrected -> phase4.method: final text

# UI Updates (parallel)
ui: GUI Updates {
  waveform: Waveform Display
  status: Status Indicator
  text: Transcript Display
}

phase1.chunks -> ui.waveform: audio levels
phase1 -> ui.status: Listening
phase2 -> ui.status: Processing
phase3 -> ui.status: Correcting
phase4 -> ui.status: Done
phase2.transcript -> ui.text: show text
phase3.corrected -> ui.text: update text
```

</details>

---

## Key Files to Understand

### Start Here (in order)

| File | Lines | What You'll Learn |
|------|-------|-------------------|
| `yammer-core/src/config.rs` | 371 | Configuration system, all settings |
| `yammer-core/src/model.rs` | 135 | Available AI models, download URLs |
| `yammer-audio/src/capture.rs` | 481 | How audio is captured with cpal |
| `yammer-audio/src/vad.rs` | 372 | Voice activity detection algorithm |
| `yammer-stt/src/transcriber.rs` | 348 | Whisper integration |
| `yammer-llm/src/corrector.rs` | 210 | LLM text correction |
| `yammer-output/src/lib.rs` | 175 | Text output via xdotool |
| `yammer-app/src-tauri/src/pipeline.rs` | ~500 | Full dictation pipeline orchestration |

### Configuration

Location: `~/.config/yammer/config.toml`

```toml
[hotkey]
modifiers = ["Control", "Alt"]
key = "D"

[models]
whisper = "base.en"      # tiny.en, base.en, small.en
llm = "tinyllama-1.1b"   # or "none" to disable

[audio]
vad_threshold = 0.01     # Speech detection sensitivity

[output]
method = "type"          # or "clipboard"
```

### Model Storage

Location: `~/.cache/yammer/models/`

| Model | Size | Purpose |
|-------|------|---------|
| `ggml-base.en.bin` | 141 MB | Whisper speech recognition |
| `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 638 MB | Grammar/punctuation correction |

---

## Running Yammer

### GUI (Production)

```bash
# Build the Tauri app
cd yammer-app && npm run tauri build

# Run (from anywhere)
./target/release/yammer gui
# Or
./target/release/yammer-app
```

### CLI Commands (Development/Testing)

```bash
# Download models first
./target/release/yammer download-models

# Test audio capture
./target/release/yammer list-devices
./target/release/yammer record --duration 5 --output test.wav

# Test VAD (voice detection)
./target/release/yammer vad-test --duration 30

# Test transcription
./target/release/yammer transcribe test.wav

# Test LLM correction
./target/release/yammer correct "im going to the store do you want anything"

# Full dictation test (no GUI)
./target/release/yammer dictate
```

---

## Data Flow Summary

```
User speaks into microphone
        │
        ▼
┌───────────────────────────────────────┐
│  PHASE 1: Audio Capture (yammer-audio)│
│  • cpal captures multi-channel audio  │
│  • VAD detects speech start/end       │
│  • Resample to 16kHz mono             │
└───────────────────────────────────────┘
        │ Vec<f32> @ 16kHz
        ▼
┌───────────────────────────────────────┐
│  PHASE 2: Transcription (yammer-stt)  │
│  • Whisper GGML model inference       │
│  • Filter hallucinations              │
│  • ~100-1500ms for 5s audio           │
└───────────────────────────────────────┘
        │ "im going to the store"
        ▼
┌───────────────────────────────────────┐
│  PHASE 3: Correction (yammer-llm)     │
│  • TinyLlama with few-shot prompt     │
│  • Fix grammar, punctuation, caps     │
│  • ~100-500ms                         │
└───────────────────────────────────────┘
        │ "I'm going to the store."
        ▼
┌───────────────────────────────────────┐
│  PHASE 4: Output (yammer-output)      │
│  • xdotool type (simulates keys)      │
│  • or clipboard + Ctrl+V paste        │
└───────────────────────────────────────┘
        │
        ▼
Text appears in focused application
```

---

## System Requirements

- **OS:** Linux with X11 (Wayland not yet supported)
- **RAM:** 4GB minimum, 8GB recommended
- **Storage:** ~800MB for models
- **GPU:** Optional (CUDA accelerates Whisper transcription)

### Required Packages (Ubuntu/Debian)

```bash
sudo apt install \
  xdotool \
  xclip \
  libasound2-dev \
  libayatana-appindicator3-dev \
  libclang-dev
```

---

## Architecture Decisions

1. **Local-only processing** - Privacy by design, no cloud APIs
2. **Whisper on GPU, LLM on CPU** - Avoids CUDA symbol conflicts between whisper-rs and llama_cpp
3. **Energy-based VAD** - Simple RMS threshold, no ML model needed
4. **Modular crates** - Each stage is independently testable
5. **Tauri v2** - Modern desktop app with small binary size

---

## Next Steps

1. **Run `./target/release/yammer download-models`** to get the AI models
2. **Try `./target/release/yammer dictate`** for CLI-based dictation
3. **Launch `./target/release/yammer gui`** for the full GUI experience
4. **Explore the config** at `~/.config/yammer/config.toml`
