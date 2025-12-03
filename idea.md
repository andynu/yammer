# Linux Dictation App Research Dossier

## Project Overview
Building a dictation application for **Linux** (Ubuntu-primary) with:
- Speech-to-text via faster-whisper (CTranslate2 backend)
- LLM-based text correction (potentially also via CTranslate2)
- Nice floating UI with waveform visualization (Tauri)
- Chromeless, rounded-edge window design

### Target Platform
| Component | Ubuntu Default | Notes |
|-----------|---------------|-------|
| **Distro** | Ubuntu 22.04+ | Primary target; broader Linux should work |
| **Display server** | Wayland | X11 still available as fallback |
| **Compositor** | Mutter (GNOME) | Handles transparency well |
| **Audio server** | PipeWire | ALSA compat layer works transparently |
| **GPU** | Varies | NVIDIA needs proprietary drivers for CUDA |

### Broader Linux Support
The stack (cpal, Tauri, whisper-rs/llama_cpp) is Linux-generic. Main distro differences:
- Package names for build dependencies
- CUDA installation method (if using GPU)

If it works on Ubuntu, it will almost certainly work on Fedora, Arch, etc. with minimal changes. Edge cases: minimal window managers without proper compositing.

---

## Ubuntu Build Dependencies (All-in-One)

```bash
# System libraries for Tauri + Audio + Build tools
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libasound2-dev \
    pkg-config \
    cmake \
    clang

# Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Node.js (for Tauri frontend tooling)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs

# Optional: CUDA (for GPU inference)
# Follow NVIDIA's official guide for your GPU
```

### For Other Distros
| Distro | ALSA Dev | WebKitGTK |
|--------|----------|-----------|
| Fedora | `alsa-lib-devel` | `webkit2gtk4.1-devel` |
| Arch | `alsa-lib` | `webkit2gtk-4.1` |

---

## 1. CTranslate2 for Both STT and LLM Inference

### What It Is
CTranslate2 is a C++ and Python library for efficient inference with Transformer models. It applies performance optimizations including weights quantization, layers fusion, batch reordering, etc.

### Supported Model Types
- **Encoder-decoder**: Transformer, M2M-100, NLLB, BART, mBART, Pegasus, T5, **Whisper**
- **Decoder-only (LLMs)**: GPT-2, GPT-J, GPT-NeoX, OPT, BLOOM, MPT, **Llama**, **Mistral**, Gemma, CodeGen, GPTBigCode, Falcon, **Qwen2**
- **Encoder-only**: BERT, DistilBERT, XLM-RoBERTa

### Key Benefits of Using CTranslate2 for Both
- **Single CUDA context** - both Whisper and LLM share GPU memory efficiently
- **Unified dependency** - one inference engine instead of two
- **Quantization support** - FP16, INT16, INT8 for reduced memory
- **CPU and GPU support** - works on both, with good CPU performance

### Model Conversion
Models must be converted to CTranslate2 format first:
```bash
# For LLMs (e.g., Llama, Mistral)
ct2-transformers-converter --model meta-llama/Llama-2-7b-chat-hf \
  --quantization float16 --output_dir llama-2-7b-ct2

# Whisper is handled by faster-whisper automatically
```

### Python API for LLM Generation
```python
import ctranslate2
import transformers

generator = ctranslate2.Generator("llama-2-7b-ct2", device="cuda")
tokenizer = transformers.AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

text = "Fix this dictation: "
start_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
results = generator.generate_batch([start_tokens], max_length=256, sampling_topk=10)
print(tokenizer.decode(results[0].sequences_ids[0]))
```

### Research Questions
- [ ] What's the smallest effective LLM for dictation correction? (Qwen2-1.5B? Phi-3-mini?)
- [ ] Can both models share a CUDA context, or do they need separate contexts?
- [ ] Memory requirements for running both simultaneously

### Resources
- GitHub: https://github.com/OpenNMT/CTranslate2
- Transformers guide: https://github.com/OpenNMT/CTranslate2/blob/master/docs/guides/transformers.md

---

## 2. Faster-Whisper

### What It Is
A reimplementation of OpenAI's Whisper using CTranslate2 for the backend. Significantly faster than the original Python implementation with lower memory usage.

### Key Features
- **No FFmpeg required** - uses PyAV for audio decoding
- **GPU acceleration** - requires CUDA 12 + cuDNN 9 (or downgrade ctranslate2 for CUDA 11)
- **VAD (Voice Activity Detection)** - built-in silence detection
- **Batched transcription** - process multiple segments in parallel
- **Word-level timestamps** - precise timing information

### Installation
```bash
pip install faster-whisper
# GPU requires: CUDA 12.x + cuDNN 9
```

### Basic Usage
```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cuda", compute_type="float16")
# Models: tiny, base, small, medium, large-v2, large-v3

segments, info = model.transcribe("audio.wav", 
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500))

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Audio Requirements
- **Sample rate**: 16kHz
- **Channels**: Mono
- **Format**: Accepts various formats via PyAV

### Real-time Streaming Options
- **WhisperLive** - real-time implementation using faster-whisper backend
- **Whisper-Streaming** - streaming with self-adaptive latency

### Model Sizes and Speed
| Model | Parameters | Relative Speed | English WER |
|-------|------------|----------------|-------------|
| tiny | 39M | ~32x | ~10% |
| base | 74M | ~16x | ~7% |
| small | 244M | ~6x | ~5% |
| medium | 769M | ~2x | ~4% |
| large-v3 | 1550M | 1x | ~3% |

### Research Questions
- [ ] Which model size gives best latency/accuracy tradeoff for dictation?
- [ ] How to implement streaming transcription for real-time feedback?
- [ ] Integration path with Rust backend (see whisper-rs)

### Resources
- GitHub: https://github.com/SYSTRAN/faster-whisper
- PyPI: https://pypi.org/project/faster-whisper/

---

## 3. Audio Capture in Rust (Linux)

### Primary Library: CPAL
Cross-platform audio I/O library in pure Rust. The standard choice for audio in Rust.

### How It Works on Ubuntu
Ubuntu 22.04+ uses **PipeWire** as the audio server, but cpal talks to it through the **ALSA compatibility layer**. This means:
- You don't need to care whether the user has PipeWire or PulseAudio
- cpal just uses ALSA APIs, and PipeWire handles the translation
- Same code works on older Ubuntu with PulseAudio

### Ubuntu Dependencies
```bash
# Required for building
sudo apt install libasound2-dev

# For other distros:
# Fedora: sudo dnf install alsa-lib-devel
# Arch: sudo pacman -S alsa-lib
```

### Basic Microphone Capture
```rust
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

fn main() {
    let host = cpal::default_host();  // Uses ALSA on Linux
    let device = host.default_input_device()
        .expect("No input device available");
    
    let config = device.default_input_config()
        .expect("Failed to get default input config");
    
    println!("Input device: {}", device.name().unwrap());
    println!("Config: {:?}", config);
    
    // Buffer to collect samples
    let samples: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let samples_clone = samples.clone();
    
    let stream = device.build_input_stream(
        &config.into(),
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            samples_clone.lock().unwrap().extend_from_slice(data);
        },
        |err| eprintln!("Stream error: {}", err),
        None
    ).unwrap();
    
    stream.play().unwrap();
    // ... capture audio ...
}
```

### Resampling to 16kHz (for Whisper)
Whisper requires 16kHz mono audio. Use the `rubato` crate for resampling:

```rust
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, 
             SincInterpolationType, WindowFunction};

let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
};

let mut resampler = SincFixedIn::<f64>::new(
    16000.0 / input_sample_rate as f64,  // ratio
    2.0,                                   // max relative ratio
    params,
    1024,                                  // chunk size
    1,                                     // channels (mono)
).unwrap();
```

### PipeWire-Specific Notes
If you ever need direct PipeWire access (e.g., for more advanced routing):
- The `pipewire` crate provides Rust bindings
- But for simple mic capture, cpal via ALSA is simpler and sufficient

### Related Crates
- `cpal` - Audio I/O
- `rubato` - Resampling
- `hound` - WAV file reading/writing
- `symphonia` - Audio decoding (many formats)

### Resources
- cpal GitHub: https://github.com/RustAudio/cpal
- Detailed example: https://janhalozan.com/2024/07/01/jarvis-part-1-microphone/

---

## 4. Tauri Frontend

### What It Is
Tauri is a toolkit for building desktop apps with web technologies (HTML/CSS/JS) and a Rust backend. Much lighter than Electron.

### Why Tauri for This Project
- **Small binary size** - uses system webview (WebKitGTK on Linux)
- **Rust backend** - natural fit for audio processing
- **Web frontend** - easy to build visualizations with Canvas/WebGL
- **Native Linux support** - works well with GTK

### Ubuntu Dependencies
```bash
sudo apt update
sudo apt install libwebkit2gtk-4.1-dev \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev
```

### Project Structure
```
my-app/
├── src-tauri/           # Rust backend
│   ├── src/
│   │   └── main.rs
│   ├── Cargo.toml
│   └── tauri.conf.json
└── src/                 # Web frontend
    ├── index.html
    ├── main.js
    └── styles.css
```

### Tauri Commands (Rust → JS communication)
```rust
// src-tauri/src/main.rs
#[tauri::command]
fn transcribe_audio(audio_data: Vec<f32>) -> Result<String, String> {
    // Process audio with Whisper
    Ok("transcribed text".to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![transcribe_audio])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```javascript
// Frontend
import { invoke } from '@tauri-apps/api/tauri';

const result = await invoke('transcribe_audio', { audioData: samples });
```

### Resources
- Official docs: https://v2.tauri.app/
- Window customization: https://v2.tauri.app/learn/window-customization/

---

## 5. Chromeless Rounded Windows in Tauri (Linux)

### The Approach
On Linux (Ubuntu/GNOME), we use transparent windows with CSS-based rounded corners. This works well with Mutter compositor on both Wayland and X11.

### Configuration
In `tauri.conf.json`:
```json
{
  "tauri": {
    "windows": [
      {
        "title": "Dictation",
        "width": 300,
        "height": 100,
        "decorations": false,
        "transparent": true,
        "resizable": false,
        "alwaysOnTop": true
      }
    ]
  }
}
```

### CSS-Based Rounded Corners
```css
/* styles.css */
* {
    margin: 0;
    padding: 0;
}

html, body {
    background-color: transparent;
    height: 100%;
    overflow: hidden;
}

.app-container {
    background: rgba(30, 30, 30, 0.95);
    border-radius: 16px;
    height: 100%;
    width: 100%;
    /* Add shadow for depth */
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}
```

```html
<body>
    <div class="app-container">
        <canvas id="waveform"></canvas>
        <!-- other UI elements -->
    </div>
</body>
```

### Making the Window Draggable
Since there's no title bar:
```html
<div class="app-container" data-tauri-drag-region>
    <!-- content -->
</div>
```

### Wayland vs X11 Considerations

Check which you're running:
```bash
echo $XDG_SESSION_TYPE
```

| Feature | Wayland | X11 |
|---------|---------|-----|
| Transparency | ✅ Works with Mutter | ✅ Works |
| Always on top | ✅ Works | ✅ Works |
| Global hotkeys | ⚠️ Restricted (see below) | ✅ Works |
| Window positioning | ⚠️ Limited | ✅ Full control |

**Wayland Hotkey Workaround**: Wayland restricts apps from grabbing global hotkeys for security. Options:
1. Use a portal (xdg-desktop-portal) - complex but proper
2. Bind hotkey in GNOME Settings → Keyboard → Custom Shortcuts to launch/toggle the app
3. Use a separate small daemon that talks to the app via IPC

### GNOME-Specific Tips
- Transparent windows work out of the box with Mutter
- For development, you can test on X11 by choosing "GNOME on Xorg" at login
- `alwaysOnTop` may require user confirmation on some GNOME versions

### Resources
- Tauri window customization: https://v2.tauri.app/learn/window-customization/

---

## 6. Audio Waveform Visualization (Frontend)

### Web Audio API Approach
The frontend can capture audio and visualize it using the Web Audio API with Canvas.

### Basic Waveform Visualizer
```javascript
// Get microphone input
const audioContext = new AudioContext();
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const source = audioContext.createMediaStreamSource(stream);

// Create analyzer
const analyser = audioContext.createAnalyser();
analyser.fftSize = 2048;
source.connect(analyser);

const bufferLength = analyser.frequencyBinCount;
const dataArray = new Uint8Array(bufferLength);

// Canvas setup
const canvas = document.getElementById('waveform');
const canvasCtx = canvas.getContext('2d');

function draw() {
    requestAnimationFrame(draw);
    
    analyser.getByteTimeDomainData(dataArray);
    
    canvasCtx.fillStyle = 'rgb(30, 30, 30)';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = 'rgb(0, 255, 100)';
    canvasCtx.beginPath();
    
    const sliceWidth = canvas.width / bufferLength;
    let x = 0;
    
    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

draw();
```

### Alternative: Send Audio Data from Rust
If capturing audio in Rust (via cpal), you can send the samples to the frontend:

```rust
// Rust side - emit audio data to frontend
app.emit_all("audio-samples", &audio_buffer)?;
```

```javascript
// Frontend - receive and visualize
import { listen } from '@tauri-apps/api/event';

listen('audio-samples', (event) => {
    const samples = event.payload;
    drawWaveform(samples);
});
```

### Libraries for Visualization
- **wavesurfer.js** - Full-featured waveform library
- **Web Audio API + Canvas** - DIY approach (more control)

### Research Questions
- [ ] Capture audio in Rust (for Whisper) vs. browser (for visualization) - or both?
- [ ] Latency considerations for real-time visualization
- [ ] Styled waveform designs (bars, smooth curves, circular)

### Resources
- MDN Web Audio Visualization: https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Visualizations_with_Web_Audio_API
- wavesurfer.js: https://wavesurfer.xyz/

---

## 7. Rust Whisper Bindings

### Option A: whisper-rs (whisper.cpp bindings)
The most mature option. Binds to whisper.cpp (C++ implementation).

```rust
use whisper_rs::{WhisperContext, WhisperContextParameters, FullParams, SamplingStrategy};

let ctx = WhisperContext::new_with_params(
    "ggml-base.bin",
    WhisperContextParameters::default()
).expect("failed to load model");

let params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

// Audio must be: mono, 16kHz, f32
let audio_data: Vec<f32> = load_audio();

let mut state = ctx.create_state().expect("failed to create state");
state.full(params, &audio_data).expect("failed to run model");

let num_segments = state.full_n_segments().expect("failed to get segments");
for i in 0..num_segments {
    let segment = state.full_get_segment_text(i).expect("failed to get segment");
    println!("{}", segment);
}
```

**Features:**
- `cuda` - CUDA GPU acceleration
- `hipblas` - ROCm/AMD GPU support
- Built-in VAD support

**Caveats:**
- Uses whisper.cpp, not faster-whisper/CTranslate2
- Requires model in GGML format

### Option B: faster-whisper-rs
High-level bindings to faster-whisper Python API (via PyO3).

```rust
use faster_whisper_rs::WhisperModel;

let fw = WhisperModel::default();
let transcript = fw.transcribe("audio.wav".to_string())?;
println!("{}", transcript);
```

**Caveats:**
- Wraps Python, so requires Python runtime
- Less mature than whisper-rs

### Option C: Call Python from Rust
Use PyO3 to call faster-whisper directly:

```rust
use pyo3::prelude::*;

Python::with_gil(|py| {
    let faster_whisper = py.import("faster_whisper")?;
    let model = faster_whisper.getattr("WhisperModel")?.call1(("base",))?;
    let result = model.call_method1("transcribe", ("audio.wav",))?;
    // Process result...
});
```

### Recommendation
For CTranslate2 consistency with LLM, **Option C** (calling faster-whisper from Rust via PyO3) keeps everything on the same inference backend. However, for pure Rust simplicity, **whisper-rs** is more straightforward.

### Resources
- whisper-rs: https://github.com/tazz4843/whisper-rs (now on Codeberg)
- faster-whisper-rs: https://crates.io/crates/faster-whisper-rs

---

## 8. Rust LLM Bindings

### llama_cpp (edgenai)
High-level, safe bindings to llama.cpp:

```rust
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use llama_cpp::standard_sampler::StandardSampler;

let model = LlamaModel::load_from_file("model.gguf", LlamaParams::default())?;
let mut ctx = model.create_session(SessionParams::default())?;

ctx.advance_context("Fix this dictation: ")?;

// Generate tokens
let mut decoded = 0;
while let Some(token) = ctx.start_completing_with(StandardSampler::default())? {
    print!("{}", token);
    decoded += 1;
    if decoded >= 256 { break; }
}
```

**Features:**
- `cuda` - NVIDIA GPU
- `vulkan` - Vulkan backend
- `metal` - Apple Silicon

### Resources
- llama_cpp-rs: https://github.com/edgenai/llama_cpp-rs
- llama_cpp docs: https://docs.rs/llama_cpp

---

## 9. Text Output Methods (Linux)

### Option A: Clipboard
Simplest approach—copy text to clipboard, user pastes with Ctrl+V.

```rust
// Using arboard crate
use arboard::Clipboard;

let mut clipboard = Clipboard::new().unwrap();
clipboard.set_text("transcribed text").unwrap();
```

**Pros**: Works everywhere, no special permissions
**Cons**: Extra step for user (must paste)

### Option B: Simulated Typing (X11)
Use `xdotool` to type text directly into the focused window.

```rust
use std::process::Command;

fn type_text(text: &str) {
    Command::new("xdotool")
        .args(["type", "--clearmodifiers", text])
        .output()
        .expect("xdotool failed");
}
```

```bash
# Install xdotool
sudo apt install xdotool
```

**Pros**: Seamless—text appears where cursor is
**Cons**: X11 only, doesn't work on Wayland

### Option C: Simulated Typing (Wayland)
Use `ydotool` for Wayland (requires a daemon).

```bash
# Install ydotool
sudo apt install ydotool

# Start the daemon (needs to run as root or with uinput permissions)
sudo ydotoold &

# Type text
ydotool type "transcribed text"
```

**Pros**: Works on Wayland
**Cons**: Requires daemon with elevated permissions, more complex setup

### Option D: wtype (Wayland-native)
Alternative to ydotool, uses Wayland's input-method protocol.

```bash
# Install wtype (may need to build from source)
wtype "transcribed text"
```

**Cons**: Only works with compositors that support the virtual-keyboard-v1 protocol

### Recommendation
Start with **clipboard** (Option A) for simplicity. Add optional typing simulation as a feature:
- Detect X11 vs Wayland at runtime
- Use xdotool on X11, ydotool/wtype on Wayland
- Fall back to clipboard if tools unavailable

```rust
fn output_text(text: &str, use_typing: bool) {
    if use_typing {
        if std::env::var("XDG_SESSION_TYPE").unwrap_or_default() == "x11" {
            // Use xdotool
        } else {
            // Try ydotool or fall back to clipboard
        }
    } else {
        // Use clipboard
    }
}
```

---

## 10. Global Hotkeys (Linux)

### The Challenge
Global hotkeys (e.g., press a key anywhere to start/stop dictation) work differently on X11 vs Wayland.

### X11: Direct Key Grabbing
On X11, apps can grab keys globally using the X11 API. The `global-hotkey` crate from Tauri works here.

```rust
use global_hotkey::{GlobalHotKeyManager, hotkey::{HotKey, Modifiers, Code}};

let manager = GlobalHotKeyManager::new().unwrap();
let hotkey = HotKey::new(Some(Modifiers::SUPER), Code::KeyD);
manager.register(hotkey).unwrap();

// Listen for hotkey events...
```

### Wayland: Restricted by Design
Wayland intentionally restricts global key grabbing for security. Options:

**Option 1: GNOME Custom Shortcut (Recommended)**
Let the user bind a shortcut in GNOME Settings that launches/toggles your app.

```bash
# Your app can be toggled via a DBus call or by checking if already running
# GNOME Settings → Keyboard → Custom Shortcuts → Add
# Command: /path/to/dictation-app --toggle
```

**Option 2: Use xdg-desktop-portal**
The portal provides a `GlobalShortcuts` interface, but support varies by compositor.

**Option 3: Small X11 Helper Daemon**
Run a tiny X11 app (even on Wayland, XWayland can grab keys) that communicates with your main app via IPC.

**Option 4: Use a Hotkey Daemon**
Tools like `sxhkd` (X11) or `keyd` (kernel-level) can trigger your app.

### Practical Recommendation
For Ubuntu/GNOME:
1. Document how to set up a GNOME custom shortcut
2. Have the app accept a `--toggle` flag that shows/hides the window
3. Optionally support X11 key grabbing for users who prefer X11

```rust
// In main.rs - handle --toggle flag
fn main() {
    let args: Vec<String> = std::env::args().collect();
    
    if args.contains(&"--toggle".to_string()) {
        // Send toggle message to running instance via IPC
        // If no instance running, start normally
    }
    
    // Normal startup...
}
```

---

## Architecture Decision: Pure Rust vs. Hybrid

### Option A: Pure Rust
- **STT**: whisper-rs (whisper.cpp)
- **LLM**: llama_cpp-rs (llama.cpp)
- **Audio**: cpal
- **UI**: Tauri

**Pros**: Single language, no Python dependency, simpler deployment (single binary + models)
**Cons**: Two different inference backends (ggml), potentially less optimized than CTranslate2

### Option B: Rust + Python Hybrid
- **STT**: faster-whisper (Python, via PyO3)
- **LLM**: CTranslate2 (Python, via PyO3)
- **Audio**: cpal (Rust)
- **UI**: Tauri

**Pros**: CTranslate2 optimization, unified backend for both models
**Cons**: Python dependency, more complex deployment

### Option C: Subprocess Architecture
- **Rust app**: Audio capture, UI, orchestration
- **Python subprocess**: faster-whisper + CTranslate2 LLM

**Pros**: Clean separation, easier to debug/develop independently
**Cons**: IPC overhead, more moving parts

### Recommendation
Start with **Option A (Pure Rust)** for simplicity. The llama.cpp and whisper.cpp backends are highly optimized and actively maintained. Deployment is cleaner—just a single binary plus model files. You can always add CTranslate2 later if performance becomes an issue.

---

## Next Steps

1. **Prototype audio capture** - Get cpal working with your microphone
2. **Test whisper-rs** - Transcribe a test audio file
3. **Create basic Tauri window** - Transparent, rounded, floating
4. **Add waveform visualization** - Canvas in the frontend
5. **Integrate pieces** - Audio → Whisper → Display
6. **Add LLM correction** - Post-process transcription
7. **Polish UI** - States (listening, processing, done), animations

---

## Open Questions

- [ ] Wayland or X11? (run `echo $XDG_SESSION_TYPE` to check)
- [ ] GPU (CUDA) or CPU-only inference?
- [ ] Preferred LLM size for corrections (balance speed vs quality)?
- [ ] How should text be output? Options:
  - Clipboard (simple, works everywhere)
  - `xdotool` / `ydotool` for simulated typing (more seamless but X11/Wayland dependent)
  - Both (clipboard + optional typing simulation)
- [ ] Global hotkey approach:
  - On X11: Can grab keys directly
  - On Wayland: Use GNOME custom shortcut to toggle app, or a small helper daemon
