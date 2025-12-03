# Building a Linux Dictation App: Blockers and Implementation Details (X11 Only)

The proposed stack—Tauri, Rust, whisper-rs, llama_cpp-rs, and cpal on Ubuntu X11 with CUDA—is viable with **no critical blockers**. Targeting X11 exclusively sidesteps the significant complexity of Wayland's security model. The main implementation challenges are VRAM management between inference engines, streaming transcription architecture, and CUDA build configuration.

---

## Target Platform: X11 on GNOME

On Ubuntu 22.04+, select **"GNOME on Xorg"** at the login screen (gear icon). For a dedicated dictation machine or kiosk setup, you can make X11 the default by editing `/etc/gdm3/custom.conf`:

```ini
[daemon]
WaylandEnable=false
```

All Tauri window features work correctly on X11: transparent windows, chromeless/borderless mode, always-on-top, precise positioning, and drag regions.

---

## Tauri Window Configuration for Overlay UI

With X11, the floating overlay design works as intended:

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
        "alwaysOnTop": true,
        "skipTaskbar": true
      }
    ]
  }
}
```

CSS-based rounded corners with transparency work reliably. The `data-tauri-drag-region` attribute enables window dragging on chromeless windows.

---

## Global Hotkeys Work Natively on X11

The **global-hotkey** crate (from Tauri maintainers) provides straightforward key grabbing:

```rust
use global_hotkey::{GlobalHotKeyManager, hotkey::{HotKey, Modifiers, Code}};

let manager = GlobalHotKeyManager::new().unwrap();
let hotkey = HotKey::new(Some(Modifiers::SUPER), Code::KeyD);
manager.register(hotkey).unwrap();

// Listen via GlobalHotKeyEvent::receiver()
```

X11 allows applications to grab global key combinations directly—no special permissions, daemons, or user configuration required.

---

## Text Injection via xdotool

On X11, **xdotool** provides reliable simulated typing:

```rust
use std::process::Command;

fn type_text(text: &str) -> std::io::Result<()> {
    Command::new("xdotool")
        .args(["type", "--clearmodifiers", "--", text])
        .status()?;
    Ok(())
}
```

Install with `sudo apt install xdotool`. The `--clearmodifiers` flag prevents interference from held modifier keys (important if using a hotkey to trigger dictation).

For clipboard-based fallback (handles special characters more reliably):

```rust
use arboard::Clipboard;

fn paste_text(text: &str) -> Result<(), arboard::Error> {
    let mut clipboard = Clipboard::new()?;
    clipboard.set_text(text)?;
    // Simulate Ctrl+V
    Command::new("xdotool").args(["key", "ctrl+v"]).status()?;
    Ok(())
}
```

---

## whisper-rs CUDA Build Configuration

**Build requirements**: CUDA 11.x or 12.x toolkit with nvcc. cuDNN is not required. The build flag changed from `WHISPER_CUBLAS=1` to `GGML_CUDA=1`.

In your `Cargo.toml`:
```toml
[dependencies]
whisper-rs = { version = "0.14", features = ["cuda"] }
```

Set the CUDA architecture for your GPU before building:
```bash
export CUDA_ARCHITECTURES="86"  # RTX 30-series
# 75 for RTX 20-series/Turing
# 89 for RTX 40-series
cargo build --release
```

**Common build issues:**

| Issue | Solution |
|-------|----------|
| "nvcc not found" | Add CUDA toolkit bin to PATH |
| Architecture mismatch | Set `CUDA_ARCHITECTURES` for your GPU |
| Linking errors | Set `CUDA_PATH` environment variable |

---

## Streaming Transcription Architecture

whisper.cpp processes audio in **30-second segments** by design. For real-time dictation feedback, use a chunked approach:

1. **Capture audio continuously** via cpal
2. **Buffer 2-5 seconds** of speech
3. **Run transcription** on the buffer
4. **Display partial results** while capturing continues

whisper-rs doesn't expose a dedicated streaming API—you call `state.full(params, &audio_data)` repeatedly on accumulated audio. The `whisper-stream` example in whisper.cpp demonstrates this pattern.

**Latency expectations** (CUDA, RTX 3080):

| Model | 5s audio chunk |
|-------|----------------|
| tiny.en | ~100ms |
| base.en | ~200ms |
| small.en | ~400ms |
| medium.en | ~800ms |

For dictation, **base.en** or **small.en** offer a good accuracy/latency tradeoff.

---

## VAD (Voice Activity Detection) is External

whisper-rs does not include VAD. You need external silence detection to avoid processing dead air and to know when utterances end.

**Options:**

- **Silero VAD** (ONNX model, ~10ms detection latency) - the `voice-stream` crate wraps this
- **Simple energy-based detection** - compute RMS of audio frames, threshold for speech
- **webrtc-vad** crate - Google's VAD algorithm

Basic energy-based approach:
```rust
fn is_speech(samples: &[f32], threshold: f32) -> bool {
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();
    rms > threshold
}
```

Silero VAD is more robust to noise but adds an ONNX runtime dependency.

---

## llama_cpp-rs and whisper-rs Don't Share CUDA Memory

Both libraries create **independent CUDA contexts** with separate VRAM allocations. Total usage equals whisper + llama + overhead.

**VRAM budget for 8GB GPU:**

| Component | Typical Usage |
|-----------|---------------|
| Whisper base.en | ~142 MB |
| Whisper small.en | ~466 MB |
| Whisper medium.en | ~1.5 GB |
| Llama 3B Q4_K_M | ~2.0 GB |
| Llama 7B Q4_K_M | ~3.8 GB |
| KV cache (2048 ctx) | ~0.5-1 GB |
| CUDA runtime overhead | ~400 MB |

**Recommended approach for 8GB**: Use whisper small.en (~466MB) + llama 3B Q4_K_M (~2GB). Total ~3GB leaves headroom.

**For larger models or smaller VRAM**: Load sequentially—transcribe with whisper, drop the context, then load llama for correction:

```rust
// Transcription phase
let whisper_ctx = WhisperContext::new_with_params(...)?;
let transcript = transcribe(&whisper_ctx, audio)?;
drop(whisper_ctx);  // Free VRAM

// Correction phase  
let llama_model = LlamaModel::load_from_file(...)?;
let corrected = correct_text(&llama_model, &transcript)?;
```

---

## cpal Audio Capture and Resampling

cpal uses ALSA on Linux. PipeWire's ALSA compatibility layer handles routing transparently—same code works whether the system uses PulseAudio, PipeWire, or raw ALSA.

**16kHz is required** for Whisper, but most hardware captures at 44.1/48kHz. Use **Rubato** for real-time resampling:

```rust
use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

let params = SincInterpolationParameters {
    sinc_len: 256,
    f_cutoff: 0.95,
    interpolation: SincInterpolationType::Linear,
    oversampling_factor: 256,
    window: WindowFunction::BlackmanHarris2,
};

let mut resampler = SincFixedIn::<f32>::new(
    16000.0 / 48000.0,  // Resample ratio
    2.0,
    params,
    1024,  // Input chunk size
    1,     // Mono
)?;
```

**Buffer size**: Request smaller buffers for lower latency. cpal's ALSA default can be ~100ms; explicitly set `BufferSize::Fixed(512)` for ~10ms at 48kHz.

**Bluetooth microphones**: A2DP profile has no mic input. HSP/HFP works but at 8kHz telephone quality—prefer USB or built-in mics.

---

## Recommended Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Tauri Frontend                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │  Waveform   │  │   Status    │  │  Transcription  │  │
│  │   Canvas    │  │  Indicator  │  │    Display      │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │ Tauri Commands
┌────────────────────────┴────────────────────────────────┐
│                    Rust Backend                          │
│                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │   cpal   │───▶│  Rubato  │───▶│   Audio Buffer   │   │
│  │ capture  │    │ resample │    │   (ring buffer)  │   │
│  └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                           │              │
│  ┌──────────┐    ┌──────────┐    ┌────────▼─────────┐   │
│  │  global  │    │   VAD    │◀───│  whisper-rs      │   │
│  │  hotkey  │    │ (Silero) │    │  (CUDA)          │   │
│  └──────────┘    └──────────┘    └────────┬─────────┘   │
│                                           │              │
│  ┌──────────┐                   ┌─────────▼─────────┐   │
│  │ xdotool  │◀──────────────────│  llama_cpp-rs     │   │
│  │  output  │                   │  (CUDA, optional) │   │
│  └──────────┘                   └───────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Data flow:**

1. Global hotkey triggers recording start
2. cpal captures at native rate (48kHz)
3. Rubato resamples to 16kHz mono
4. VAD detects speech segments
5. whisper-rs transcribes speech chunks
6. (Optional) llama_cpp-rs corrects/formats text
7. xdotool types result into focused application

---

## Open Implementation Questions

- **LLM for correction**: What's the smallest effective model? Phi-3-mini (3.8B) and Qwen2-1.5B are candidates. The correction task is narrow—fixing dictation errors, adding punctuation—so smaller models may suffice.

- **Streaming UX**: Show partial transcription as it processes, or wait for complete utterance? Partial display feels more responsive but may show corrections/changes that feel jarring.

- **Error handling**: What happens when CUDA OOM occurs? Graceful fallback to CPU, or require user to choose smaller models?

- **Model management**: Bundle models with app, or download on first run? Whisper small.en is ~466MB, llama 3B is ~2GB—significant for bundling.

---

## Summary: Why X11 Simplifies Everything

By targeting X11 exclusively, the following Wayland complications are eliminated entirely:

| Feature | Wayland | X11 |
|---------|---------|-----|
| Window positioning | ❌ Not supported | ✅ Full control |
| Always-on-top | ❌ Returns error | ✅ Works |
| Chromeless windows | ⚠️ Requires workarounds | ✅ Works |
| Global hotkeys | ⚠️ Portal/evdev hacks | ✅ Native support |
| Text injection | ⚠️ Needs ydotool daemon | ✅ xdotool just works |

The tradeoff is requiring users to run GNOME on Xorg, but for a specialized dictation tool this is a reasonable constraint that avoids significant implementation complexity.
