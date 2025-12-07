# yammer-audio Onboarding Guide

Welcome to yammer-audio, the audio capture and processing crate for the Yammer dictation application. This guide will help you understand how audio flows from your microphone to speech-ready samples.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### Why Audio Capture Matters for Dictation

Dictation converts spoken words into text. The first step is capturing audio from the microphone in a format that speech recognition can process. This involves:

- Capturing raw audio samples from the hardware
- Converting to a consistent format (mono, specific sample rate)
- Detecting when the user is actually speaking (vs silence/noise)
- Buffering speech segments for transcription

### The Linux Audio Stack

Audio on Linux flows through several layers:

```
Microphone Hardware
       ↓
    ALSA (kernel driver)
       ↓
PulseAudio or PipeWire (sound server)
       ↓
    cpal (cross-platform audio library)
       ↓
    Your Application
```

**ALSA** (Advanced Linux Sound Architecture) provides the kernel-level audio drivers. Applications rarely talk to ALSA directly.

**PulseAudio** and **PipeWire** are sound servers that handle routing, mixing, and device management. PipeWire is the newer replacement, common on modern distros.

**cpal** abstracts over all of this, giving us a cross-platform API that works on Linux, macOS, and Windows without code changes.

### Audio Fundamentals

Key concepts you'll encounter:

| Term | Meaning |
|------|---------|
| **Sample** | A single measurement of audio amplitude at one instant |
| **Sample Rate** | How many samples per second (e.g., 48000 Hz = 48,000 samples/second) |
| **Channels** | Mono (1) or stereo (2). We convert to mono for speech |
| **Bit Depth** | Precision of each sample (16-bit, 32-bit float). We use f32 internally |
| **Frame** | One sample from each channel (2 samples for stereo) |

Common sample rates:
- 44100 Hz: CD quality, common default
- 48000 Hz: Video/professional audio standard
- 16000 Hz: Speech recognition standard (Whisper uses this)

### Voice Activity Detection (VAD)

VAD answers: "Is the user speaking right now?"

Why it matters:
- **Efficiency**: Don't transcribe silence
- **Accuracy**: Send complete utterances to Whisper
- **UX**: Know when to start/stop recording

Our approach: Energy-based detection with hysteresis (explained below).

---

## Library Choices

### Why cpal?

We chose [cpal](https://github.com/RustAudio/cpal) for audio capture:

| Alternative | Why Not |
|-------------|---------|
| **Direct ALSA** | Linux-only, complex API, no hot-plug support |
| **PortAudio** | C library, FFI complexity, cpal is pure Rust |
| **rodio** | Built on cpal, adds playback features we don't need |

cpal provides:
- Cross-platform support (Linux/macOS/Windows)
- Device enumeration and hot-plug detection
- Callback-based streaming (low latency)
- Support for multiple sample formats

### Why rubato for Resampling?

Whisper requires 16kHz audio. Most microphones capture at 44.1kHz or 48kHz. We need high-quality resampling.

[rubato](https://github.com/HEnquist/rubato) is a Rust resampling library using FFT-based algorithms:

| Alternative | Why rubato wins |
|-------------|-----------------|
| **Linear interpolation** | Audible artifacts, frequency aliasing |
| **libsamplerate** | C library, FFI overhead |
| **dasp** | Good, but rubato has better FFT performance |

### Why Custom VAD?

We use a simple RMS (Root Mean Square) energy-based VAD rather than:

| Alternative | Trade-off |
|-------------|-----------|
| **Silero VAD** | Neural network, adds ~50MB, ONNX dependency |
| **WebRTC VAD** | C library, overkill for our needs |

Our RMS-based VAD:
- Zero external dependencies
- ~50 lines of code
- Works well for dictation (quiet room, clear speech)
- Easy to tune threshold

### Why hound?

[hound](https://github.com/ruuda/hound) is a simple WAV file library for debugging. We save captured audio to WAV files for inspection when troubleshooting.

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     yammer-audio                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ AudioCapture │────▶│ VadProcessor │────▶│ Resampler   │ │
│  │              │     │              │     │             │ │
│  │ • Device enum│     │ • RMS calc   │     │ • 48k→16k   │ │
│  │ • Stream mgmt│     │ • State machine    │ • FFT-based │ │
│  │ • Format conv│     │ • Buffering  │     │             │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                    │                    │        │
│         ▼                    ▼                    ▼        │
│    AudioChunk           VadEvent           Vec<f32>        │
│    (raw samples)     (speech/silence)    (16kHz mono)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Capture**: `AudioCapture` gets raw samples from cpal in device's native format
2. **Normalize**: Samples converted to mono f32 in range [-1.0, 1.0]
3. **Detect**: `VadProcessor` identifies speech vs silence
4. **Buffer**: Speech segments accumulated until silence detected
5. **Resample**: Convert from device rate (e.g., 48kHz) to 16kHz for Whisper

### Key Types

```rust
// Raw audio from capture
pub struct AudioChunk {
    pub samples: Vec<f32>,    // Mono, normalized -1.0 to 1.0
    pub sample_rate: u32,      // Device's native rate
}

// VAD state machine states
pub enum VadState {
    Silence,       // No speech detected
    MaybeSpeech,   // Possible speech starting
    Speech,        // Confirmed speech
    MaybeSilence,  // Possible speech ending
}

// Events from VAD processing
pub enum VadEvent {
    SpeechStart,                    // User started talking
    SpeechEnd { samples: Vec<f32> }, // Speech segment complete
    Speaking,                        // Currently speaking
    Silent,                          // Currently silent
}
```

---

## Code Walkthrough

### AudioCapture (capture.rs)

**Device Initialization**:

```rust
pub fn new() -> AudioResult<Self> {
    let host = cpal::default_host();
    let device = host.default_input_device()
        .ok_or(AudioError::NoInputDevice)?;

    let supported_config = device.default_input_config()?;
    let config: StreamConfig = supported_config.into();

    Ok(Self { host, device, config, is_recording: ... })
}
```

The `Host` represents the audio backend (ALSA/PulseAudio on Linux). We get the default input device and its preferred configuration.

**Stream Callback Pattern**:

cpal uses callbacks for low-latency audio. When audio is available, cpal calls your function:

```rust
let stream = device.build_input_stream(
    &config,
    move |data: &[f32], _: &cpal::InputCallbackInfo| {
        // Called ~every 10-20ms with new samples
        // Convert stereo to mono, send to channel
        let mono = convert_to_mono(data, channels);
        tx.try_send(mono).ok();
    },
    |err| error!("Stream error: {}", err),
    None,
)?;
```

**Format Handling**:

Different devices use different sample formats. We handle:
- `f32`: Native float, no conversion needed
- `i16`: Common format, scale by `i16::MAX`
- `u16`: Offset by 32768, then scale

### VAD (vad.rs)

**RMS Calculation**:

RMS measures signal energy, works well for speech detection:

```rust
pub fn calculate_rms(samples: &[f32]) -> f32 {
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    (sum_squares / samples.len() as f32).sqrt()
}
```

- Silent audio: RMS ≈ 0.0
- Quiet speech: RMS ≈ 0.01-0.05
- Normal speech: RMS ≈ 0.05-0.2
- Loud speech: RMS ≈ 0.2-0.5

**Hysteresis State Machine**:

Simple threshold crossing causes "flapping" (rapid on/off). Hysteresis requires multiple consecutive frames:

```
┌─────────┐  N consecutive    ┌─────────────┐  M more    ┌────────┐
│ Silence │─────speech─────▶ │ MaybeSpeech │──speech──▶ │ Speech │
└─────────┘                   └─────────────┘            └────────┘
     ▲                              │                        │
     │                         any silence                   │
     └──────────────────────────────┴────────────────────────┘
                                                    P consecutive
                                                       silence
```

Default values:
- Speech threshold: 0.01 RMS
- Speech start: 3 consecutive frames (~90ms)
- Speech end: 15 consecutive frames (~450ms)

### Resampler (resample.rs)

**Why FFT Resampling?**

Naive resampling (drop/duplicate samples) causes aliasing artifacts. FFT-based resampling:
1. Transform to frequency domain
2. Adjust frequency components
3. Transform back to time domain

This preserves audio quality.

**Usage Pattern**:

```rust
pub fn resample_to_whisper(samples: &[f32], input_rate: u32) -> Result<Vec<f32>, String> {
    if input_rate == WHISPER_SAMPLE_RATE {
        return Ok(samples.to_vec());  // Already 16kHz, skip
    }

    let mut resampler = AudioResampler::for_whisper(input_rate)?;
    resampler.resample_buffer(samples)
}
```

The resampler processes in chunks (1024 samples) for memory efficiency.

---

## Common Tasks

### List Available Audio Devices

```rust
use yammer_audio::AudioCapture;

fn list_devices() {
    match AudioCapture::list_devices() {
        Ok(devices) => {
            for device in devices {
                println!("{} {}",
                    if device.is_default { "*" } else { " " },
                    device.name
                );
                for config in device.configs {
                    println!("  {} channels, {}-{} Hz, {}",
                        config.channels,
                        config.min_sample_rate,
                        config.max_sample_rate,
                        config.sample_format
                    );
                }
            }
        }
        Err(e) => eprintln!("Error: {}", e),
    }
}
```

### Record Audio for a Duration

```rust
use yammer_audio::AudioCapture;
use std::time::Duration;

async fn record_five_seconds() -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let capture = AudioCapture::new()?;
    let chunk = capture.record_duration(Duration::from_secs(5)).await?;

    println!("Captured {} samples at {} Hz",
        chunk.samples.len(),
        chunk.sample_rate
    );

    Ok(chunk.samples)
}
```

### Detect Speech Boundaries

```rust
use yammer_audio::{VadProcessor, VadEvent};

fn process_audio_with_vad(audio_chunks: impl Iterator<Item = Vec<f32>>) {
    let mut vad = VadProcessor::new();

    for chunk in audio_chunks {
        let events = vad.process(&chunk);

        for event in events {
            match event {
                VadEvent::SpeechStart => {
                    println!("Speech started!");
                }
                VadEvent::SpeechEnd { samples } => {
                    println!("Speech ended: {} samples", samples.len());
                    // Send samples to transcription
                }
                VadEvent::Speaking => {
                    // Optional: show recording indicator
                }
                VadEvent::Silent => {
                    // Optional: hide recording indicator
                }
            }
        }
    }
}
```

### Save Audio to WAV for Debugging

```rust
use yammer_audio::write_wav;
use std::path::Path;

fn save_debug_audio(samples: &[f32], sample_rate: u32, filename: &str) {
    let path = Path::new(filename);
    match write_wav(path, samples, sample_rate) {
        Ok(()) => println!("Saved to {}", filename),
        Err(e) => eprintln!("Failed to save: {}", e),
    }
}
```

### Continuous Capture with VAD

```rust
use yammer_audio::{AudioCapture, VadProcessor, VadEvent, resample_to_whisper};

async fn capture_and_detect() -> Result<(), Box<dyn std::error::Error>> {
    let capture = AudioCapture::new()?;
    let sample_rate = capture.sample_rate();

    // Start capture with 30ms chunks
    let (handle, mut rx) = capture.start_capture(30)?;

    let mut vad = VadProcessor::new();

    while let Some(chunk) = rx.recv().await {
        for event in vad.process(&chunk) {
            if let VadEvent::SpeechEnd { samples } = event {
                // Resample to 16kHz for Whisper
                let resampled = resample_to_whisper(&samples, sample_rate)?;
                println!("Got {} samples ready for transcription", resampled.len());
                // Send resampled to yammer-stt
            }
        }
    }

    drop(handle);  // Stop capture
    Ok(())
}
```

### Tune VAD Sensitivity

```rust
use yammer_audio::{Vad, VadConfig, VadProcessor};

// For noisy environments, raise threshold
let config = VadConfig {
    speech_threshold: 0.03,      // Higher = less sensitive
    speech_start_frames: 4,      // More frames = slower to trigger
    speech_end_frames: 20,       // More frames = longer trailing silence
    min_speech_frames: 5,
};

let vad = Vad::with_config(config);
let processor = VadProcessor::with_vad(vad);

// For quiet environments, lower threshold
let quiet_config = VadConfig {
    speech_threshold: 0.005,     // Lower = more sensitive
    speech_start_frames: 2,      // Fewer frames = faster trigger
    speech_end_frames: 10,       // Fewer frames = quicker cutoff
    min_speech_frames: 3,
};
```

---

## Summary

The yammer-audio crate handles the audio pipeline from microphone to transcription-ready samples:

1. **AudioCapture**: Get audio from the microphone via cpal
2. **VadProcessor**: Detect speech using RMS energy with hysteresis
3. **AudioResampler**: Convert to 16kHz mono for Whisper

Key files:
- `src/capture.rs`: Device enumeration, stream management, WAV export
- `src/vad.rs`: Voice activity detection state machine
- `src/resample.rs`: FFT-based resampling via rubato

For questions or issues, check the existing tests in each module or open an issue in the project tracker.
