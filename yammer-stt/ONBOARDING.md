# yammer-stt Onboarding Guide

Welcome to yammer-stt, the speech-to-text crate for the Yammer dictation application. This guide covers how we convert audio into text using Whisper.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### What is Speech-to-Text?

Speech-to-text (STT), also called automatic speech recognition (ASR), converts spoken audio into written text. Modern systems use neural networks trained on thousands of hours of transcribed speech.

The basic pipeline:

```
Audio Samples → Feature Extraction → Neural Network → Text Output
                (mel spectrogram)    (transformer)    (tokens → words)
```

### The Whisper Model

[Whisper](https://github.com/openai/whisper) is OpenAI's speech recognition model, released in 2022. Key characteristics:

| Feature | Description |
|---------|-------------|
| **Architecture** | Encoder-decoder Transformer |
| **Training Data** | 680,000 hours of multilingual audio from the web |
| **Languages** | 99 languages with varying quality |
| **Capabilities** | Transcription, translation, language detection, timestamps |

Whisper is trained on diverse, noisy web data, making it robust to:
- Background noise
- Accents and dialects
- Technical jargon
- Disfluencies (um, uh, stuttering)

### Model Sizes

Whisper comes in several sizes, trading accuracy for speed and memory:

| Model | Parameters | VRAM/RAM | English WER* | Relative Speed |
|-------|------------|----------|--------------|----------------|
| tiny | 39M | ~1GB | 7.6% | 32x |
| base | 74M | ~1GB | 5.0% | 16x |
| small | 244M | ~2GB | 3.4% | 6x |
| medium | 769M | ~5GB | 2.9% | 2x |
| large | 1550M | ~10GB | 2.7% | 1x |

*WER = Word Error Rate on LibriSpeech test-clean

**For dictation**, we recommend:
- **small** or **base** for most users (good balance)
- **tiny** for low-resource systems
- **medium** if you need higher accuracy and have the RAM

### GGML/GGUF Model Formats

The original Whisper models are PyTorch (`.pt` files). We use **quantized GGML/GGUF** versions:

| Format | Description |
|--------|-------------|
| **GGML** | Original format from llama.cpp ecosystem |
| **GGUF** | Newer format with better metadata support |

Quantization reduces model size and improves CPU inference speed by using lower-precision numbers:

| Quantization | Precision | Size Reduction | Quality Loss |
|--------------|-----------|----------------|--------------|
| f32 | 32-bit float | 1x | None |
| f16 | 16-bit float | 2x | Negligible |
| q8_0 | 8-bit int | 4x | Very small |
| q4_0 | 4-bit int | 8x | Noticeable |

We use **q5_0** or **q8_0** quantization for the best quality/size trade-off.

Download models from: https://huggingface.co/ggerganov/whisper.cpp/tree/main

### Whisper's Input Requirements

Whisper expects:
- **Sample rate**: 16,000 Hz (16 kHz)
- **Channels**: Mono
- **Format**: f32 samples in range [-1.0, 1.0]
- **Minimum duration**: ~1 second (16,000 samples)

The `yammer-audio` crate handles resampling to these requirements.

---

## Library Choices

### Why whisper-rs?

We chose [whisper-rs](https://github.com/tazz4843/whisper-rs) (Rust bindings for whisper.cpp):

| Alternative | Why Not |
|-------------|---------|
| **OpenAI Whisper (Python)** | Requires Python runtime, slower CPU inference |
| **Cloud APIs** (Google, AWS, Azure) | Latency, privacy concerns, internet required |
| **Vosk** | Lower accuracy, less active development |
| **DeepSpeech** | Deprecated by Mozilla |
| **faster-whisper** | Python-based, CTranslate2 backend |

whisper-rs advantages:
- **Pure CPU inference** - no GPU required
- **Quantization support** - smaller models, faster inference
- **Low latency** - optimized C++ backend
- **Privacy** - all processing local
- **Rust-native** - no Python/FFI complexity

### whisper.cpp Backend

[whisper.cpp](https://github.com/ggerganov/whisper.cpp) is the underlying engine, written by Georgi Gerganov (creator of llama.cpp). Benefits:

- SIMD optimizations (AVX2, NEON)
- OpenBLAS/MKL acceleration for matrix ops
- Memory-mapped model loading
- Streaming support (real-time transcription)

### hound for WAV Loading

We use [hound](https://github.com/ruuda/hound) for loading WAV files during testing/debugging. It's a simple, pure-Rust WAV library.

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       yammer-stt                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌────────────────────────────────┐│
│  │   Transcriber   │     │        Hallucination Filter    ││
│  │                 │     │                                ││
│  │ • Load model    │     │ • Known patterns               ││
│  │ • Create state  │────▶│ • Bracketed annotations        ││
│  │ • Run inference │     │ • Short fillers                ││
│  │ • Extract segs  │     │                                ││
│  └─────────────────┘     └────────────────────────────────┘│
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌─────────────────┐           ┌─────────────────┐         │
│  │  Transcript     │           │     String      │         │
│  │                 │───────────│                 │         │
│  │ • Segments[]    │  .text()  │ (filtered text) │         │
│  │ • Timestamps    │           │                 │         │
│  └─────────────────┘           └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio In**: 16kHz mono f32 samples from `yammer-audio`
2. **Pad if Short**: Whisper needs at least 1 second
3. **Create State**: whisper-rs context manages inference state
4. **Run Inference**: Greedy decoding with English language setting
5. **Extract Segments**: Get text and timestamps per segment
6. **Filter Output**: Remove hallucinations before returning

### Key Types

```rust
/// A segment with timing information
pub struct TranscriptSegment {
    pub start_ms: i64,   // Start time in milliseconds
    pub end_ms: i64,     // End time in milliseconds
    pub text: String,    // Transcribed text for this segment
}

/// Complete transcription result
pub struct Transcript {
    pub segments: Vec<TranscriptSegment>,
}

impl Transcript {
    /// Get full text with hallucinations filtered
    pub fn text(&self) -> String { ... }
}

/// The transcription engine
pub struct Transcriber {
    ctx: WhisperContext,  // Holds loaded model
}
```

### Why Hallucination Filtering?

Whisper has a known issue: it generates plausible-sounding text even when there's no speech. Common hallucinations:

- **Silence artifacts**: "[BLANK_AUDIO]", "[inaudible]"
- **YouTube training data leakage**: "Thank you for watching", "Subscribe"
- **Music notation**: "[Music]", "♪"
- **Subtitle artifacts**: "Sous-titres réalisés par..."

Our filter catches these patterns and removes them from the output.

---

## Code Walkthrough

### Model Loading (transcriber.rs:151-166)

```rust
pub fn new(model_path: &Path) -> TranscribeResult<Self> {
    info!("Loading Whisper model from {:?}", model_path);

    let params = WhisperContextParameters::default();
    let ctx = WhisperContext::new_with_params(
        model_path.to_str()?,
        params,
    )?;

    info!("Whisper model loaded successfully");
    Ok(Self { ctx })
}
```

Model loading:
1. Takes path to `.bin` (GGML) or `.gguf` model file
2. Creates default context parameters
3. Memory-maps the model (fast loading, shared memory)
4. Returns `Transcriber` instance

Loading time varies by model size: tiny ~1s, base ~2s, small ~5s.

### Transcription Process (transcriber.rs:169-233)

```rust
pub fn transcribe(&self, samples: &[f32]) -> TranscribeResult<Transcript> {
    // 1. Pad short audio
    let samples = if samples.len() < WHISPER_MIN_SAMPLES {
        let mut padded = samples.to_vec();
        padded.resize(WHISPER_MIN_SAMPLES, 0.0);
        Cow::Owned(padded)
    } else {
        Cow::Borrowed(samples)
    };

    // 2. Configure parameters
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    // 3. Create inference state
    let mut state = self.ctx.create_state()?;

    // 4. Run inference
    state.full(params, &samples)?;

    // 5. Extract segments
    let num_segments = state.full_n_segments()?;
    let mut segments = Vec::new();

    for i in 0..num_segments {
        let text = state.full_get_segment_text(i)?;
        let start = state.full_get_segment_t0(i)?;  // centiseconds
        let end = state.full_get_segment_t1(i)?;

        segments.push(TranscriptSegment {
            start_ms: start as i64 * 10,  // convert to ms
            end_ms: end as i64 * 10,
            text,
        });
    }

    Ok(Transcript { segments })
}
```

Key parameters explained:
- `SamplingStrategy::Greedy { best_of: 1 }`: Fast, deterministic decoding
- `set_language(Some("en"))`: Force English (faster than auto-detect)
- `set_print_*`: Disable whisper.cpp's built-in logging

### Hallucination Filter (transcriber.rs:60-122)

```rust
const HALLUCINATION_PATTERNS: &[&str] = &[
    "[inaudible]", "(inaudible)", "[BLANK_AUDIO]",
    "[MUSIC]", "[Music]", "[Applause]", "[Laughter]",
    "Thank you for watching", "Thanks for watching",
    "Subscribe to my channel", "Please subscribe",
    "Thank you.", "Thanks.", "Bye.", "Bye-bye.",
    "you", "You", "Sous-titres", "...", "♪",
];

fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim();

    // Empty check
    if trimmed.is_empty() { return true; }

    // Exact pattern match (case-insensitive)
    for pattern in HALLUCINATION_PATTERNS {
        if trimmed.eq_ignore_ascii_case(pattern) { return true; }
    }

    // Bracketed annotations
    if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')')) {
        return true;
    }

    // Very short single words (likely fillers)
    let word_count = trimmed.split_whitespace().count();
    if word_count == 1 && trimmed.len() <= 3 { return true; }

    false
}
```

The filter runs when calling `transcript.text()`, not during segment extraction, so you can still access raw segments if needed.

### WAV File Loading (transcriber.rs:245-289)

```rust
pub fn load_wav_16k(path: &Path) -> TranscribeResult<Vec<f32>> {
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    // Verify sample rate
    if spec.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(TranscribeError::AudioLoad(...));
    }

    // Convert samples to f32
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => { /* normalize by bit depth */ }
        hound::SampleFormat::Float => { /* direct collection */ }
    };

    // Convert stereo to mono by averaging
    let mono = if spec.channels > 1 {
        samples.chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels)
            .collect()
    } else {
        samples
    };

    Ok(mono)
}
```

Note: This function requires 16kHz audio. If your WAV is a different rate, use `yammer-audio::resample_to_whisper()` first.

---

## Common Tasks

### Load a Model

```rust
use yammer_stt::Transcriber;
use std::path::Path;

fn load_model() -> Result<Transcriber, Box<dyn std::error::Error>> {
    let model_path = Path::new("/path/to/ggml-base.en.bin");
    let transcriber = Transcriber::new(model_path)?;
    Ok(transcriber)
}
```

Model files are typically stored in:
- `~/.local/share/yammer/models/` (user)
- Or specified via config

### Transcribe Audio Samples

```rust
use yammer_stt::Transcriber;
use yammer_audio::resample_to_whisper;

fn transcribe_audio(
    transcriber: &Transcriber,
    samples: &[f32],
    sample_rate: u32,
) -> Result<String, Box<dyn std::error::Error>> {
    // Ensure 16kHz
    let samples_16k = if sample_rate != 16000 {
        resample_to_whisper(samples, sample_rate)?
    } else {
        samples.to_vec()
    };

    // Transcribe
    let transcript = transcriber.transcribe(&samples_16k)?;

    // Get filtered text
    Ok(transcript.text())
}
```

### Transcribe a WAV File

```rust
use yammer_stt::{Transcriber, load_wav_16k};
use std::path::Path;

fn transcribe_file(
    transcriber: &Transcriber,
    wav_path: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    // Load already-16kHz WAV
    let samples = load_wav_16k(wav_path)?;

    // Transcribe
    let transcript = transcriber.transcribe(&samples)?;

    // Show segments with timing
    for segment in &transcript.segments {
        println!("{}", segment);  // "[0.00s -> 1.50s] Hello world"
    }

    Ok(transcript.text())
}
```

### Get Individual Segments with Timing

```rust
use yammer_stt::Transcriber;

fn get_segments(
    transcriber: &Transcriber,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let transcript = transcriber.transcribe(samples)?;

    for segment in &transcript.segments {
        println!(
            "{:.2}s - {:.2}s: {}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text.trim()
        );
    }

    Ok(())
}
```

### Handle Long Audio (Chunking)

Whisper works best on 30-second chunks. For longer audio:

```rust
use yammer_stt::Transcriber;

const CHUNK_SAMPLES: usize = 16000 * 30;  // 30 seconds at 16kHz

fn transcribe_long_audio(
    transcriber: &Transcriber,
    samples: &[f32],
) -> Result<String, Box<dyn std::error::Error>> {
    let mut full_text = String::new();

    for chunk in samples.chunks(CHUNK_SAMPLES) {
        let transcript = transcriber.transcribe(chunk)?;
        full_text.push_str(&transcript.text());
        full_text.push(' ');
    }

    Ok(full_text.trim().to_string())
}
```

### Debug Transcription Issues

When transcription produces unexpected results:

```rust
use yammer_stt::Transcriber;
use yammer_audio::write_wav;
use std::path::Path;

fn debug_transcription(
    transcriber: &Transcriber,
    samples: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    // Save audio for manual inspection
    write_wav(Path::new("/tmp/debug_audio.wav"), samples, 16000)?;
    println!("Audio saved to /tmp/debug_audio.wav");

    // Transcribe and show raw segments (before filtering)
    let transcript = transcriber.transcribe(samples)?;

    println!("Raw segments:");
    for (i, seg) in transcript.segments.iter().enumerate() {
        println!("  {}: [{:.2}s-{:.2}s] {:?}",
            i,
            seg.start_ms as f64 / 1000.0,
            seg.end_ms as f64 / 1000.0,
            seg.text
        );
    }

    println!("\nFiltered text: {}", transcript.text());

    Ok(())
}
```

Play the saved WAV to verify audio quality. Common issues:
- Too quiet → boost gain before transcription
- Clipping → reduce input level
- Wrong sample rate → verify 16kHz
- Too short → Whisper hallucinates on very short clips

### Check Available Model

```rust
use std::path::Path;

fn verify_model(model_path: &Path) -> bool {
    if !model_path.exists() {
        eprintln!("Model not found: {:?}", model_path);
        eprintln!("Download from: https://huggingface.co/ggerganov/whisper.cpp");
        return false;
    }

    let size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("Model: {:?} ({:.1} MB)", model_path, size as f64 / 1_000_000.0);
    true
}
```

---

## Summary

yammer-stt provides Whisper-based speech recognition:

1. **Transcriber**: Load GGML/GGUF models, run inference
2. **Transcript**: Segments with text and timing
3. **Hallucination Filter**: Remove Whisper artifacts

Key files:
- `src/transcriber.rs`: Model loading, inference, segment extraction

Requirements:
- 16kHz mono f32 audio (use `yammer-audio` for conversion)
- GGML/GGUF model file (download from Hugging Face)

Performance tips:
- Use quantized models (q5_0 or q8_0)
- Process in ~30-second chunks
- Pre-load model at startup (loading takes seconds)
