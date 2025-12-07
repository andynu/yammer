# yammer-llm Onboarding Guide

Welcome to yammer-llm, the LLM text correction crate for the Yammer dictation application. This guide covers how we use local language models to improve transcribed speech.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### Why LLM Text Correction?

Speech-to-text engines like Whisper produce raw transcriptions that lack:
- **Punctuation**: "im going to the store do you want anything"
- **Capitalization**: proper nouns, sentence starts
- **Apostrophes**: contractions like "it's" vs "its"
- **Context-aware fixes**: homophone errors ("their" vs "there")

Traditional spell checkers can't handle these issues because they:
- Don't understand sentence context
- Can't insert punctuation
- Don't know what you meant to say

LLMs excel at this because they understand language structure and context.

### What are LLMs?

Large Language Models are neural networks trained on vast amounts of text to predict the next word in a sequence. Key concepts:

| Term | Description |
|------|-------------|
| **Transformer** | The neural network architecture (attention mechanism) |
| **Token** | A word piece (roughly 3-4 characters on average) |
| **Context Window** | Maximum tokens the model can "see" at once |
| **Generation** | Predicting tokens one at a time |
| **Temperature** | Randomness control (0 = deterministic, 1 = creative) |

For text correction, we:
1. Provide a prompt explaining the task
2. Include the text to correct
3. Generate the corrected version

### Understanding Model Sizes

Smaller models are faster but less capable:

| Model | Parameters | RAM | Speed | Best For |
|-------|------------|-----|-------|----------|
| TinyLlama 1.1B | 1.1 billion | ~1GB | Very fast | Simple punctuation |
| Phi-3 Mini 3.8B | 3.8 billion | ~3GB | Fast | Good grammar fixes |
| Gemma-2 2B | 2 billion | ~2GB | Fast | General correction |
| Llama-3.2 3B | 3 billion | ~3GB | Medium | Quality results |

**For dictation**, we recommend small models (1-4B parameters) because:
- Latency matters more than perfect prose
- Corrections are simple (punctuation, capitalization)
- Larger models add latency without much benefit

### Quantization Explained

Neural network weights are normally 32-bit floats. Quantization converts them to smaller representations:

| Quantization | Bits | Size Reduction | Quality |
|--------------|------|----------------|---------|
| f32 | 32-bit | 1x | Perfect |
| f16 | 16-bit | 2x | Excellent |
| q8_0 | 8-bit | 4x | Very good |
| q5_0 | 5-bit | 6x | Good |
| q4_0 | 4-bit | 8x | Acceptable |

Quantized models:
- Use less memory
- Run faster (less data to move)
- Have slightly lower quality

**Recommendation**: Use q4_K_M or q5_K_M quantization for best speed/quality.

### Prompt Engineering Basics

LLMs follow instructions via "prompts". For text correction, we use few-shot prompting:

```
Fix transcription errors. Add punctuation. Fix capitalization.

Input: im going to the store
Output: I'm going to the store.

Input: its raining outside
Output: It's raining outside.

Input: {user's text here}
Output:
```

The examples teach the model:
- What kind of fixes we want
- Expected output format
- What NOT to do (don't rephrase)

---

## Library Choices

### Why llama-cpp-rs?

We use [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) (Rust bindings for llama.cpp):

| Alternative | Why Not |
|-------------|---------|
| **Cloud APIs** (OpenAI, Claude) | Latency, privacy, internet required, cost |
| **candle** | Excellent but less mature, fewer model formats |
| **transformers (HuggingFace)** | Python dependency, slower CPU inference |
| **Ollama** | External process, more overhead |

llama-cpp-rs advantages:
- **Pure CPU inference** - no GPU required
- **GGUF format** - efficient, well-supported
- **Quantization** - small models that run fast
- **Low latency** - optimized C++ backend
- **Memory mapping** - fast model loading
- **Rust-native** - clean integration

### llama.cpp Backend

[llama.cpp](https://github.com/ggerganov/llama.cpp) is the underlying engine by Georgi Gerganov. It's become the standard for local LLM inference:

- SIMD optimizations (AVX2, ARM NEON)
- Supports hundreds of model architectures
- Active community, frequent updates
- Battle-tested at scale

### Model Selection

Good models for text correction (2024-2025):

| Model | Why It Works |
|-------|--------------|
| **TinyLlama 1.1B** | Tiny, fast, adequate for simple fixes |
| **Phi-3 Mini** | Microsoft's efficient 3.8B, good reasoning |
| **Gemma-2 2B** | Google's small model, well-balanced |
| **Qwen2.5 0.5B** | Extremely small but capable |

Download GGUF models from:
- https://huggingface.co/models?library=gguf

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                       yammer-llm                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌────────────────────────────────┐│
│  │    Corrector    │     │        Prompt Builder          ││
│  │                 │     │                                ││
│  │ • Load model    │     │ • Few-shot template            ││
│  │ • Create session│────▶│ • Custom prompt support        ││
│  │ • Run generation│     │ • {text} placeholder           ││
│  └─────────────────┘     └────────────────────────────────┘│
│         │                              │                    │
│         ▼                              ▼                    │
│  ┌─────────────────┐           ┌─────────────────┐         │
│  │ CorrectorConfig │           │ CorrectionResult│         │
│  │                 │           │                 │         │
│  │ • max_tokens    │           │ • text          │         │
│  │ • temperature   │           │ • latency_ms    │         │
│  │ • context_size  │           │                 │         │
│  └─────────────────┘           └─────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Receive Text**: Raw transcription from `yammer-stt`
2. **Build Prompt**: Combine template with user text
3. **Create Session**: llama.cpp inference session
4. **Feed Context**: Pass prompt to model
5. **Generate**: Sample tokens until newline
6. **Extract**: Return corrected text with timing

### Key Types

```rust
/// Configuration for generation
pub struct CorrectorConfig {
    pub max_tokens: usize,   // Maximum output length (default: 256)
    pub temperature: f32,     // Sampling randomness (default: 0.1)
    pub context_size: u32,    // Context window (default: 2048)
}

/// Result with performance data
pub struct CorrectionResult {
    pub text: String,        // The corrected text
    pub latency_ms: u64,     // How long correction took
}

/// The corrector engine
pub struct Corrector {
    model: LlamaModel,       // Loaded GGUF model
    config: CorrectorConfig,
}
```

### The Default Prompt

Our built-in prompt uses few-shot examples:

```
Fix transcription errors. Add punctuation (periods, commas,
apostrophes). Fix capitalization. Don't rephrase.

Input: im going to the store do you want anything
Output: I'm going to the store. Do you want anything?

Input: its raining outside we should bring an umbrella
Output: It's raining outside. We should bring an umbrella.

Input: {user text here}
Output:
```

Key design decisions:
- **"Don't rephrase"**: Prevents the model from changing meaning
- **Few-shot examples**: Shows expected behavior
- **Simple output format**: Easy to parse (stop at newline)

---

## Code Walkthrough

### Model Loading (corrector.rs:91-107)

```rust
pub fn with_config(model_path: &Path, config: CorrectorConfig) -> CorrectorResult<Self> {
    info!("Loading LLM model from {:?}", model_path);

    let params = LlamaParams::default();

    let model = LlamaModel::load_from_file(model_path, params)
        .map_err(|e| CorrectorError::ModelLoad(e.to_string()))?;

    info!("LLM model loaded successfully");
    Ok(Self { model, config })
}
```

Model loading:
1. Takes path to `.gguf` model file
2. Creates default llama.cpp parameters
3. Memory-maps the model for fast access
4. Returns `Corrector` instance

### Prompt Construction (corrector.rs:82-88)

```rust
fn build_custom_prompt(template: &str, text: &str) -> String {
    if template.contains(TEXT_PLACEHOLDER) {
        template.replace(TEXT_PLACEHOLDER, text)
    } else {
        format!("{}{}", template, text)
    }
}
```

Two modes:
- **Default prompt**: Few-shot template with "Input:/Output:" format
- **Custom prompt**: User template with `{text}` placeholder

### Generation Process (corrector.rs:115-174)

```rust
pub fn correct_with_prompt(
    &self,
    text: &str,
    custom_prompt: Option<&str>,
) -> CorrectorResult<CorrectionResult> {
    let start = Instant::now();

    // 1. Build prompt
    let prompt = match custom_prompt {
        Some(template) => build_custom_prompt(template, text),
        None => format!("{}{}{}", CORRECTION_PROMPT, text, CORRECTION_SUFFIX),
    };

    // 2. Create session with context size
    let mut session_params = SessionParams::default();
    session_params.n_ctx = self.config.context_size;
    let mut session = self.model.create_session(session_params)?;

    // 3. Feed the prompt
    session.advance_context(&prompt)?;

    // 4. Configure sampler
    let sampler = StandardSampler::new_softmax(
        vec![
            SamplerStage::Temperature(self.config.temperature),
            SamplerStage::TopP(0.95),
            SamplerStage::MinP(0.05),
        ],
        1 // min_keep
    );

    // 5. Generate tokens
    let completions = session
        .start_completing_with(sampler, self.config.max_tokens)?;

    // 6. Collect until newline
    let mut result = String::new();
    for token in completions.into_strings() {
        if token.contains('\n') {
            let before_newline = token.split('\n').next().unwrap_or("");
            result.push_str(before_newline);
            break;
        }
        result.push_str(&token);
    }

    let latency_ms = start.elapsed().as_millis() as u64;

    Ok(CorrectionResult {
        text: result.trim().to_string(),
        latency_ms,
    })
}
```

Sampling parameters explained:
- **Temperature 0.1**: Very deterministic (we want consistent corrections)
- **TopP 0.95**: Consider top 95% probability mass
- **MinP 0.05**: Ignore tokens below 5% probability

Generation stops at:
- Newline character (expected end of output)
- Max tokens reached

---

## Common Tasks

### Load a Model

```rust
use yammer_llm::Corrector;
use std::path::Path;

fn load_corrector() -> Result<Corrector, Box<dyn std::error::Error>> {
    let model_path = Path::new("/path/to/tinyllama-1.1b.Q4_K_M.gguf");
    let corrector = Corrector::new(model_path)?;
    Ok(corrector)
}
```

### Correct Transcribed Text

```rust
use yammer_llm::Corrector;

fn correct_text(
    corrector: &Corrector,
    raw_text: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let result = corrector.correct(raw_text)?;

    println!("Input:  {}", raw_text);
    println!("Output: {}", result.text);
    println!("Time:   {}ms", result.latency_ms);

    Ok(result.text)
}

// Example:
// Input:  im going to the store do you want anything
// Output: I'm going to the store. Do you want anything?
// Time:   127ms
```

### Use Custom Configuration

```rust
use yammer_llm::{Corrector, CorrectorConfig};
use std::path::Path;

fn load_with_config() -> Result<Corrector, Box<dyn std::error::Error>> {
    let config = CorrectorConfig {
        max_tokens: 512,      // Allow longer outputs
        temperature: 0.0,     // Fully deterministic
        context_size: 4096,   // Larger context window
    };

    let model_path = Path::new("/path/to/model.gguf");
    let corrector = Corrector::with_config(model_path, config)?;
    Ok(corrector)
}
```

### Use Custom Prompt Template

```rust
use yammer_llm::Corrector;

fn correct_with_custom_prompt(
    corrector: &Corrector,
    text: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    // Custom prompt with {text} placeholder
    let custom_prompt = r#"You are a text editor. Fix grammar and punctuation.

Text: {text}
Fixed:"#;

    let result = corrector.correct_with_prompt(text, Some(custom_prompt))?;
    Ok(result.text)
}
```

### Batch Correction

```rust
use yammer_llm::Corrector;

fn correct_batch(
    corrector: &Corrector,
    texts: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();
    let mut total_latency = 0u64;

    for text in texts {
        let result = corrector.correct(text)?;
        total_latency += result.latency_ms;
        results.push(result.text);
    }

    println!("Corrected {} texts in {}ms (avg {}ms)",
        texts.len(),
        total_latency,
        total_latency / texts.len() as u64
    );

    Ok(results)
}
```

### Debug Correction Quality

```rust
use yammer_llm::{Corrector, CorrectorConfig};
use std::path::Path;

fn debug_correction(
    model_path: &Path,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let corrector = Corrector::new(model_path)?;

    // Test with different temperatures
    for temp in [0.0, 0.1, 0.3, 0.5] {
        let config = CorrectorConfig {
            temperature: temp,
            ..Default::default()
        };

        // Recreate corrector with new config
        let corrector = Corrector::with_config(model_path, config)?;
        let result = corrector.correct(text)?;

        println!("temp={:.1}: {} ({}ms)",
            temp,
            result.text,
            result.latency_ms
        );
    }

    Ok(())
}
```

### Handle Long Text

For text longer than typical dictation segments:

```rust
use yammer_llm::Corrector;

const MAX_CHUNK_CHARS: usize = 500;  // Keep chunks manageable

fn correct_long_text(
    corrector: &Corrector,
    text: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    if text.len() <= MAX_CHUNK_CHARS {
        return Ok(corrector.correct(text)?.text);
    }

    // Split on sentence boundaries when possible
    let mut result = String::new();
    let mut current_chunk = String::new();

    for sentence in text.split(". ") {
        if current_chunk.len() + sentence.len() > MAX_CHUNK_CHARS {
            // Correct current chunk
            if !current_chunk.is_empty() {
                let corrected = corrector.correct(&current_chunk)?;
                result.push_str(&corrected.text);
                result.push(' ');
            }
            current_chunk = sentence.to_string();
        } else {
            if !current_chunk.is_empty() {
                current_chunk.push_str(". ");
            }
            current_chunk.push_str(sentence);
        }
    }

    // Handle last chunk
    if !current_chunk.is_empty() {
        let corrected = corrector.correct(&current_chunk)?;
        result.push_str(&corrected.text);
    }

    Ok(result.trim().to_string())
}
```

### Verify Model Exists

```rust
use std::path::Path;

fn verify_model(model_path: &Path) -> bool {
    if !model_path.exists() {
        eprintln!("Model not found: {:?}", model_path);
        eprintln!("Download GGUF models from:");
        eprintln!("  https://huggingface.co/models?library=gguf");
        return false;
    }

    let size = std::fs::metadata(model_path)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("Model: {:?} ({:.1} MB)", model_path, size as f64 / 1_000_000.0);

    if size < 100_000_000 {
        println!("Warning: Model is very small, correction quality may be limited");
    }

    true
}
```

---

## Tuning Guidance

### When Corrections Are Wrong

1. **Model too small**: Try a larger model (1B → 3B)
2. **Temperature too high**: Lower to 0.0-0.1 for consistency
3. **Bad prompt**: Add more few-shot examples
4. **Context too small**: Increase context_size

### When Corrections Are Too Slow

1. **Model too large**: Use smaller model (3B → 1B)
2. **Quantization**: Use q4_K_M instead of f16
3. **Context too large**: Reduce context_size
4. **Max tokens**: Reduce if outputs are short

### Recommended Starting Point

```rust
let config = CorrectorConfig {
    max_tokens: 256,       // Usually enough for dictation
    temperature: 0.1,      // Slight randomness prevents loops
    context_size: 2048,    // Enough for prompt + input + output
};
```

Model: TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf (~700MB)

---

## Summary

yammer-llm provides LLM-based text correction:

1. **Corrector**: Load GGUF models, run inference
2. **Prompt Template**: Few-shot examples for consistent output
3. **CorrectionResult**: Corrected text with latency

Key files:
- `src/corrector.rs`: Model loading, prompt building, generation

Requirements:
- GGUF model file (download from Hugging Face)
- Sufficient RAM for model size

Performance tips:
- Use quantized models (Q4_K_M or Q5_K_M)
- Keep temperature low (0.0-0.1)
- Pre-load model at startup (loading takes seconds)
- Use small models (1-3B) for dictation
