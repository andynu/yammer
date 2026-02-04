# ADR-001: Voxtral Realtime Evaluation

**Status:** Rejected (revisit when ecosystem matures)
**Date:** 2026-02-04
**Context:** Evaluating alternative STT backends for Yammer

## Summary

Evaluated Voxtral Realtime as a potential replacement for the current whisper.cpp + TinyLlama pipeline. **Decision: Do not adopt.** The 16GB VRAM requirement and vLLM-only runtime don't fit Yammer's embedded Rust architecture or modest hardware goals.

## Context

Yammer's current STT pipeline:

```
Mic → VAD → Audio Buffer → whisper.cpp (batch) → TinyLlama cleanup → xdotool
                              ~500ms-1.5s              ~100-500ms
```

Voxtral Realtime (released January 2026) promised:
- True streaming transcription with sub-200ms latency
- Built-in punctuation/capitalization (potentially eliminating LLM cleanup)
- Open weights under Apache 2.0

## Evaluation

### Voxtral Realtime Specifications

| Property | Value |
|----------|-------|
| Model | Voxtral-Mini-4B-Realtime-2602 |
| Parameters | 4B (~3.4B LLM + 0.6B audio encoder) |
| VRAM Required | 16GB minimum (BF16) |
| License | Apache 2.0 |
| Latency | Configurable 80ms–2.4s (480ms recommended) |
| Languages | 13 (en, zh, hi, es, ar, fr, pt, ru, de, ja, ko, it, nl) |
| WER (English) | 4.90% on FLEURS at 480ms delay |

### Runtime Support

| Framework | Status |
|-----------|--------|
| vLLM | Production-ready (WebSocket `/v1/realtime` endpoint) |
| Transformers | Community contribution needed |
| llama.cpp | Not supported |

### Comparison to Current Stack

| Requirement | Voxtral Realtime | Current (whisper + TinyLlama) |
|-------------|------------------|-------------------------------|
| VRAM | 16GB | ~5GB (whisper large) |
| Embedded Rust | No (Python vLLM sidecar) | Yes (whisper-rs, llama_cpp) |
| Streaming | Native | Batch with callbacks |
| LLM cleanup needed | No | Yes |
| Hardware target | RTX 4070 Ti+ | RTX 3060 / integrated |

## Decision

**Rejected** for the following reasons:

1. **VRAM requirement (16GB)** exceeds available hardware (12GB RTX 3060) and conflicts with "modest hardware" design goal

2. **No embedded runtime** — vLLM is the only supported inference engine, requiring a Python sidecar process with WebSocket communication

3. **No llama.cpp port** — cannot integrate directly into Rust codebase like current whisper-rs approach

4. **Architecture mismatch** — would require rewriting from embedded library calls to client-server WebSocket streaming

## What Would Change This Decision

Revisit if any of the following occur:

1. **llama.cpp port becomes available** — would enable embedded Rust integration via FFI bindings

2. **Quantized versions released** — Q4/Q8 quantization could reduce VRAM to 8GB or less, fitting mid-range GPUs

3. **Smaller streaming variant** — Mistral releases a 1-2B parameter streaming model

4. **Hardware upgrade** — if Yammer's target hardware moves to 16GB+ VRAM cards

## Alternatives Considered

### API-only prototype
Could validate output quality at $0.006/min, but conflicts with Yammer's "100% local" philosophy and doesn't solve the integration problem.

### Voxtral Mini 3B (batch model)
- 9.5GB VRAM (closer to feasible)
- Has Transformers support
- But: batch-only, no streaming advantage over whisper.cpp
- Would still require sidecar process

### Wait and watch
**Selected approach.** Monitor:
- HuggingFace model page for runtime updates
- llama.cpp issues/PRs for Voxtral support
- Mistral announcements for smaller/quantized variants

## References

- [Voxtral Transcribe 2 Announcement](https://mistral.ai/news/voxtral-transcribe-2)
- [Voxtral-Mini-4B-Realtime-2602 Model Card](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Voxtral-Mini-3B-2507 Model Card](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507)

## Appendix: Current Architecture Reference

```
yammer-stt/src/transcriber.rs   — whisper-rs integration (batch transcription)
yammer-llm/src/corrector.rs     — llama_cpp text cleanup (punctuation, capitalization)
yammer-audio/src/vad.rs         — voice activity detection
yammer-audio/src/capture.rs     — microphone capture
```

Key architectural constraint from `Cargo.toml`:
```toml
# GPU Acceleration Configuration
# whisper-rs and llama_cpp both bundle ggml with CUDA support. When both enable CUDA,
# the linker fails with duplicate symbol errors. Current workaround: CUDA only for
# whisper-rs (STT), LLM runs on CPU.
whisper-rs = { version = "0.14", features = ["cuda"] }
llama_cpp = { version = "0.3" }  # CPU-only
```

If Voxtral becomes viable, it would replace both crates with a single streaming model, resolving the CUDA symbol conflict as a side benefit.
