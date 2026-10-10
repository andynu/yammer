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

## Revisit: 2026-10-10

Two of the four triggers have fired, though not in the form the list above expected.

1. **llama.cpp port.** Still not there. The feature request
   ([#19696](https://github.com/ggml-org/llama.cpp/issues/19696)) and the planning issue
   that replaced it ([#20914](https://github.com/ggml-org/llama.cpp/issues/20914)) were
   both closed, the latter as stale on 2026-07-17. The three Voxtral Realtime PRs
   (#19698, #20625, #20638) were closed unmerged. llama.cpp gained file transcription
   through `/v1/audio/transcriptions` (#21863), but libmtmd still has no realtime audio
   input.
2. **An embeddable ggml runtime exists anyway.**
   [audio.cpp](https://github.com/0xShug0/audio.cpp) (Apache-2.0, C++ on ggml, v0.9.1
   released 2026-10-07) supports Voxtral Realtime in offline and streaming modes and
   ships a C ABI (`include/audiocpp.h`, `docs/c_api.md`) with a pull-based streaming
   session: `audiocpp_stream_start`, `audiocpp_stream_push` per audio chunk,
   `audiocpp_stream_next_event`, `audiocpp_stream_finish`. That covers the "embedded Rust
   via FFI" requirement without a sidecar. Prebuilt CUDA 12.8 and 13.3 library archives
   are published for Linux. `libaudiocpp.so` uses a version script that exports only
   `audiocpp_*`, so its bundled ggml should not collide with the ggml inside `llama_cpp`
   (the clash that kept the LLM on CPU in the whisper era). That is the library's stated
   intent and has not been tried here.
3. **Quantized variants.** Mistral's `mistral-experimental` org publishes audio.cpp GGUFs
   ([AudioCPP-Voxtral-Mini-4B-Realtime-2602-GGUF](https://huggingface.co/mistral-experimental/AudioCPP-Voxtral-Mini-4B-Realtime-2602-GGUF)):
   Q8_0 at 5.1 GB and Q4_K at 3.1 GB on disk, so the 16 GB BF16 figure no longer applies.
   Runtime VRAM is unmeasured. The 12 GB target card and iris's 16 GB RTX A4000
   both have room.
4. **Smaller streaming variant.** None. Mistral's only addition since February is
   `Voxtral-Mini-4B-Realtime-Arabic` (2026-10-08), the same 4B size.

audio.cpp's own throughput note matters for a live source: a streaming step advances
80 ms of audio and must finish in under 80 ms, and a session that falls behind stays
behind (the decoder runs through silence too). On an M3 Air with Q8_0 it measured
78 to 88 ms per step. CUDA numbers on our hardware are unmeasured.

The comparison has also moved. This ADR was written against batch whisper.cpp. Yammer
now streams with Kyutai `stt-1b-en_fr` through candle, so Voxtral's case is no longer
"streaming at all" but more languages (13 against 2) and possibly better accuracy, at
four times the parameter count. Adopting it is a separate evaluation: per-step latency
on CUDA, WER on the WAV fixtures in the end-to-end test, and VRAM next to the LLM.

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
