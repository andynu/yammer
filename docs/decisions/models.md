# Models Considered

Every speech or language model evaluated for Yammer, and what was decided. Add a row
whenever one is looked at, including the ones rejected after a quick read. Longer
evaluations get their own ADR in this directory and a link here.

## Speech-to-text

| Model | Considered | Decision | Why | Details |
|-------|-----------|----------|-----|---------|
| whisper.cpp via `whisper-rs` | 2025-12-03 | Adopted, then replaced 2026-02-17 | Batch only: no words until the user stops talking. Its bundled ggml clashed with llama_cpp's when both used CUDA, so the LLM was kept on CPU. | `implementation_research.md` |
| Voxtral Mini 4B Realtime (Mistral) | 2026-02-04 | Rejected, revisit if the ecosystem matures | 16 GB VRAM minimum, and vLLM is the only runtime, which means a Python sidecar instead of an embedded Rust library. | [ADR-001](001-voxtral-realtime-evaluation.md) |
| Kyutai `stt-1b-en_fr-candle` | 2026-02-17 | **In use** | Streams words as they are spoken, runs in-process through candle. English and French. Needs CUDA: CPU inference is too slow for live audio. | [plan 001](../../plans/001-kyutai-stt-integration.md) |
| Whistle (Cactus Compute) | 2026-10-08 | Rejected | 16.9 MB, CPU only, Apache 2.0, 7 languages, C API. Batch only with a 30 s cap per pass, so the live transcript would be lost and longer dictations need chunking. Its published accuracy is about Whisper base, well short of a 1B model. Only worth reopening if a GPU-free build becomes a goal. | [blog](https://cactuscompute.com/blog/whistle), [`Cactus-Compute/whistle`](https://huggingface.co/Cactus-Compute/whistle) |

## Text correction

| Model | Considered | Decision | Why | Details |
|-------|-----------|----------|-----|---------|
| TinyLlama 1.1B Chat (Q4_K_M GGUF) via `llama_cpp` | 2025-12-04 | **In use** | Replaced earlier models that were incompatible with the llama_cpp version. Small enough to run on CPU next to the STT model. | commit `b66043c` |
