# Kyutai STT Integration — Replace Whisper with Streaming On-Device ASR

## Executive Summary

Replace the Whisper batch-transcription backend with Kyutai's streaming STT engine
(`moshi` crate + `candle-core` CUDA) so that transcription happens concurrently with
audio capture — emitting words to the UI while the user is still speaking, rather than
after they stop.

**Key decisions:**
- Whisper (`whisper-rs`) is removed entirely from this branch — no dual-backend toggle
- Target model: `kyutai/stt-1b-en_fr-candle` (1B params, ~2GB, English + French)
- Model weights are auto-downloaded via `hf-hub` on first use, cached in
  `~/.cache/huggingface/`
- GPU inference via `candle-core` CUDA features (no Python, no WebSocket server)
- Sample rate changes from 16 kHz (Whisper) to 24 kHz (Kyutai/Mimi codec)

---

## Technical Approach

### Architecture change

Current pipeline (sequential):
```
listen_blocking()       → collect ALL audio into Vec<f32>
transcribe_blocking()   → send entire buffer to Whisper → partial segment callbacks
correct_blocking()      → LLM pass
output_blocking()       → type/paste
```

New pipeline (concurrent listen + transcribe):
```
listen_and_transcribe_blocking()  → capture 80 ms chunks → step_pcm() in same thread
                                     → Word events emitted while user is still talking
correct_blocking()                → unchanged
output_blocking()                 → unchanged
```

The `moshi::asr::State` state machine processes audio in 1920-sample chunks (80 ms @
24 kHz) via `step_pcm()`, returning `AsrMsg::Word` tokens as Kyutai recognises them.
The pipeline loop runs capture and inference sequentially in one `spawn_blocking` thread:
receive chunk → resample → step_pcm → check flags → repeat.

### Crate additions / removals

| Crate | Before | After |
|---|---|---|
| `whisper-rs` | ✓ (CUDA via whisper.cpp) | removed |
| `moshi` | — | `0.6.1` |
| `candle-core` | — | `0.9.1` + `cuda` feature |
| `candle-nn` | — | `0.9.1` + `cuda` feature |
| `hf-hub` | — | `0.4.3` |
| `sentencepiece` | — | `0.11.3` |

---

## Implementation Steps

### 1. Dependencies

#### 1.1 Workspace `Cargo.toml`
- Remove `whisper-rs` from `[workspace.dependencies]`
- Add to `[workspace.dependencies]`:
  ```toml
  moshi         = { version = "0.6.1" }
  candle-core   = { version = "0.9.1", features = ["cuda"] }
  candle-nn     = { version = "0.9.1", features = ["cuda"] }
  hf-hub        = { version = "0.4.3", features = ["tokio"] }
  sentencepiece = { version = "0.11.3" }
  ```
- Verify `cuda` feature propagates correctly (candle uses `cudarc` internally)

#### 1.2 `yammer-stt/Cargo.toml`
- Remove `whisper-rs`
- Add `moshi`, `candle-core`, `candle-nn`, `hf-hub`, `sentencepiece`
- Keep `anyhow` / `tracing` as they are

#### 1.3 `yammer-app/src-tauri/Cargo.toml`
- Remove `whisper-rs` if referenced directly (it probably isn't — goes via yammer-stt)

---

### 2. Audio resampling (`yammer-audio`)

#### 2.1 `yammer-audio/src/resample.rs`
- Add constant: `pub const KYUTAI_SAMPLE_RATE: u32 = 24_000;`
- Add function `pub fn resample_to_kyutai(samples: &[f32], input_rate: u32) -> Result<Vec<f32>, String>`
  - Reuse existing rubato-based resampling infrastructure (same pattern as `resample_to_whisper`)
  - Target rate: 24000 Hz

#### 2.2 `yammer-audio/src/lib.rs`
- Re-export `KYUTAI_SAMPLE_RATE` and `resample_to_kyutai` from the crate root
- Keep `WHISPER_SAMPLE_RATE` and `resample_to_whisper` only if they're used elsewhere; otherwise remove them

---

### 3. STT crate rewrite (`yammer-stt`)

This is the core new work. Replace the Whisper `Transcriber` with a Kyutai transcriber
that exposes a chunk-based streaming interface.

#### 3.1 `yammer-stt/src/transcriber.rs` — new `KyutaiTranscriber`

```rust
pub struct KyutaiTranscriber {
    model: moshi::asr::State,       // holds LM + Mimi codec
    tokenizer: sentencepiece::SentencePieceProcessor,
    device: candle_core::Device,
}
```

**`KyutaiTranscriber::new(device: candle_core::Device) -> Result<Self>`**
- Use `hf_hub::api::sync::Api::new()` to get an `ApiRepo` for
  `"kyutai/stt-1b-en_fr-candle"`
- Download (or use cached):
  - `config.json`
  - `tokenizer.model`
  - `model.safetensors` (may be sharded: `model-00001-of-N.safetensors` etc.)
  - `mimi_weights.safetensors` (audio codec, from `kyutai/mimi` repo)
- Load safetensors weights into `candle_core::Tensor`s
- Construct `moshi::asr::State` from config + weights + device
- Run warmup: call `step_pcm` with ~10 zero-filled chunks to prime GPU kernels
- Return `Self`

**`KyutaiTranscriber::step(&mut self, pcm_24k: &[f32]) -> Result<Vec<String>>`**
- `pcm_24k` must be exactly 1920 samples (caller's responsibility to buffer)
- Convert slice to `candle_core::Tensor` with shape `(1, 1, 1920)` on `self.device`
- Call `self.model.step_pcm(tensor, None, &().into(), |_, _, _| ())`
- For each `AsrMsg::Word { tokens, .. }` in returned `Vec<AsrMsg>`:
  - Decode token IDs with `self.tokenizer`
  - Collect decoded strings
- Return the new words (empty vec if no word completed this chunk)

#### 3.2 `yammer-stt/src/lib.rs`
- Export `KyutaiTranscriber` as the public type
- Remove the old `Transcriber`, `WhisperContext`, `Transcript` types
- Keep `pub use` of anything pipeline.rs uses by name — update those names

---

### 4. Config changes (`yammer-core`)

#### 4.1 `yammer-core/src/config.rs` — `ModelsConfig`
- Rename field: `whisper: String` → `stt: String`
- Update default value: `"base.en"` → `"kyutai/stt-1b-en_fr-candle"`
- Add serde alias `#[serde(alias = "whisper")]` on `stt` so existing config files keep
  working without a breaking change
- Replace `whisper_model_path(&self) -> PathBuf` with
  `stt_model_repo(&self) -> String` — simply returns `self.models.stt.clone()`
  (the HF repo ID; hf-hub manages the local cache path internally)
- Remove the `ggml-*.bin` filename resolution match block

---

### 5. Pipeline restructure (`yammer-app/src-tauri/src/pipeline.rs`)

This is the biggest structural change.

#### 5.1 `PipelineConfig` — update fields
```rust
// Before:
pub whisper_model_path: PathBuf,

// After:
pub stt_model_repo: String,   // e.g. "kyutai/stt-1b-en_fr-candle"
```

#### 5.2 `DictationPipeline` — swap transcriber type
```rust
// Before:
transcriber: Option<Arc<Transcriber>>,   // Whisper

// After:
transcriber: Option<Arc<KyutaiTranscriber>>,
```

#### 5.3 `initialize()` — update model loading
- Replace Whisper model loading block with Kyutai:
  ```rust
  let device = candle_core::Device::new_cuda(0)
      .unwrap_or(candle_core::Device::Cpu);
  let transcriber = KyutaiTranscriber::new(device)?;
  self.transcriber = Some(Arc::new(transcriber));
  ```
- Remove `whisper_model_path.exists()` check (hf-hub handles download)
- Log the model repo being used

#### 5.4 Merge `listen_blocking` + `transcribe_blocking` → `listen_and_transcribe_blocking`

Replace both methods with a single method. Key details:

**Sample buffer management:**
Audio capture chunks arrive at device sample rate (e.g. 48000 Hz) in ~50 ms bursts.
After resampling to 24 kHz, each burst ≈ 1200 samples. Kyutai needs exactly 1920.
Maintain a `Vec<f32>` carry buffer between iterations:

```
loop {
    chunk = rx.blocking_recv()             // raw device-rate samples
    chunk_24k = resample_to_kyutai(&chunk, input_sr)
    carry_buf.extend_from_slice(&chunk_24k)

    while carry_buf.len() >= 1920 {
        let slice = carry_buf.drain(..1920).collect::<Vec<_>>();
        let new_words = transcriber.step(&slice)?;
        accumulated_text.extend(new_words);
        if !accumulated_text.is_empty() {
            send_transcript(accumulated_text.join(" "), true);
        }
    }

    // check cancel/discard, VAD silence timeout — same logic as before
}

// On cancel: pad carry_buf to 1920 with zeros and do one final step_pcm
// to flush any partial word
```

**State changes:**
- `PipelineState::Listening` remains active throughout (user is speaking AND model is
  running simultaneously)
- Remove the `PipelineState::Processing` transition that happened between listen and
  transcribe — it no longer makes sense as a distinct phase
- After the loop ends and the final word is flushed, transition directly to
  `Correcting` (or `Done` if no LLM)

**Silence timeout:**
- Keep the existing `ever_had_speech` + `silence_timeout` logic unchanged
- The VAD `rms` calculation runs on the raw chunk before resampling (same as now)

**Discard handling:**
- Same as now: check `is_discarded()` each iteration, return `Err("Discarded")` if set

#### 5.5 `run_blocking()` — update call sites
- Replace `listen_blocking()` + `transcribe_blocking()` calls with single
  `listen_and_transcribe_blocking()`
- `correct_blocking()` and `output_blocking()` are unchanged
- Update the `is_cancelled` reset logic (currently resets after listen phase; adjust
  to reset after `listen_and_transcribe_blocking` returns)

#### 5.6 `lib.rs` — update `initialize_pipeline` command
- Change `whisper_model` param references to `stt_model`
- Update `PipelineConfig` construction: `stt_model_repo` instead of `whisper_model_path`
- Remove the `whisper_path.exists()` guard (hf-hub handles it)

---

### 6. Remove dead code

After the above changes, remove:
- `yammer-stt/src/transcriber.rs` old Whisper types (`WhisperContext`, `Transcript`,
  `SegmentCallback`, `transcribe_streaming`, `transcribe`)
- `yammer-audio/src/resample.rs`: `WHISPER_SAMPLE_RATE` and `resample_to_whisper` (if
  unused)
- Any `#[allow(dead_code)]` that was suppressing warnings on the old path
- `PipelineState::Reloading` variant is also currently dead — decide whether to keep
  for the idle-unload reload path or remove

---

### 7. Frontend (minor)

The UI already renders `is_partial: true` transcripts (the `transcript` event with
`{ text, isPartial }` payload). No structural changes needed, but consider:

- **Status label:** Currently shows "Processing..." during the batch transcribe phase.
  With streaming, this phase no longer exists. The `Listening` state now covers the
  whole recording period including live words. The frontend should show partial text
  under the waveform while state is `listening` — verify this already works or add it.
- **Word-by-word animation:** Partial transcript updates will now arrive while state is
  `listening`, not `processing`. Check that the frontend doesn't hide the transcript
  area until `processing` state.

---

## Testing Strategy

### 8.1 Unit tests

- `KyutaiTranscriber::step()` with a zero-filled 1920-sample buffer — should return
  empty vec without panicking (smoke test that GPU inference path works)
- Carry-buffer logic: given N samples of varying sizes, verify exactly the right number
  of `step_pcm` calls are made and no samples are dropped
- Config migration: `stt_model_repo()` returns the repo string; `serde(alias = "whisper")`
  allows old config to deserialize into `stt` field

### 8.2 Integration / manual

1. **Cold start (no cache):** First launch downloads ~2 GB of model weights. Verify
   progress is logged and the app doesn't appear hung.
2. **Warm start:** Second launch uses cache; `initialize()` should complete in under 5s
   (model load + warmup).
3. **Live transcription:** Hold hotkey, speak a sentence — words should appear in the
   UI before releasing the hotkey.
4. **Discard mid-session:** Hold hotkey, speak, press Escape — no text should be output.
5. **Silence timeout:** Hold hotkey, say nothing for 5s — should auto-dismiss.
6. **LLM correction still works:** Enable LLM in config, verify it runs on the final
   Kyutai transcript.
7. **GPU utilisation:** Run `nvidia-smi` while dictating — GPU should show activity
   during the `Listening` phase.

---

## Acceptance Criteria

- [ ] Build succeeds with `cargo build --features cuda` (no Python deps, no whisper.cpp)
- [ ] Model downloads automatically on first run; subsequent runs skip download
- [ ] Words appear in the UI while the user is still speaking (not only after release)
- [ ] Discard, cancel, and silence-timeout still work correctly
- [ ] LLM correction still applies to the Kyutai transcript
- [ ] `nvidia-smi` confirms GPU is used during inference
- [ ] `cargo test` passes across all workspace members

---

## Known Risks and Open Questions

| Risk | Mitigation |
|---|---|
| `sentencepiece` requires system lib (`libsentencepiece-dev`) | Check crate's build.rs — it may bundle; if not, add to build notes |
| `step_pcm` slower than 80 ms on GPU → accumulates lag | Profile with `nvidia-smi dmon`; 1B model on any modern GPU should be well under 80 ms per chunk |
| HF Hub download has no UI progress bar | Log progress via `tracing::info!` at download time; add progress events later if needed |
| `moshi` crate API may differ from `stt-rs` binary's usage | Use `stt-rs/src/main.rs` as the reference implementation throughout |
| Candle CUDA requires compatible CUDA toolkit version | candle 0.9.x targets CUDA 11.2+; user's GPU is confirmed available |
