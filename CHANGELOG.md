# Changelog

## [0.5.0] - 2026-10-14

### Added
- **IBus input engine**: Yammer registers as an IBus input method ("Yammer
  Dictation" under Settings > Keyboard > Input Sources) and commits dictated text
  through IBus into the focused app. Normal typing passes through unchanged. The
  clipboard/xdotool path remains the fallback when IBus is unavailable or the
  commit fails. The engine lives in a separate `libyammer_ibus_shim.so`, loaded at
  runtime, so the LGPL `librush` dependency is only ever dynamically linked. See
  `docs/IBUS_SETUP.md` for the one-time setup.
- Stop dictation with "Microphone is muted" after one second of exact-zero
  samples, instead of listening silently and reporting "No audio recorded".
- One info log line per dictation with a stage timing breakdown (keypress to
  listening, speech to transcript, correction, typing, total).
- WAV fixtures and an `#[ignore]`d end-to-end transcription regression test, run
  by hand before releases.

### Changed
- Pasting sets both PRIMARY and CLIPBOARD, so dictated text works with
  Shift+Insert in terminals and Ctrl+V in GUI apps.
- The paste shortcut is sent from a uinput virtual keyboard. On GNOME Wayland,
  xdotool's XTEST events queued behind the RemoteDesktop portal's approval dialog
  and arrived late. xdotool remains the fallback without `/dev/uinput`.
- The dictation window is non-focusable on X11. Escape-to-discard no longer
  works there, since the window never receives keys.
- Before output, Yammer hides its own window if it holds keyboard focus and waits
  up to 500ms for focus to return to the target app. On Wayland the window takes
  focus when shown regardless of the non-focusable setting, so dictated text went
  into Yammer's own window and the overlay never auto-hid.
- Keep the raw transcript when LLM correction comes back empty or rewrites the
  text beyond punctuation, capitalization and word-splitting edits.

### Fixed
- The IBus engine stopped working after the first hotkey press of a session: each
  `--toggle` second instance took over the engine's IBus bus name, then exited.
  The engine now starts only in the primary instance.
- `yammer transcribe` dropped the last ~0.5s of speech and leaked held-back words
  into the next segment; each call now resets the model and flushes trailing
  silence.
- The CPU fallback for speech-to-text failed on every step (missing BF16 kernels);
  the language model now loads as F32 on CPU. CUDA keeps BF16.
- Build failure under GCC 15 in `sentencepiece-sys` (missing `<cstdint>`).

## [0.4.3] - 2026-03-23

### Fixed
- **Fix dropped spaces in output**: `xdotool type --delay 0` silently drops
  characters (especially spaces) on X11. Clipboard mode now uses PRIMARY
  selection + Shift+Insert, which pastes reliably across terminals and GUI apps.
  Also enforce a minimum 2ms delay for the `type` output method as a safeguard.
- **Fix last word cutoff on stop**: When pressing F6 to stop recording, the
  final word was lost because (a) the audio capture buffer held a partial chunk
  that never got sent, and (b) the STT model needs silence after speech to emit
  boundary tokens. Now flushes the capture buffer and feeds ~1s of silence to
  push the model to emit any pending final word.
- **Fix session bleed-through**: Reset the ASR model state between dictation
  sessions so words from a previous session don't leak into the next one.
- **Fix SentencePiece spacing**: Accumulate raw token IDs and decode as a single
  sequence instead of decoding per-word and joining with spaces, letting
  SentencePiece handle all `▁` → space conversion correctly.

## [0.4.1] - 2026-02-25

### Fixed
- Silence timeout no longer fires in toggle mode (hotkey or record button).
  Dictation now waits for an explicit stop instead of auto-discarding after 5
  seconds of silence.

### Changed
- Release builds default to warn-level logging with no log file. Set
  `RUST_LOG=yammer=debug` for full diagnostics and file logging to
  `~/.local/share/yammer/yammer.log`. Debug builds retain full logging.

## [0.4.0] - 2026-02-17

### Changed
- **Replace Whisper with Kyutai streaming STT**: Speech-to-text now uses Kyutai's
  Moshi-based streaming pipeline instead of batch Whisper inference. Audio is
  processed as a continuous stream, reducing latency for real-time transcription.
- Build language model config dynamically from HuggingFace `config.json` instead
  of hardcoded Moshi preset
- Migrate existing Whisper model name settings to Kyutai defaults automatically
- Simplify pipeline by removing unsafe raw pointer usage

### Fixed
- Frontend crash when checking model status (updated JSON shape for Kyutai backend)

### Added
- Resampling unit tests for Kyutai audio input requirements

## [0.3.0] - 2026-02-12

### Added
- Windows support: text output via enigo (keystroke simulation) and clipboard-win
  (clipboard paste), with platform-gated implementations for Linux and Windows
- Configurable hold-to-talk hotkey: set `hotkey.hold_keys` in config.toml to any
  combination of modifier keys (Control, Super, Alt, Shift). Default: Ctrl+Super
- GitHub Actions CI for Windows builds alongside Linux
- Startup grace period (5s) for hotkey listener to prevent false activation from
  spurious key events during desktop startup
- Auto-dismiss listening window after 5 seconds of silence
- Start window hidden on launch to prevent flash of empty window

### Fixed
- Suppress ALSA/JACK stderr noise on Linux
- Set explicit tray icon ID to prevent duplicates
- Always auto-hide window on discard

## [0.2.0] - 2026-02-10

### Added
- Idle model unloading: Whisper and LLM models are automatically freed from memory
  after a configurable idle period (default 2 hours). Models transparently reload
  when the user next triggers dictation, with amber "Reloading models..." indicator.
  Set `gui.idle_unload_seconds` in config.toml (0 to disable).
- Ctrl+Super press-hold-release hotkey for dictation
- AI cleanup toggle to switch between raw and corrected transcripts
- Configurable max recording duration (`audio.max_recording_seconds`)
- Streaming transcription shows partial results during processing

### Fixed
- Sanitize control characters in text output to prevent terminal issues
- Black corner artifacts in UI window transparency
- Waveform display showing only faint blips for quiet speakers

## [0.1.0] - 2025-12-06

### Added
- Local speech-to-text dictation using Whisper (CUDA-accelerated)
- LLM text correction using TinyLlama
- Tauri overlay window with real-time waveform visualization
- Click-to-toggle and press-hold recording modes
- Global hotkey support (Super+H toggle, Ctrl+Super hold-to-record)
- System tray with show/toggle/copy-last/quit actions
- Minimize to tray on close/Escape
- Auto-hide window after successful dictation
- Click-to-copy transcript text
- Audio device selection via config
- Configurable output method (xdotool type or clipboard paste)
- Configurable typing delay for slow applications
- Discard recording with Escape key during dictation
- GNOME shell integration via single-instance --toggle flag
- Window position persistence across launches (multi-monitor aware)
- Audio feedback sounds for recording state changes
- Whisper hallucination and special token filtering
- Model download manager with verified checksums
- Comprehensive unit test suite
