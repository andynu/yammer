# Changelog

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
