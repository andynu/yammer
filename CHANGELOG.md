# Changelog

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
