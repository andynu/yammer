# Windows Support Implementation Plan

## Overview

Add Windows support to Yammer. The codebase is already largely cross-platform
thanks to Tauri 2.0, `cpal`, `whisper-rs`, `llama_cpp`, `dirs`, and `rdev`.
The primary work is implementing a Windows text output backend and wiring up
build/packaging configuration.

## Prerequisites

- Windows 10/11 development machine
- Rust toolchain (`rustup` with `stable-x86_64-pc-windows-msvc`)
- Visual Studio Build Tools (C++ workload for linking)
- WebView2 runtime (bundled with Windows 11, installable on 10)
- CUDA Toolkit if building with GPU support for Whisper

## Phase 1: Compile on Windows

**Goal:** `cargo build` succeeds for the workspace.

### 1.1 Make `libc` conditional in `yammer-audio`

File: `yammer-audio/Cargo.toml`

Change `libc` from unconditional to Linux-only:

```toml
[target.'cfg(target_os = "linux")'.dependencies]
libc = "0.2"
```

The `StderrSuppressor` in `capture.rs` is already gated with
`#[cfg(target_os = "linux")]`, so no Rust code changes needed here.

### 1.2 Verify workspace compiles

```powershell
cargo check --workspace
```

Fix any compilation errors. Most likely all other crates compile cleanly since
their dependencies (`cpal`, `whisper-rs`, `llama_cpp`, `dirs`, `rdev`,
`tauri`) are already cross-platform.

## Phase 2: Windows Text Output Backend

**Goal:** `yammer-output` works on Windows.

This is the critical path. The current implementation calls `xdotool` and
`xclip` which are Linux-only.

### 2.1 Add Windows dependencies

File: `yammer-output/Cargo.toml`

```toml
[target.'cfg(target_os = "windows")'.dependencies]
enigo = "0.2"          # Cross-platform keyboard/mouse simulation
clipboard-win = "5"    # Native Windows clipboard access
```

Alternative: use `windows-rs` directly with `SendInput()` for finer control.
`enigo` is simpler but may have edge cases with Unicode input. Evaluate both.

### 2.2 Implement Windows output methods

File: `yammer-output/src/lib.rs`

The current code has two output strategies:
- **Type mode:** Simulates keystrokes character-by-character (`xdotool type`)
- **Clipboard mode:** Copies text to clipboard, simulates Ctrl+V (`xclip` + `xdotool key`)

Implement Windows equivalents behind `#[cfg(target_os = "windows")]`:

```rust
#[cfg(target_os = "windows")]
fn type_text(text: &str, typing_delay_ms: u64) -> OutputResult<()> {
    // Use enigo to type each character with delay
    // OR use SendInput via windows-rs for more control
}

#[cfg(target_os = "windows")]
fn paste_text(text: &str) -> OutputResult<()> {
    // 1. Copy text to clipboard (clipboard-win or windows-rs)
    // 2. Simulate Ctrl+V via enigo or SendInput
}

#[cfg(target_os = "linux")]
fn type_text(text: &str, typing_delay_ms: u64) -> OutputResult<()> {
    // Existing xdotool implementation
}

#[cfg(target_os = "linux")]
fn paste_text(text: &str) -> OutputResult<()> {
    // Existing xclip + xdotool implementation
}
```

### 2.3 Update error messages

The current error for missing `xdotool` says `sudo apt install xdotool`.
Gate this behind `#[cfg(target_os = "linux")]` and provide a Windows-appropriate
message (or remove the check entirely since the Windows path won't shell out).

### 2.4 Test text output

- Type mode: open Notepad, trigger dictation, verify text appears
- Clipboard mode: same test, verify clipboard paste works
- Unicode: test with accented characters, emoji
- Speed: verify typing delay setting works on Windows

## Phase 3: Hotkey Adjustments

**Goal:** Hotkeys work sensibly on Windows.

### 3.1 Review hotkey bindings

File: `yammer-app/src-tauri/src/lib.rs`

Current hotkeys:
- Ctrl+Super (press-hold-release) for dictation trigger via `rdev`
- Super+H registered via `tauri_plugin_global_shortcut`

`rdev` works on Windows. The Super key maps to the Windows key. Verify:
- Ctrl+Win doesn't conflict with Windows system shortcuts
- If conflicts exist, consider making the hotkey configurable or choosing a
  different default on Windows (e.g., Ctrl+Alt or a function key)

### 3.2 Test hotkey registration

- Verify global shortcut registers without conflict
- Verify press-hold-release detection works with `rdev` on Windows
- Test with common Windows keyboard layouts

## Phase 4: Audio Capture Verification

**Goal:** Confirm audio capture works via WASAPI.

### 4.1 Test device enumeration

`cpal::default_host()` will use WASAPI on Windows. Verify:
- Default input device is detected
- Device list populates correctly in the UI
- Sample rate and channel count are reported accurately

### 4.2 Test recording

- Record audio, verify the VAD (voice activity detection) triggers
- Verify audio quality is sufficient for Whisper transcription
- Test with different microphones (USB, built-in, Bluetooth)

### 4.3 No stderr suppression needed

The `StderrSuppressor` exists to quiet ALSA/JACK probe noise on Linux.
WASAPI doesn't produce this noise, so the existing `#[cfg(target_os = "linux")]`
gate is sufficient. No work needed.

## Phase 5: Tauri Packaging for Windows

**Goal:** Produce a Windows installer.

### 5.1 Configure Windows bundler

File: `yammer-app/src-tauri/tauri.conf.json`

Add/verify Windows bundle configuration:

```json
{
  "bundle": {
    "targets": "all",
    "windows": {
      "certificateThumbprint": null,
      "digestAlgorithm": "sha256",
      "timestampUrl": ""
    }
  }
}
```

Tauri supports NSIS and MSI installers. NSIS is recommended for most cases.

### 5.2 Verify icons

Windows icon (`icon.ico`) already exists in the icons directory. Verify it
contains the standard sizes (16x16, 32x32, 48x48, 256x256).

### 5.3 Build installer

```powershell
cd yammer-app
npx tauri build
```

Output: `target/release/bundle/nsis/Yammer_*_x64-setup.exe`

### 5.4 Test installation

- Install via the generated installer
- Verify Start Menu shortcut
- Verify system tray icon appears
- Verify uninstaller works cleanly

## Phase 6: CI/CD

**Goal:** Automated Windows builds.

### 6.1 Add Windows job to GitHub Actions

File: `.github/workflows/build.yml`

Add a `windows` job alongside the existing `linux` job:

```yaml
build-windows:
  runs-on: windows-latest
  steps:
    - uses: actions/checkout@v4
    - uses: dtolnay/rust-toolchain@stable
    - uses: swatinem/rust-cache@v2
    - name: Install frontend dependencies
      run: cd yammer-app && npm install
    - name: Build
      run: cd yammer-app && npx tauri build
    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: yammer-windows
        path: target/release/bundle/nsis/*.exe
```

Note: CUDA support in CI requires a self-hosted runner with a GPU, or build
a CPU-only variant for CI.

## Phase 7: Model Distribution

**Goal:** Users can obtain Whisper and LLM models on Windows.

### 7.1 Verify model download paths

`yammer-core/src/config.rs` uses `dirs::data_local_dir()` which resolves to
`C:\Users\<user>\AppData\Local` on Windows. Verify:
- Model download/placement works at this path
- Path handling uses `std::path::PathBuf` (not string concatenation with `/`)

### 7.2 Document model setup for Windows

Users need to know where to place or how to download models. Update any
setup documentation to include Windows paths.

## Summary of Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `yammer-output/src/lib.rs` | Add Windows text output backend | **Critical** |
| `yammer-output/Cargo.toml` | Add Windows-only dependencies | **Critical** |
| `yammer-audio/Cargo.toml` | Make `libc` conditional on Linux | High |
| `yammer-app/src-tauri/tauri.conf.json` | Windows bundler config | Medium |
| `.github/workflows/build.yml` | Add Windows CI job | Medium |
| `yammer-app/src-tauri/src/lib.rs` | Verify/adjust hotkey defaults | Low |

## Decision Log

Decisions to make during implementation:

1. **Text injection library:** `enigo` vs `windows-rs` + `SendInput()` directly?
   - `enigo`: simpler API, may have Unicode quirks
   - `windows-rs`: more control, more boilerplate
   - Try `enigo` first, fall back to `windows-rs` if needed

2. **Default hotkey on Windows:** Keep Ctrl+Win or change?
   - Test for conflicts with Windows system shortcuts first

3. **CUDA on Windows:** Support GPU acceleration?
   - CPU-only is simpler to start with
   - CUDA on Windows requires CUDA Toolkit + cuDNN

4. **Installer format:** NSIS vs MSI?
   - NSIS is Tauri's default and generally preferred
   - MSI if enterprise/GPO deployment is needed later
