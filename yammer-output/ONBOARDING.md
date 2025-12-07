# yammer-output Onboarding Guide

Welcome to yammer-output, the text output crate for the Yammer dictation application. This guide covers how we inject dictated text into your focused application.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### The Problem: Getting Text into Applications

After speech recognition produces text, we need to inject it into whatever application the user is working in (text editor, browser, terminal, etc.). This is surprisingly complex because:

- Applications don't share a universal input mechanism
- Window focus must be correct
- Modifier keys (Shift, Ctrl, Alt) can interfere
- Different apps handle input differently

### How Desktop Input Works

On Linux, input flows through the display server:

```
                    ┌────────────────────┐
Keyboard/Mouse ────▶│   Display Server   │────▶ Application
                    │  (X11 or Wayland)  │
                    └────────────────────┘
                             ▲
                             │
                    ┌────────────────────┐
                    │  Input Simulation  │
                    │    (xdotool)       │
                    └────────────────────┘
```

### X11 vs Wayland

Linux has two display server protocols:

| Protocol | Description | Input Simulation |
|----------|-------------|------------------|
| **X11** | Legacy protocol (1987), universal support | Full support via XTEST extension |
| **Wayland** | Modern protocol (2012), better security | Heavily restricted by design |

**Current state (2024-2025)**:
- Most distros default to Wayland (GNOME, KDE)
- X11 still available via XWayland
- Input simulation tools (xdotool) only work on X11

**Yammer's approach**: Target X11/XWayland. Users on pure Wayland need to run their apps under XWayland or use clipboard mode.

### Two Approaches to Text Output

#### 1. Keystroke Simulation ("Type" method)

Simulate individual key presses as if typed:

```
Text "Hello" → KeyPress(H) KeyPress(e) KeyPress(l) KeyPress(l) KeyPress(o)
```

**Pros**:
- Works like real typing
- Triggers autocomplete, keybindings
- No clipboard pollution

**Cons**:
- Can be slow for long text
- Modifier interference (held Super/Alt keys)
- Some apps don't accept synthetic keystrokes

#### 2. Clipboard + Paste ("Clipboard" method)

Copy text to clipboard, simulate Ctrl+V:

```
Text "Hello" → SetClipboard("Hello") → KeyPress(Ctrl+V)
```

**Pros**:
- Fast for any length
- Works in more applications
- Simple implementation

**Cons**:
- Overwrites user's clipboard
- Ctrl+V not universal (vim, some terminals)
- Some apps don't support rich paste

### Modifier Key Interference

When you press a hotkey to trigger dictation (e.g., Super+D), the modifier key is still held when we try to type. This causes problems:

```
User holds Super → Dictation runs → We type "a" → App receives Super+a (wrong!)
```

Solution: `xdotool --clearmodifiers` releases held modifiers before typing.

---

## Library Choices

### Why xdotool?

We use [xdotool](https://github.com/jordansissel/xdotool), a mature command-line tool for X11 automation:

| Alternative | Why Not |
|-------------|---------|
| **enigo** (Rust) | Less reliable, X11 support issues, unmaintained periods |
| **rdev** (Rust) | Read-only on Linux (capture but not simulate) |
| **ydotool** | Wayland-focused, requires root for uinput |
| **wtype** | Wayland-only, limited adoption |
| **Native X11 bindings** | Complex FFI, maintenance burden |

xdotool advantages:
- Battle-tested (10+ years)
- Handles edge cases (modifiers, focus, delays)
- Simple subprocess interface
- Packaged in all major distros
- Documentation and community support

### Why Shell Out?

We spawn xdotool as a subprocess rather than linking to a library because:

1. **Isolation**: xdotool crashes don't affect our process
2. **Simplicity**: No FFI complexity
3. **Updates**: Get xdotool fixes via system package manager
4. **Debugging**: Can test xdotool commands manually

### Why xclip?

For clipboard operations, we use [xclip](https://github.com/astrand/xclip):

| Alternative | Why Not |
|-------------|---------|
| **xsel** | Similar, xclip more common |
| **wl-copy** | Wayland-only |
| **arboard** (Rust) | Cross-platform but adds dependency |

xclip is simple, reliable, and universally available.

### Required System Dependencies

```bash
# Debian/Ubuntu
sudo apt install xdotool xclip

# Fedora
sudo dnf install xdotool xclip

# Arch
sudo pacman -S xdotool xclip
```

---

## Architecture Overview

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      yammer-output                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌────────────────────────────────┐│
│  │   TextOutput    │     │         OutputMethod           ││
│  │                 │     │                                ││
│  │ • method        │────▶│ • Type: xdotool type           ││
│  │ • typing_delay  │     │ • Clipboard: xclip + Ctrl+V    ││
│  │ • output()      │     │                                ││
│  └─────────────────┘     └────────────────────────────────┘│
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                     Subprocess                          ││
│  │                                                         ││
│  │  xdotool type --clearmodifiers --delay 0 -- "text"     ││
│  │  xclip -selection clipboard < "text"                    ││
│  │  xdotool key --clearmodifiers ctrl+v                    ││
│  │                                                         ││
│  └─────────────────────────────────────────────────────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Receive Text**: Corrected text from `yammer-llm`
2. **Check Method**: Type or Clipboard mode
3. **Execute Command**: Spawn xdotool/xclip subprocess
4. **Handle Result**: Check exit code, report errors

### Key Types

```rust
/// Which method to use for output
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMethod {
    #[default]
    Type,       // Simulate keystrokes
    Clipboard,  // Copy + Ctrl+V
}

/// The output handler
pub struct TextOutput {
    method: OutputMethod,
    typing_delay_ms: u32,  // Delay between keystrokes
}

/// Errors that can occur
pub enum OutputError {
    XdotoolNotFound,
    ExecutionFailed(std::io::Error),
    NonZeroExit(i32),
    ClipboardFailed(String),
}
```

---

## Code Walkthrough

### Type Method (lib.rs:101-121)

```rust
fn type_text(&self, text: &str) -> OutputResult<()> {
    info!(
        "Typing {} characters via xdotool (delay={}ms)",
        text.len(),
        self.typing_delay_ms
    );

    let delay_str = self.typing_delay_ms.to_string();
    let status = Command::new("xdotool")
        .args(["type", "--clearmodifiers", "--delay", &delay_str, "--", text])
        .status()?;

    if status.success() {
        debug!("xdotool type completed successfully");
        Ok(())
    } else {
        let code = status.code().unwrap_or(-1);
        warn!("xdotool type failed with exit code {}", code);
        Err(OutputError::NonZeroExit(code))
    }
}
```

xdotool arguments explained:
- `type`: Simulate typing (vs `key` for single keypresses)
- `--clearmodifiers`: Release held Shift/Ctrl/Alt/Super before typing
- `--delay N`: Milliseconds between keystrokes (0 = as fast as possible)
- `--`: Separator (allows text starting with `-`)
- `text`: The actual text to type

### Clipboard Method (lib.rs:127-164)

```rust
fn paste_text(&self, text: &str) -> OutputResult<()> {
    info!("Pasting {} characters via clipboard", text.len());

    // Step 1: Set clipboard content using xclip
    let mut child = Command::new("xclip")
        .args(["-selection", "clipboard"])
        .stdin(std::process::Stdio::piped())
        .spawn()?;

    if let Some(stdin) = child.stdin.as_mut() {
        use std::io::Write;
        stdin.write_all(text.as_bytes())?;
    }

    let status = child.wait()?;
    if !status.success() {
        return Err(OutputError::ClipboardFailed(
            "xclip failed to set clipboard".into(),
        ));
    }

    // Step 2: Small delay to ensure clipboard is ready
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Step 3: Simulate Ctrl+V to paste
    let status = Command::new("xdotool")
        .args(["key", "--clearmodifiers", "ctrl+v"])
        .status()?;

    if status.success() {
        debug!("Clipboard paste completed successfully");
        Ok(())
    } else {
        let code = status.code().unwrap_or(-1);
        Err(OutputError::NonZeroExit(code))
    }
}
```

Key implementation details:
- `xclip -selection clipboard`: Use system clipboard (not primary selection)
- Pipe text via stdin (handles all characters safely)
- 50ms delay ensures clipboard is set before paste
- `xdotool key ctrl+v`: Simulate single keystroke combo

### Dependency Check (lib.rs:77-82)

```rust
pub fn check_xdotool() -> OutputResult<()> {
    match Command::new("which").arg("xdotool").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(OutputError::XdotoolNotFound),
    }
}
```

Called at startup to fail fast if xdotool isn't installed.

---

## Common Tasks

### Basic Text Output

```rust
use yammer_output::{TextOutput, OutputMethod};

fn output_text(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output = TextOutput::new();
    output.output(text)?;
    Ok(())
}
```

### Use Clipboard Mode

```rust
use yammer_output::{TextOutput, OutputMethod};

fn paste_text(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output = TextOutput::with_method(OutputMethod::Clipboard);
    output.output(text)?;
    Ok(())
}
```

### Configure Typing Delay

Some applications need a delay between keystrokes to process them:

```rust
use yammer_output::{TextOutput, OutputMethod};

fn slow_type(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    // 50ms delay between each keystroke
    let output = TextOutput::with_options(OutputMethod::Type, 50);
    output.output(text)?;
    Ok(())
}
```

### Check xdotool Installation

```rust
use yammer_output::TextOutput;

fn check_dependencies() -> Result<(), Box<dyn std::error::Error>> {
    TextOutput::check_xdotool()?;
    println!("xdotool is installed");
    Ok(())
}
```

### Convenience Functions

```rust
use yammer_output::{type_text, paste_text};

// Quick typing
type_text("Hello, world!")?;

// Quick pasting
paste_text("Hello, world!")?;
```

### Handle Errors

```rust
use yammer_output::{TextOutput, OutputError};

fn output_with_fallback(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    let output = TextOutput::new();

    match output.output(text) {
        Ok(()) => Ok(()),
        Err(OutputError::XdotoolNotFound) => {
            eprintln!("xdotool not found. Install with: sudo apt install xdotool");
            Err("Missing dependency".into())
        }
        Err(OutputError::NonZeroExit(code)) => {
            eprintln!("xdotool failed with code {}", code);
            // Maybe try clipboard method as fallback
            let clipboard = TextOutput::with_method(yammer_output::OutputMethod::Clipboard);
            clipboard.output(text)?;
            Ok(())
        }
        Err(e) => Err(e.into()),
    }
}
```

### Debug Output Issues

```bash
# Test xdotool directly
xdotool type "test"

# Test with clearmodifiers
xdotool type --clearmodifiers "test"

# Test clipboard
echo "test" | xclip -selection clipboard
xdotool key ctrl+v

# Check if running under Wayland
echo $XDG_SESSION_TYPE  # "x11" or "wayland"

# Check focused window
xdotool getactivewindow getwindowname
```

---

## Troubleshooting

### "xdotool not found"

**Cause**: xdotool not installed.

**Fix**:
```bash
# Debian/Ubuntu
sudo apt install xdotool

# Fedora
sudo dnf install xdotool

# Arch
sudo pacman -S xdotool
```

### Text Not Appearing

**Possible causes**:

1. **Wrong window focused**
   ```bash
   # Check focused window
   xdotool getactivewindow getwindowname
   ```

2. **Running under Wayland**
   ```bash
   echo $XDG_SESSION_TYPE
   # If "wayland", either:
   # - Run your app with XWayland
   # - Use clipboard method
   # - Switch to X11 session
   ```

3. **Modifier key stuck**
   ```bash
   # Release all modifiers
   xdotool keyup Super_L Super_R Control_L Control_R Alt_L Alt_R Shift_L Shift_R
   ```

4. **App doesn't accept synthetic keystrokes**
   - Try clipboard method instead
   - Some electron apps need `--enable-features=UseOzonePlatform`

### Special Characters Not Working

**Cause**: Keyboard layout mismatch or Unicode handling.

**Workarounds**:
```bash
# Use xdotool's unicode mode (if available)
xdotool type --clearmodifiers -- "café"

# Or use clipboard method
echo "café" | xclip -selection clipboard
xdotool key ctrl+v
```

### Clipboard Method Doesn't Paste

**Possible causes**:

1. **xclip not installed**
   ```bash
   sudo apt install xclip
   ```

2. **App uses different paste key**
   - Vim: `"+p` or `"*p`
   - Some terminals: Ctrl+Shift+V or Shift+Insert

3. **App blocks Ctrl+V** (some terminals)
   - Use typing method instead
   - Or configure terminal to accept Ctrl+V

### Output Is Garbled

**Cause**: Typing too fast for the application.

**Fix**: Add delay between keystrokes:
```rust
let output = TextOutput::with_options(OutputMethod::Type, 10);
output.output(text)?;
```

### Wayland Compatibility

On pure Wayland (no XWayland):

1. **ydotool** is an alternative but requires root access
2. **wtype** works but only for wlroots-based compositors (Sway)
3. **Best option**: Run target apps with `GDK_BACKEND=x11` or `DISPLAY=:0`

```bash
# Force app to use X11
GDK_BACKEND=x11 your-app

# Or use XWayland
DISPLAY=:0 your-app
```

---

## Summary

yammer-output provides text injection for dictation:

1. **TextOutput**: Main interface for output operations
2. **OutputMethod::Type**: Simulate keystrokes via xdotool
3. **OutputMethod::Clipboard**: Copy + Ctrl+V via xclip

Key files:
- `src/lib.rs`: All output functionality

Requirements:
- xdotool (always required)
- xclip (for clipboard method)
- X11 session (or XWayland on Wayland)

Common issues:
- xdotool not installed → `sudo apt install xdotool`
- Wayland → switch to X11 or use XWayland
- Modifier interference → `--clearmodifiers` handles this
- App-specific issues → try clipboard method
