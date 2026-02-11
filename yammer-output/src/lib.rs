//! Text output for yammer dictation app
//!
//! Provides methods to inject text into the user's focused application.
//! On Linux, uses xdotool/xclip. On Windows, uses enigo/clipboard-win.

use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during text output
#[derive(Error, Debug)]
pub enum OutputError {
    #[cfg(target_os = "linux")]
    #[error("xdotool not found - install with: sudo apt install xdotool")]
    XdotoolNotFound,

    #[error("Failed to execute output command: {0}")]
    ExecutionFailed(#[from] std::io::Error),

    #[error("Output command returned non-zero exit code: {0}")]
    NonZeroExit(i32),

    #[error("Clipboard operation failed: {0}")]
    ClipboardFailed(String),

    #[error("Text input simulation failed: {0}")]
    InputSimulationFailed(String),
}

/// Result type for output operations
pub type OutputResult<T> = Result<T, OutputError>;

/// Output method to use for text injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMethod {
    /// Simulate keystrokes (xdotool type on Linux, enigo on Windows)
    #[default]
    Type,
    /// Use clipboard + paste (more compatible with some apps)
    Clipboard,
}

/// Text output handler for injecting dictated text
pub struct TextOutput {
    method: OutputMethod,
    /// Delay between keystrokes in milliseconds (for Type method)
    typing_delay_ms: u32,
}

impl Default for TextOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl TextOutput {
    /// Create a new text output handler with default settings
    pub fn new() -> Self {
        Self {
            method: OutputMethod::Type,
            typing_delay_ms: 0,
        }
    }

    /// Create a text output handler with specified method
    pub fn with_method(method: OutputMethod) -> Self {
        Self {
            method,
            typing_delay_ms: 0,
        }
    }

    /// Create a text output handler with method and typing delay
    pub fn with_options(method: OutputMethod, typing_delay_ms: u32) -> Self {
        Self {
            method,
            typing_delay_ms,
        }
    }

    /// Check if the text output backend is available
    pub fn check_backend() -> OutputResult<()> {
        #[cfg(target_os = "linux")]
        {
            check_xdotool()
        }
        #[cfg(target_os = "windows")]
        {
            // enigo doesn't need an external tool check
            Ok(())
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            Err(OutputError::InputSimulationFailed(
                "Unsupported platform".into(),
            ))
        }
    }

    /// Output text to the focused application
    pub fn output(&self, text: &str) -> OutputResult<()> {
        if text.is_empty() {
            debug!("Empty text, nothing to output");
            return Ok(());
        }

        // Sanitize control characters that can break terminals
        let sanitized = sanitize_for_terminal(text);
        if sanitized.len() != text.len() {
            warn!(
                "Stripped {} dangerous control characters from output",
                text.len() - sanitized.len()
            );
        }

        if sanitized.is_empty() {
            debug!("Text was entirely control characters, nothing to output");
            return Ok(());
        }

        match self.method {
            OutputMethod::Type => self.type_text(&sanitized),
            OutputMethod::Clipboard => self.paste_text(&sanitized),
        }
    }

    // ── Linux implementation ──────────────────────────────────────────

    #[cfg(target_os = "linux")]
    fn type_text(&self, text: &str) -> OutputResult<()> {
        use std::process::Command;

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

    #[cfg(target_os = "linux")]
    fn paste_text(&self, text: &str) -> OutputResult<()> {
        use std::process::Command;

        info!("Pasting {} characters via clipboard", text.len());

        // Set clipboard content using xclip
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

        // Small delay to ensure clipboard is set
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Simulate Ctrl+V to paste
        let status = Command::new("xdotool")
            .args(["key", "--clearmodifiers", "ctrl+v"])
            .status()?;

        if status.success() {
            debug!("Clipboard paste completed successfully");
            Ok(())
        } else {
            let code = status.code().unwrap_or(-1);
            warn!("xdotool key failed with exit code {}", code);
            Err(OutputError::NonZeroExit(code))
        }
    }

    // ── Windows implementation ────────────────────────────────────────

    #[cfg(target_os = "windows")]
    fn type_text(&self, text: &str) -> OutputResult<()> {
        use enigo::{Enigo, Keyboard, Settings};

        info!(
            "Typing {} characters via enigo (delay={}ms)",
            text.len(),
            self.typing_delay_ms
        );

        let mut enigo = Enigo::new(&Settings::default()).map_err(|e| {
            OutputError::InputSimulationFailed(format!("Failed to create Enigo instance: {}", e))
        })?;

        if self.typing_delay_ms == 0 {
            // Type entire string at once for speed
            enigo.text(text).map_err(|e| {
                OutputError::InputSimulationFailed(format!("Failed to type text: {}", e))
            })?;
        } else {
            // Type character-by-character with delay
            let delay = std::time::Duration::from_millis(self.typing_delay_ms as u64);
            for ch in text.chars() {
                enigo.text(&ch.to_string()).map_err(|e| {
                    OutputError::InputSimulationFailed(format!("Failed to type '{}': {}", ch, e))
                })?;
                std::thread::sleep(delay);
            }
        }

        debug!("enigo type completed successfully");
        Ok(())
    }

    #[cfg(target_os = "windows")]
    fn paste_text(&self, text: &str) -> OutputResult<()> {
        use clipboard_win::{formats, set_clipboard};
        use enigo::{Direction, Enigo, Key, Keyboard, Settings};

        info!("Pasting {} characters via clipboard", text.len());

        // Set clipboard content
        set_clipboard(formats::Unicode, text).map_err(|e| {
            OutputError::ClipboardFailed(format!("Failed to set clipboard: {}", e))
        })?;

        // Small delay to ensure clipboard is set
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Simulate Ctrl+V to paste
        let mut enigo = Enigo::new(&Settings::default()).map_err(|e| {
            OutputError::InputSimulationFailed(format!("Failed to create Enigo instance: {}", e))
        })?;

        enigo.key(Key::Control, Direction::Press).map_err(|e| {
            OutputError::InputSimulationFailed(format!("Failed to press Ctrl: {}", e))
        })?;
        enigo
            .key(Key::Unicode('v'), Direction::Click)
            .map_err(|e| {
                OutputError::InputSimulationFailed(format!("Failed to press V: {}", e))
            })?;
        enigo.key(Key::Control, Direction::Release).map_err(|e| {
            OutputError::InputSimulationFailed(format!("Failed to release Ctrl: {}", e))
        })?;

        debug!("Clipboard paste completed successfully");
        Ok(())
    }
}

// ── Linux-only helpers ────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn check_xdotool() -> OutputResult<()> {
    use std::process::Command;
    match Command::new("which").arg("xdotool").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(OutputError::XdotoolNotFound),
    }
}

/// Sanitize text by removing control characters that can break terminals
///
/// Strips C0 control characters (0x00-0x1F) except:
/// - Tab (0x09) - harmless whitespace
/// - Newline (0x0A) - needed for multi-line text
/// - Carriage return (0x0D) - part of Windows line endings
///
/// Dangerous characters this removes:
/// - Ctrl+C (0x03) - SIGINT, kills processes
/// - Ctrl+D (0x04) - EOF, exits shells/programs
/// - Ctrl+S (0x13) - XOFF, freezes terminal
/// - Ctrl+Z (0x1A) - SIGTSTP, suspends processes
/// - Escape (0x1B) - starts terminal escape sequences
/// - And other control characters
pub fn sanitize_for_terminal(text: &str) -> String {
    text.chars()
        .filter(|&c| {
            // Allow printable characters and safe whitespace
            !c.is_control() || c == '\t' || c == '\n' || c == '\r'
        })
        .collect()
}

/// Convenience function to type text using default settings
pub fn type_text(text: &str) -> OutputResult<()> {
    TextOutput::new().output(text)
}

/// Convenience function to paste text via clipboard
pub fn paste_text(text: &str) -> OutputResult<()> {
    TextOutput::with_method(OutputMethod::Clipboard).output(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_method_default() {
        let method = OutputMethod::default();
        assert_eq!(method, OutputMethod::Type);
    }

    #[test]
    fn test_output_method_debug() {
        assert_eq!(format!("{:?}", OutputMethod::Type), "Type");
        assert_eq!(format!("{:?}", OutputMethod::Clipboard), "Clipboard");
    }

    #[test]
    fn test_output_method_clone() {
        let method = OutputMethod::Clipboard;
        let cloned = method.clone();
        assert_eq!(method, cloned);
    }

    #[test]
    fn test_output_error_display() {
        #[cfg(target_os = "linux")]
        {
            let err = OutputError::XdotoolNotFound;
            assert!(err.to_string().contains("xdotool not found"));
            assert!(err.to_string().contains("sudo apt install xdotool"));
        }

        let err = OutputError::NonZeroExit(1);
        assert!(err.to_string().contains("non-zero"));
        assert!(err.to_string().contains("1"));

        let err = OutputError::ClipboardFailed("test error".into());
        assert!(err.to_string().contains("Clipboard"));
        assert!(err.to_string().contains("test error"));

        let err = OutputError::InputSimulationFailed("sim error".into());
        assert!(err.to_string().contains("simulation failed"));
        assert!(err.to_string().contains("sim error"));
    }

    #[test]
    fn test_text_output_new() {
        let output = TextOutput::new();
        assert_eq!(output.method, OutputMethod::Type);
    }

    #[test]
    fn test_text_output_default() {
        let output = TextOutput::default();
        assert_eq!(output.method, OutputMethod::Type);
    }

    #[test]
    fn test_text_output_with_method() {
        let output = TextOutput::with_method(OutputMethod::Type);
        assert_eq!(output.method, OutputMethod::Type);
        assert_eq!(output.typing_delay_ms, 0);

        let output = TextOutput::with_method(OutputMethod::Clipboard);
        assert_eq!(output.method, OutputMethod::Clipboard);
        assert_eq!(output.typing_delay_ms, 0);
    }

    #[test]
    fn test_text_output_with_options() {
        let output = TextOutput::with_options(OutputMethod::Type, 50);
        assert_eq!(output.method, OutputMethod::Type);
        assert_eq!(output.typing_delay_ms, 50);

        let output = TextOutput::with_options(OutputMethod::Clipboard, 100);
        assert_eq!(output.method, OutputMethod::Clipboard);
        assert_eq!(output.typing_delay_ms, 100);

        // Zero delay should be valid
        let output = TextOutput::with_options(OutputMethod::Type, 0);
        assert_eq!(output.typing_delay_ms, 0);
    }

    #[test]
    fn test_empty_text() {
        let output = TextOutput::new();
        assert!(output.output("").is_ok());

        let output = TextOutput::with_method(OutputMethod::Clipboard);
        assert!(output.output("").is_ok());
    }

    #[test]
    fn test_sanitize_preserves_normal_text() {
        assert_eq!(sanitize_for_terminal("Hello, world!"), "Hello, world!");
        assert_eq!(sanitize_for_terminal("Testing 123"), "Testing 123");
        assert_eq!(sanitize_for_terminal("Special chars: @#$%^&*()"), "Special chars: @#$%^&*()");
    }

    #[test]
    fn test_sanitize_preserves_safe_whitespace() {
        // Tab, newline, carriage return should be preserved
        assert_eq!(sanitize_for_terminal("line1\nline2"), "line1\nline2");
        assert_eq!(sanitize_for_terminal("col1\tcol2"), "col1\tcol2");
        assert_eq!(sanitize_for_terminal("windows\r\nline"), "windows\r\nline");
    }

    #[test]
    fn test_sanitize_removes_dangerous_control_chars() {
        // Ctrl+C (0x03) - SIGINT
        assert_eq!(sanitize_for_terminal("before\x03after"), "beforeafter");
        // Ctrl+D (0x04) - EOF
        assert_eq!(sanitize_for_terminal("before\x04after"), "beforeafter");
        // Ctrl+S (0x13) - XOFF (freezes terminal)
        assert_eq!(sanitize_for_terminal("before\x13after"), "beforeafter");
        // Ctrl+Z (0x1A) - SIGTSTP
        assert_eq!(sanitize_for_terminal("before\x1Aafter"), "beforeafter");
        // Escape (0x1B) - starts escape sequences
        assert_eq!(sanitize_for_terminal("before\x1Bafter"), "beforeafter");
        // Ctrl+\ (0x1C) - SIGQUIT
        assert_eq!(sanitize_for_terminal("before\x1Cafter"), "beforeafter");
    }

    #[test]
    fn test_sanitize_removes_null_and_other_c0() {
        // Null byte
        assert_eq!(sanitize_for_terminal("before\x00after"), "beforeafter");
        // Bell
        assert_eq!(sanitize_for_terminal("before\x07after"), "beforeafter");
        // Backspace
        assert_eq!(sanitize_for_terminal("before\x08after"), "beforeafter");
    }

    #[test]
    fn test_sanitize_handles_all_control_chars() {
        // Text with multiple dangerous chars
        let nasty = "Hello\x03\x04\x13\x1B[31mWorld\x1A";
        let clean = sanitize_for_terminal(nasty);
        assert_eq!(clean, "Hello[31mWorld");
    }

    #[test]
    fn test_sanitize_unicode_preserved() {
        assert_eq!(sanitize_for_terminal("Hello 世界 🌍"), "Hello 世界 🌍");
        assert_eq!(sanitize_for_terminal("Ümläuts"), "Ümläuts");
    }

    #[test]
    fn test_check_backend() {
        // This test will pass if the backend is available
        let result = TextOutput::check_backend();
        // Don't assert - xdotool may not be installed in CI
        println!("backend check result: {:?}", result);
    }

    // Integration tests that require platform tools installed
    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when tools are available
    fn test_type_text_integration() {
        if TextOutput::check_backend().is_ok() {
            let output = TextOutput::new();
            let result = output.output("test");
            println!("type_text result: {:?}", result);
        }
    }

    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when tools are available
    fn test_paste_text_integration() {
        if TextOutput::check_backend().is_ok() {
            let output = TextOutput::with_method(OutputMethod::Clipboard);
            let result = output.output("test");
            println!("paste_text result: {:?}", result);
        }
    }
}
