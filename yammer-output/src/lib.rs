//! Text output for yammer dictation app
//!
//! Provides methods to inject text into the user's focused application using xdotool.

use std::process::Command;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during text output
#[derive(Error, Debug)]
pub enum OutputError {
    #[error("xdotool not found - install with: sudo apt install xdotool")]
    XdotoolNotFound,

    #[error("Failed to execute xdotool: {0}")]
    ExecutionFailed(#[from] std::io::Error),

    #[error("xdotool returned non-zero exit code: {0}")]
    NonZeroExit(i32),

    #[error("Clipboard operation failed: {0}")]
    ClipboardFailed(String),
}

/// Result type for output operations
pub type OutputResult<T> = Result<T, OutputError>;

/// Output method to use for text injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMethod {
    /// Use xdotool type command (simulates keystrokes)
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

    /// Check if xdotool is available
    pub fn check_xdotool() -> OutputResult<()> {
        match Command::new("which").arg("xdotool").output() {
            Ok(output) if output.status.success() => Ok(()),
            _ => Err(OutputError::XdotoolNotFound),
        }
    }

    /// Output text to the focused application
    pub fn output(&self, text: &str) -> OutputResult<()> {
        if text.is_empty() {
            debug!("Empty text, nothing to output");
            return Ok(());
        }

        match self.method {
            OutputMethod::Type => self.type_text(text),
            OutputMethod::Clipboard => self.paste_text(text),
        }
    }

    /// Type text using xdotool type command
    ///
    /// The `--clearmodifiers` flag releases any held keys (like Super from hotkey)
    /// before typing to prevent modifier interference.
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

    /// Paste text via clipboard using xclip + xdotool key
    ///
    /// This method is more compatible with some applications (electron apps,
    /// some terminals) that don't handle simulated keystrokes well.
    fn paste_text(&self, text: &str) -> OutputResult<()> {
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
        let err = OutputError::XdotoolNotFound;
        assert!(err.to_string().contains("xdotool not found"));
        assert!(err.to_string().contains("sudo apt install xdotool"));

        let err = OutputError::NonZeroExit(1);
        assert!(err.to_string().contains("non-zero"));
        assert!(err.to_string().contains("1"));

        let err = OutputError::ClipboardFailed("test error".into());
        assert!(err.to_string().contains("Clipboard"));
        assert!(err.to_string().contains("test error"));
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
    fn test_check_xdotool() {
        // This test will pass if xdotool is installed
        let result = TextOutput::check_xdotool();
        // Don't assert - xdotool may not be installed in CI
        println!("xdotool check result: {:?}", result);
    }

    // Integration tests that require xdotool installed
    // These are conditionally run based on xdotool availability
    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when xdotool is available
    fn test_type_text_integration() {
        if TextOutput::check_xdotool().is_ok() {
            // Type to a non-focused window won't affect anything
            let output = TextOutput::new();
            let result = output.output("test");
            // May fail if no window focused, but shouldn't panic
            println!("type_text result: {:?}", result);
        }
    }

    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when xclip/xdotool available
    fn test_paste_text_integration() {
        if TextOutput::check_xdotool().is_ok() {
            let output = TextOutput::with_method(OutputMethod::Clipboard);
            let result = output.output("test");
            println!("paste_text result: {:?}", result);
        }
    }
}
