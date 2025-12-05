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
        }
    }

    /// Create a text output handler with specified method
    pub fn with_method(method: OutputMethod) -> Self {
        Self { method }
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
        info!("Typing {} characters via xdotool", text.len());

        let status = Command::new("xdotool")
            .args(["type", "--clearmodifiers", "--delay", "12", "--", text])
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
    fn test_check_xdotool() {
        // This test will pass if xdotool is installed
        let result = TextOutput::check_xdotool();
        // Don't assert - xdotool may not be installed in CI
        println!("xdotool check result: {:?}", result);
    }

    #[test]
    fn test_empty_text() {
        let output = TextOutput::new();
        assert!(output.output("").is_ok());
    }
}
