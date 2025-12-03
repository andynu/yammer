//! Whisper transcription

/// Whisper transcriber
pub struct Transcriber {
    // Will be populated when whisper-rs integration is added
}

impl Transcriber {
    /// Create a new transcriber (placeholder)
    pub fn new() -> Self {
        Self {}
    }

    /// Transcribe audio samples to text
    pub fn transcribe(&self, _samples: &[f32]) -> String {
        // Placeholder - will implement actual transcription
        String::new()
    }
}

impl Default for Transcriber {
    fn default() -> Self {
        Self::new()
    }
}
