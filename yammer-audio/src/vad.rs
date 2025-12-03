//! Voice Activity Detection

/// Voice activity detector
pub struct Vad {
    // Will be populated when VAD integration is added
}

impl Vad {
    /// Create a new VAD instance
    pub fn new() -> Self {
        Self {}
    }

    /// Check if audio buffer contains speech
    pub fn is_speech(&self, _samples: &[f32]) -> bool {
        // Placeholder - will implement actual VAD
        false
    }
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}
