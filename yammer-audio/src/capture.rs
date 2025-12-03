//! Audio capture functionality

/// Audio capture handle
pub struct AudioCapture {
    // Will be populated when cpal integration is added
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for AudioCapture {
    fn default() -> Self {
        Self::new()
    }
}
