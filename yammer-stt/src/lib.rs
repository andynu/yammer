//! yammer-stt: streaming speech-to-text
//!
//! On macOS (Tahoe+), uses Apple's SpeechAnalyzer via the Neural Engine.
//! On Linux/Windows, uses the Kyutai STT (moshi/candle) model.

pub mod wav;

#[cfg(not(target_os = "macos"))]
pub mod transcriber;

#[cfg(target_os = "macos")]
pub mod apple;

/// Common interface for all STT backends.
pub trait SttBackend: Send {
    /// Feed a single chunk of PCM audio at the backend's native sample rate.
    /// Returns any newly recognised words (may be empty).
    fn step(&mut self, pcm: &[f32]) -> anyhow::Result<Vec<String>>;

    /// Transcribe a complete buffer of PCM audio in one call.
    fn transcribe_buffer(&mut self, pcm: &[f32]) -> anyhow::Result<String>;

    /// Number of samples per chunk expected by [`step`].
    fn chunk_samples(&self) -> usize;

    /// Sample rate expected by this backend (e.g. 24000 for Kyutai, 16000 for Apple).
    fn sample_rate(&self) -> u32;
}

// Platform-specific type alias — callers use `Transcriber` everywhere.
#[cfg(target_os = "macos")]
pub type Transcriber = apple::AppleSttTranscriber;

#[cfg(not(target_os = "macos"))]
pub type Transcriber = transcriber::KyutaiTranscriber;

// Re-export the shared WAV loader.
pub use wav::load_wav_resampled;

// Backwards-compatible re-exports for non-macOS (Kyutai).
#[cfg(not(target_os = "macos"))]
pub use transcriber::{KyutaiTranscriber, KYUTAI_CHUNK_SAMPLES};

/// Load a WAV file resampled to 24 kHz (legacy helper, delegates to [`load_wav_resampled`]).
#[cfg(not(target_os = "macos"))]
pub fn load_wav_24k(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    load_wav_resampled(path, 24_000)
}
