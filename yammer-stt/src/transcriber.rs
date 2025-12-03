//! Whisper transcription

use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Target sample rate for Whisper
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Transcription errors
#[derive(Error, Debug)]
pub enum TranscribeError {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Audio load failed: {0}")]
    AudioLoad(String),

    #[error("Transcription failed: {0}")]
    Transcription(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for transcription operations
pub type TranscribeResult<T> = Result<T, TranscribeError>;

/// A segment of transcribed text with timing
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    /// Start time in milliseconds
    pub start_ms: i64,
    /// End time in milliseconds
    pub end_ms: i64,
    /// Transcribed text
    pub text: String,
}

impl std::fmt::Display for TranscriptSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let start_sec = self.start_ms as f64 / 1000.0;
        let end_sec = self.end_ms as f64 / 1000.0;
        write!(f, "[{:.2}s -> {:.2}s] {}", start_sec, end_sec, self.text)
    }
}

/// Complete transcription result
#[derive(Debug, Clone)]
pub struct Transcript {
    /// Individual segments
    pub segments: Vec<TranscriptSegment>,
}

impl Transcript {
    /// Get the full text without timestamps
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl std::fmt::Display for Transcript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for segment in &self.segments {
            writeln!(f, "{}", segment)?;
        }
        Ok(())
    }
}

/// Whisper transcriber
pub struct Transcriber {
    ctx: WhisperContext,
}

impl Transcriber {
    /// Create a new transcriber with the given model path
    pub fn new(model_path: &Path) -> TranscribeResult<Self> {
        info!("Loading Whisper model from {:?}", model_path);

        let params = WhisperContextParameters::default();
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| {
                TranscribeError::ModelLoad("Invalid model path".to_string())
            })?,
            params,
        )
        .map_err(|e| TranscribeError::ModelLoad(e.to_string()))?;

        info!("Whisper model loaded successfully");
        Ok(Self { ctx })
    }

    /// Transcribe audio samples (must be 16kHz mono f32)
    pub fn transcribe(&self, samples: &[f32]) -> TranscribeResult<Transcript> {
        debug!("Transcribing {} samples ({:.2}s)", samples.len(),
               samples.len() as f32 / WHISPER_SAMPLE_RATE as f32);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure parameters for better results
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Create state for this transcription
        let mut state = self.ctx.create_state()
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        // Run transcription
        state.full(params, samples)
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        // Extract segments
        let num_segments = state.full_n_segments()
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        debug!("Got {} segments", num_segments);

        let mut segments = Vec::new();
        for i in 0..num_segments {
            let text = state.full_get_segment_text(i)
                .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

            let start = state.full_get_segment_t0(i)
                .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

            let end = state.full_get_segment_t1(i)
                .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

            // Convert from centiseconds to milliseconds
            segments.push(TranscriptSegment {
                start_ms: start as i64 * 10,
                end_ms: end as i64 * 10,
                text,
            });
        }

        Ok(Transcript { segments })
    }

    /// Transcribe a WAV file
    pub fn transcribe_file(&self, path: &Path) -> TranscribeResult<Transcript> {
        info!("Transcribing file: {:?}", path);

        let samples = load_wav_16k(path)?;
        self.transcribe(&samples)
    }
}

/// Load a WAV file and convert to 16kHz mono f32
pub fn load_wav_16k(path: &Path) -> TranscribeResult<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| TranscribeError::AudioLoad(e.to_string()))?;

    let spec = reader.spec();
    debug!("WAV: {} channels, {} Hz, {:?}",
           spec.channels, spec.sample_rate, spec.sample_format);

    if spec.sample_rate != WHISPER_SAMPLE_RATE {
        return Err(TranscribeError::AudioLoad(format!(
            "Expected {} Hz audio, got {} Hz. Use --resample when recording.",
            WHISPER_SAMPLE_RATE, spec.sample_rate
        )));
    }

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono_samples = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    debug!("Loaded {} mono samples", mono_samples.len());
    Ok(mono_samples)
}
