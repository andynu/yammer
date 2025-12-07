//! Whisper transcription

use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};
use whisper_rs::{FullParams, SamplingStrategy, SegmentCallbackData, WhisperContext, WhisperContextParameters};

/// Target sample rate for Whisper
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Minimum audio duration in samples (Whisper requires at least 1 second)
pub const WHISPER_MIN_SAMPLES: usize = 16000;

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

/// Known Whisper hallucination patterns to filter out
const HALLUCINATION_PATTERNS: &[&str] = &[
    // Common hallucinations during silence
    "[inaudible]",
    "(inaudible)",
    "[BLANK_AUDIO]",
    "[MUSIC]",
    "[Music]",
    "[Applause]",
    "[Laughter]",
    // YouTube-style endings Whisper hallucinates
    "Thank you for watching",
    "Thanks for watching",
    "Subscribe to my channel",
    "Please subscribe",
    "Like and subscribe",
    // Short filler hallucinations
    "Thank you.",
    "Thanks.",
    "Bye.",
    "Bye-bye.",
    "you",
    "You",
    // Foreign language hallucinations (common with English models)
    "Sous-titres",
    "sous-titres",
    "Amara.org",
    // Repeated phrases (exact matches)
    "...",
    "♪",
];

/// Check if text is likely a Whisper hallucination
fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim();

    // Empty or whitespace-only
    if trimmed.is_empty() {
        return true;
    }

    // Exact match with known patterns
    for pattern in HALLUCINATION_PATTERNS {
        if trimmed.eq_ignore_ascii_case(pattern) {
            return true;
        }
    }

    // Contains bracketed annotations like [inaudible], [music], etc.
    if (trimmed.starts_with('[') && trimmed.ends_with(']'))
        || (trimmed.starts_with('(') && trimmed.ends_with(')'))
    {
        return true;
    }

    // Very short single-word responses that are likely hallucinations
    // (but allow short real words that might be commands)
    let word_count = trimmed.split_whitespace().count();
    if word_count == 1 && trimmed.len() <= 3 {
        return true;
    }

    false
}

impl Transcript {
    /// Get the full text without timestamps, filtering hallucinations
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.text.trim())
            .filter(|s| !is_hallucination(s))
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
        debug!(
            "Transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
        );

        // Pad short audio with silence to meet Whisper's minimum requirement
        let samples = if samples.len() < WHISPER_MIN_SAMPLES {
            debug!(
                "Padding audio from {} to {} samples",
                samples.len(),
                WHISPER_MIN_SAMPLES
            );
            let mut padded = samples.to_vec();
            padded.resize(WHISPER_MIN_SAMPLES, 0.0);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(samples)
        };

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
        state.full(params, &samples)
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

    /// Transcribe with streaming segment callbacks
    ///
    /// The callback is invoked for each segment as it's transcribed, allowing
    /// the caller to display partial results. The callback receives the
    /// cumulative text so far (all segments joined).
    ///
    /// Returns the final complete transcript.
    pub fn transcribe_streaming<F>(
        &self,
        samples: &[f32],
        mut on_segment: F,
    ) -> TranscribeResult<Transcript>
    where
        F: FnMut(&str) + Send + 'static,
    {
        debug!(
            "Transcribing {} samples ({:.2}s) with streaming",
            samples.len(),
            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
        );

        // Pad short audio with silence to meet Whisper's minimum requirement
        let samples = if samples.len() < WHISPER_MIN_SAMPLES {
            debug!(
                "Padding audio from {} to {} samples",
                samples.len(),
                WHISPER_MIN_SAMPLES
            );
            let mut padded = samples.to_vec();
            padded.resize(WHISPER_MIN_SAMPLES, 0.0);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(samples)
        };

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });

        // Configure parameters for better results
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        // Accumulate segments as they arrive
        let segments_accumulator = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let segments_for_callback = segments_accumulator.clone();

        // Set up segment callback - fires as each segment is transcribed
        params.set_segment_callback_safe(move |segment_data: SegmentCallbackData| {
            let trimmed = segment_data.text.trim();

            // Skip empty or hallucination segments
            if !trimmed.is_empty() && !is_hallucination(trimmed) {
                let mut segments = segments_for_callback.lock().unwrap();
                segments.push(trimmed.to_string());

                // Build cumulative text and invoke callback
                let cumulative = segments.join(" ");
                on_segment(&cumulative);
            }
        });

        // Create state for this transcription
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        // Run transcription (callbacks fire during this)
        state
            .full(params, &samples)
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        // Extract final segments for the Transcript struct
        let num_segments = state
            .full_n_segments()
            .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

        debug!("Got {} segments", num_segments);

        let mut segments = Vec::new();
        for i in 0..num_segments {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

            let start = state
                .full_get_segment_t0(i)
                .map_err(|e| TranscribeError::Transcription(e.to_string()))?;

            let end = state
                .full_get_segment_t1(i)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hallucination_filter_known_patterns() {
        // Known hallucination patterns should be filtered
        assert!(is_hallucination("[inaudible]"));
        assert!(is_hallucination("(inaudible)"));
        assert!(is_hallucination("[BLANK_AUDIO]"));
        assert!(is_hallucination("[Music]"));
        assert!(is_hallucination("Thank you for watching"));
        assert!(is_hallucination("THANK YOU FOR WATCHING")); // case insensitive
        assert!(is_hallucination("Thanks."));
        assert!(is_hallucination("Bye."));
        assert!(is_hallucination("you"));
    }

    #[test]
    fn test_hallucination_filter_bracketed() {
        // Any bracketed text should be filtered
        assert!(is_hallucination("[anything]"));
        assert!(is_hallucination("(something)"));
        assert!(is_hallucination("[random noise]"));
    }

    #[test]
    fn test_hallucination_filter_short_words() {
        // Very short single words (3 chars or less) are filtered
        assert!(is_hallucination("ok"));
        assert!(is_hallucination("um"));
        assert!(is_hallucination("uh"));
    }

    #[test]
    fn test_hallucination_filter_real_speech() {
        // Real speech should NOT be filtered
        assert!(!is_hallucination("Hello world"));
        assert!(!is_hallucination("This is a test"));
        assert!(!is_hallucination("Send email to Bob"));
        assert!(!is_hallucination("open"));  // 4 chars, single word - not filtered
        assert!(!is_hallucination("Hello")); // Real greeting
    }

    #[test]
    fn test_transcript_text_filters_hallucinations() {
        let transcript = Transcript {
            segments: vec![
                TranscriptSegment { start_ms: 0, end_ms: 100, text: "Hello".to_string() },
                TranscriptSegment { start_ms: 100, end_ms: 200, text: "[inaudible]".to_string() },
                TranscriptSegment { start_ms: 200, end_ms: 300, text: "world".to_string() },
                TranscriptSegment { start_ms: 300, end_ms: 400, text: "Thank you for watching".to_string() },
            ],
        };

        assert_eq!(transcript.text(), "Hello world");
    }
}
