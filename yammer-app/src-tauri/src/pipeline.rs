//! Dictation pipeline - coordinates audio capture, STT, LLM correction, and text output

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use yammer_audio::{AudioCapture, VadProcessor, VadEvent, resample_to_whisper, Vad, WHISPER_SAMPLE_RATE};
use yammer_output::{TextOutput, OutputMethod};
use yammer_stt::Transcriber;
use yammer_llm::Corrector;

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Listening,
    Processing,
    Correcting,
    Done,
    Error,
    Discarded,
}

impl PipelineState {
    pub fn as_str(&self) -> &'static str {
        match self {
            PipelineState::Idle => "idle",
            PipelineState::Listening => "listening",
            PipelineState::Processing => "processing",
            PipelineState::Correcting => "correcting",
            PipelineState::Done => "done",
            PipelineState::Error => "error",
            PipelineState::Discarded => "discarded",
        }
    }
}

/// Events emitted by the pipeline
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    StateChanged(PipelineState),
    AudioLevel(f32),
    Transcript { text: String, is_partial: bool },
    Error(String),
}

/// Configuration for the pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub whisper_model_path: PathBuf,
    pub llm_model_path: Option<PathBuf>,
    pub use_llm_correction: bool,
    pub output_method: OutputMethod,
    /// Delay between keystrokes in milliseconds (for Type output method)
    pub typing_delay_ms: u32,
    pub vad_threshold: f64,
    pub audio_device: Option<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            whisper_model_path: PathBuf::new(),
            llm_model_path: None,
            use_llm_correction: false,
            output_method: OutputMethod::Type,
            typing_delay_ms: 0,
            vad_threshold: 0.01,
            audio_device: None,
        }
    }
}

/// The dictation pipeline
pub struct DictationPipeline {
    config: PipelineConfig,
    is_cancelled: Arc<AtomicBool>,
    /// If true, cancel completely without processing/outputting
    is_discarded: Arc<AtomicBool>,
    transcriber: Option<Arc<Transcriber>>,
    corrector: Option<Arc<Corrector>>,
    event_tx: mpsc::Sender<PipelineEvent>,
    /// Pre-initialized audio capture to reduce keypress-to-listening latency
    audio_capture: Option<AudioCapture>,
}

impl DictationPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig, event_tx: mpsc::Sender<PipelineEvent>) -> Self {
        Self {
            config,
            is_cancelled: Arc::new(AtomicBool::new(false)),
            is_discarded: Arc::new(AtomicBool::new(false)),
            transcriber: None,
            corrector: None,
            event_tx,
            audio_capture: None,
        }
    }

    /// Initialize the pipeline (load models)
    pub fn initialize(&mut self) -> Result<(), String> {
        info!("Initializing dictation pipeline...");
        info!("Whisper model: {:?}", self.config.whisper_model_path);

        // Load Whisper model
        if !self.config.whisper_model_path.exists() {
            let err = format!(
                "Whisper model not found: {:?}",
                self.config.whisper_model_path
            );
            error!("{}", err);
            return Err(err);
        }

        info!("Loading Whisper model...");
        match Transcriber::new(&self.config.whisper_model_path) {
            Ok(transcriber) => {
                self.transcriber = Some(Arc::new(transcriber));
                info!("Whisper model loaded successfully");
            }
            Err(e) => {
                let err = format!("Failed to load Whisper model: {}", e);
                error!("{}", err);
                return Err(err);
            }
        }

        // Load LLM model if configured
        if self.config.use_llm_correction {
            if let Some(ref llm_path) = self.config.llm_model_path {
                if !llm_path.exists() {
                    warn!("LLM model not found: {:?}, correction disabled", llm_path);
                } else {
                    info!("Loading LLM model from {:?}...", llm_path);
                    match Corrector::new(llm_path) {
                        Ok(corrector) => {
                            self.corrector = Some(Arc::new(corrector));
                            info!("LLM model loaded successfully");
                        }
                        Err(e) => {
                            let err = format!("Failed to load LLM model: {}", e);
                            error!("{}", err);
                            return Err(err);
                        }
                    }
                }
            }
        } else {
            info!("LLM correction disabled");
        }

        // Pre-initialize audio capture to reduce keypress-to-listening latency
        info!("Pre-initializing audio capture...");
        let capture = if let Some(ref device_name) = self.config.audio_device {
            info!("Using configured audio device: {}", device_name);
            match AudioCapture::with_device(device_name) {
                Ok(c) => {
                    info!(
                        "Audio capture initialized: {} Hz, {} channels",
                        c.sample_rate(),
                        c.channels()
                    );
                    Some(c)
                }
                Err(e) => {
                    warn!("Failed to initialize audio device '{}': {}", device_name, e);
                    warn!("Audio capture will be initialized on first use");
                    None
                }
            }
        } else {
            info!("Using default audio device");
            match AudioCapture::new() {
                Ok(c) => {
                    info!(
                        "Audio capture initialized: {} Hz, {} channels",
                        c.sample_rate(),
                        c.channels()
                    );
                    Some(c)
                }
                Err(e) => {
                    warn!("Failed to initialize default audio: {}", e);
                    warn!("Audio capture will be initialized on first use");
                    None
                }
            }
        };
        self.audio_capture = capture;

        info!("Pipeline initialization complete");
        Ok(())
    }

    /// Check if pipeline is initialized
    pub fn is_initialized(&self) -> bool {
        self.transcriber.is_some()
    }

    /// Get a handle to the cancel flag that can be used to stop the pipeline
    /// without needing to hold the pipeline lock
    pub fn get_cancel_handle(&self) -> Arc<AtomicBool> {
        self.is_cancelled.clone()
    }

    /// Get a handle to the discard flag that can be used to discard the recording
    /// without needing to hold the pipeline lock
    pub fn get_discard_handle(&self) -> Arc<AtomicBool> {
        self.is_discarded.clone()
    }

    fn send_state(&self, state: PipelineState) {
        debug!("Pipeline state: {:?}", state);
        let _ = self.event_tx.try_send(PipelineEvent::StateChanged(state));
    }

    fn send_audio_level(&self, level: f32) {
        let _ = self.event_tx.try_send(PipelineEvent::AudioLevel(level));
    }

    fn send_transcript(&self, text: String, is_partial: bool) {
        let _ = self.event_tx.try_send(PipelineEvent::Transcript { text, is_partial });
    }

    fn send_error(&self, error: String) {
        error!("Pipeline error: {}", error);
        let _ = self.event_tx.try_send(PipelineEvent::Error(error));
    }

    /// Reset cancelled and discarded flags
    fn reset_cancel(&self) {
        self.is_cancelled.store(false, Ordering::SeqCst);
        self.is_discarded.store(false, Ordering::SeqCst);
    }

    fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::SeqCst)
    }

    fn is_discarded(&self) -> bool {
        self.is_discarded.load(Ordering::SeqCst)
    }

    /// Create a new audio capture instance based on config
    fn create_audio_capture(&self) -> Result<AudioCapture, String> {
        if let Some(ref device_name) = self.config.audio_device {
            info!("Using configured audio device: {}", device_name);
            AudioCapture::with_device(device_name).map_err(|e| {
                let err = format!("Failed to initialize audio device '{}': {}", device_name, e);
                error!("{}", err);
                err
            })
        } else {
            info!("Using default audio device");
            AudioCapture::new().map_err(|e| {
                let err = format!("Failed to initialize audio: {}", e);
                error!("{}", err);
                err
            })
        }
    }

    /// Try to start capture on a pre-initialized AudioCapture
    /// Returns the capture reference, sample rate, handle, and receiver
    fn try_start_capture<'a>(
        &self,
        capture: &'a Option<AudioCapture>,
    ) -> Result<
        (
            &'a AudioCapture,
            u32,
            yammer_audio::CaptureHandle,
            mpsc::Receiver<Vec<f32>>,
        ),
        String,
    > {
        let c = capture
            .as_ref()
            .ok_or_else(|| "No pre-initialized capture".to_string())?;
        let sr = c.sample_rate();
        debug!("Using pre-initialized audio capture: {} Hz", sr);
        let (h, r) = c
            .start_capture(50)
            .map_err(|e| format!("Failed to start pre-initialized capture: {}", e))?;
        Ok((c, sr, h, r))
    }

    /// Run the complete dictation pipeline (blocking)
    /// This should be called from spawn_blocking
    pub fn run_blocking(&self) -> Result<String, String> {
        info!("Starting dictation pipeline...");
        self.reset_cancel();

        // Check initialized
        if !self.is_initialized() {
            let err = "Pipeline not initialized".to_string();
            self.send_error(err.clone());
            self.send_state(PipelineState::Error);
            return Err(err);
        }

        // 1. Listen for speech (cancel flag is used to stop listening)
        let samples = match self.listen_blocking() {
            Ok(s) => s,
            Err(e) => {
                if e == "No audio recorded" || e == "Discarded" {
                    self.send_state(PipelineState::Idle);
                } else {
                    self.send_error(e.clone());
                    self.send_state(PipelineState::Error);
                }
                return Err(e);
            }
        };

        // Check if discarded after listening - skip all processing
        if self.is_discarded() {
            info!("Recording discarded by user");
            self.send_state(PipelineState::Discarded);
            return Err("Discarded".to_string());
        }

        // Reset cancel flag - user clicked stop to END listening, not to cancel processing
        // We have audio samples, so proceed with transcription and correction
        // Note: we do NOT reset is_discarded - it can be set at any time
        self.is_cancelled.store(false, Ordering::SeqCst);

        // 2. Transcribe
        let text = match self.transcribe_blocking(&samples) {
            Ok(t) => t,
            Err(e) => {
                if e == "Cancelled" || e == "Discarded" {
                    self.send_state(PipelineState::Idle);
                } else {
                    self.send_error(e.clone());
                    self.send_state(PipelineState::Error);
                }
                return Err(e);
            }
        };

        // Check if discarded after transcription - skip output
        if self.is_discarded() {
            info!("Recording discarded by user after transcription");
            self.send_state(PipelineState::Discarded);
            return Err("Discarded".to_string());
        }

        // 3. Correct (optional)
        let corrected = match self.correct_blocking(&text) {
            Ok(t) => t,
            Err(e) if e == "Cancelled" => {
                // Check for discard during correction
                if self.is_discarded() {
                    info!("Recording discarded by user during correction");
                    self.send_state(PipelineState::Discarded);
                    return Err("Discarded".to_string());
                }
                // Output uncorrected text on cancel during correction
                self.output_blocking(&text)?;
                return Ok(text);
            }
            Err(e) => {
                // On correction error, output uncorrected text
                warn!("Correction failed, outputting uncorrected: {}", e);
                self.output_blocking(&text)?;
                return Ok(text);
            }
        };

        // Final check before output
        if self.is_discarded() {
            info!("Recording discarded by user before output");
            self.send_state(PipelineState::Discarded);
            return Err("Discarded".to_string());
        }

        // 4. Output text
        match self.output_blocking(&corrected) {
            Ok(()) => Ok(corrected),
            Err(e) => {
                self.send_error(e.clone());
                self.send_state(PipelineState::Error);
                Err(e)
            }
        }
    }

    /// Listen for audio (blocking version)
    fn listen_blocking(&self) -> Result<Vec<f32>, String> {
        self.send_state(PipelineState::Listening);
        info!("Starting audio capture...");

        // Try to use pre-initialized capture, falling back to on-demand creation
        let fallback_capture: Option<AudioCapture>;
        let (capture, input_sample_rate, _handle, rx) =
            match self.try_start_capture(&self.audio_capture) {
                Ok((c, sr, h, r)) => (c, sr, h, r),
                Err(pre_init_err) => {
                    // Pre-initialized capture failed (device disconnected?), try fresh init
                    warn!(
                        "Pre-initialized capture failed: {}, attempting fresh initialization",
                        pre_init_err
                    );
                    fallback_capture = Some(self.create_audio_capture()?);
                    let c = fallback_capture.as_ref().unwrap();
                    let sr = c.sample_rate();
                    match c.start_capture(50) {
                        Ok((h, r)) => (c, sr, h, r),
                        Err(e) => {
                            let err = format!("Failed to start capture: {}", e);
                            error!("{}", err);
                            return Err(err);
                        }
                    }
                }
            };

        let _ = capture; // Silence unused warning (we just need to keep fallback_capture alive)
        info!("Audio capture ready: {} Hz", input_sample_rate);
        let mut rx = rx;

        // VAD processor (still useful for detecting speech patterns)
        let mut vad = VadProcessor::with_threshold(self.config.vad_threshold as f32);
        let mut all_samples: Vec<f32> = Vec::new();

        // Process audio chunks in a blocking loop
        // Note: We use blocking_recv since we're in spawn_blocking
        // For click-to-toggle mode: collect ALL audio, not just VAD-detected speech
        while let Some(chunk) = rx.blocking_recv() {
            // Always collect samples while listening (for click-to-toggle mode)
            all_samples.extend_from_slice(&chunk);

            // Check for discard (user wants to cancel completely)
            if self.is_discarded() {
                info!("Recording discarded by user during listening");
                return Err("Discarded".to_string());
            }

            // Check for cancellation (user clicked stop to process audio)
            if self.is_cancelled() {
                info!("Listening stopped by user, {} samples collected", all_samples.len());

                if all_samples.is_empty() {
                    warn!("No audio recorded");
                    return Err("No audio recorded".to_string());
                }

                // Process whatever we have
                info!(
                    "Processing {} samples ({:.2}s) at {} Hz",
                    all_samples.len(),
                    all_samples.len() as f32 / input_sample_rate as f32,
                    input_sample_rate
                );
                return self.maybe_resample(all_samples, input_sample_rate);
            }

            // Calculate audio level for visualization
            let rms = Vad::calculate_rms(&chunk);
            self.send_audio_level(rms);

            // Process through VAD (still useful for detecting natural speech end)
            let events = vad.process(&chunk);

            for event in events {
                match event {
                    VadEvent::SpeechStart => {
                        debug!("Speech started");
                    }
                    VadEvent::Speaking => {
                        // Samples already collected above
                    }
                    VadEvent::SpeechEnd { samples: _ } => {
                        // In click-to-toggle mode, we don't auto-stop on speech end
                        // User controls when to stop via the button
                        debug!("VAD detected speech end (continuing recording)");
                    }
                    VadEvent::Silent => {
                        // Continue recording
                    }
                }
            }

            // Safety timeout: if recording for too long, stop
            if all_samples.len() > input_sample_rate as usize * 30 {
                warn!("Recording timeout (30s max)");

                info!(
                    "Processing {} samples ({:.2}s) at {} Hz",
                    all_samples.len(),
                    all_samples.len() as f32 / input_sample_rate as f32,
                    input_sample_rate
                );

                return self.maybe_resample(all_samples, input_sample_rate);
            }
        }

        // Channel closed - process whatever we have
        if !all_samples.is_empty() {
            info!(
                "Audio capture ended, processing {} samples ({:.2}s)",
                all_samples.len(),
                all_samples.len() as f32 / input_sample_rate as f32
            );
            self.maybe_resample(all_samples, input_sample_rate)
        } else {
            warn!("Audio capture ended with no samples");
            Err("No audio recorded".to_string())
        }
    }

    fn maybe_resample(&self, samples: Vec<f32>, input_rate: u32) -> Result<Vec<f32>, String> {
        if input_rate != WHISPER_SAMPLE_RATE {
            info!("Resampling from {} Hz to {} Hz", input_rate, WHISPER_SAMPLE_RATE);
            resample_to_whisper(&samples, input_rate).map_err(|e| {
                let err = format!("Resampling failed: {}", e);
                error!("{}", err);
                err
            })
        } else {
            Ok(samples)
        }
    }

    /// Transcribe audio samples (blocking)
    fn transcribe_blocking(&self, samples: &[f32]) -> Result<String, String> {
        self.send_state(PipelineState::Processing);

        if self.is_cancelled() {
            return Err("Cancelled".to_string());
        }

        let transcriber = self
            .transcriber
            .as_ref()
            .ok_or("Transcriber not initialized")?;

        info!(
            "Transcribing {} samples ({:.2}s)",
            samples.len(),
            samples.len() as f32 / WHISPER_SAMPLE_RATE as f32
        );

        let start = Instant::now();

        let transcript = match transcriber.transcribe(samples) {
            Ok(t) => t,
            Err(e) => {
                let err = format!("Transcription failed: {}", e);
                error!("{}", err);
                return Err(err);
            }
        };

        let text = transcript.text();
        let elapsed = start.elapsed();

        info!(
            "Transcription complete in {:.2}s: \"{}\"",
            elapsed.as_secs_f32(),
            text
        );

        // Send transcript (partial if LLM correction is pending)
        self.send_transcript(text.clone(), self.corrector.is_some());

        Ok(text)
    }

    /// Correct text using LLM (blocking)
    fn correct_blocking(&self, text: &str) -> Result<String, String> {
        if !self.config.use_llm_correction || self.corrector.is_none() {
            return Ok(text.to_string());
        }

        self.send_state(PipelineState::Correcting);

        if self.is_cancelled() {
            return Err("Cancelled".to_string());
        }

        let corrector = self.corrector.as_ref().unwrap();

        info!("Correcting text: \"{}\"", text);

        let start = Instant::now();

        let result = match corrector.correct(text) {
            Ok(r) => r,
            Err(e) => {
                let err = format!("Correction failed: {}", e);
                error!("{}", err);
                return Err(err);
            }
        };

        let elapsed = start.elapsed();
        info!(
            "Correction complete in {:.2}s: \"{}\"",
            elapsed.as_secs_f32(),
            result.text
        );

        // Send final result
        self.send_transcript(result.text.clone(), false);

        Ok(result.text)
    }

    /// Check if text contains only Whisper special tokens (not real speech)
    fn is_special_token_only(text: &str) -> bool {
        let trimmed = text.trim();
        // Common Whisper special tokens that indicate non-speech
        let special_patterns = [
            "[BLANK_AUDIO]",
            "[MUSIC]",
            "[INAUDIBLE]",
            "(music)",
            "(inaudible)",
            "(silence)",
            "[SILENCE]",
        ];
        special_patterns.iter().any(|&p| trimmed.eq_ignore_ascii_case(p))
    }

    /// Output text (blocking)
    fn output_blocking(&self, text: &str) -> Result<(), String> {
        if text.is_empty() {
            warn!("Empty text, nothing to output");
            self.send_state(PipelineState::Done);
            return Ok(());
        }

        // Skip Whisper special tokens
        if Self::is_special_token_only(text) {
            warn!("Skipping special token: \"{}\"", text);
            self.send_state(PipelineState::Done);
            return Ok(());
        }

        info!("Outputting text: \"{}\"", text);

        let output = TextOutput::with_options(
            self.config.output_method,
            self.config.typing_delay_ms,
        );

        match output.output(text) {
            Ok(()) => {
                info!("Text output successful");
            }
            Err(e) => {
                let err = format!("Output failed: {}", e);
                error!("{}", err);
                return Err(err);
            }
        }

        self.send_state(PipelineState::Done);

        Ok(())
    }
}

// Compile-time assertion that DictationPipeline is Send + Sync.
// All fields implement these traits:
// - PipelineConfig: Contains PathBuf, Option<PathBuf>, bool, OutputMethod, f64, Option<String> (all Send+Sync)
// - Arc<AtomicBool>: Send+Sync
// - Option<Arc<Transcriber>>: WhisperContext implements Send+Sync (verified in whisper-rs 0.14.4)
// - Option<Arc<Corrector>>: LlamaModel implements Send+Sync (verified in llama_cpp 0.3.2)
// - tokio::sync::mpsc::Sender: Send+Sync
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DictationPipeline>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_state_as_str() {
        assert_eq!(PipelineState::Idle.as_str(), "idle");
        assert_eq!(PipelineState::Listening.as_str(), "listening");
        assert_eq!(PipelineState::Processing.as_str(), "processing");
        assert_eq!(PipelineState::Correcting.as_str(), "correcting");
        assert_eq!(PipelineState::Done.as_str(), "done");
        assert_eq!(PipelineState::Error.as_str(), "error");
        assert_eq!(PipelineState::Discarded.as_str(), "discarded");
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert!(config.whisper_model_path.as_os_str().is_empty());
        assert!(config.llm_model_path.is_none());
        assert!(!config.use_llm_correction);
        assert_eq!(config.output_method, OutputMethod::Type);
        assert_eq!(config.typing_delay_ms, 0);
        assert!((config.vad_threshold - 0.01).abs() < f64::EPSILON);
        assert!(config.audio_device.is_none());
    }

    #[test]
    fn test_is_special_token_only() {
        // These should be recognized as special tokens
        assert!(DictationPipeline::is_special_token_only("[BLANK_AUDIO]"));
        assert!(DictationPipeline::is_special_token_only("[MUSIC]"));
        assert!(DictationPipeline::is_special_token_only("[INAUDIBLE]"));
        assert!(DictationPipeline::is_special_token_only("(music)"));
        assert!(DictationPipeline::is_special_token_only("(inaudible)"));
        assert!(DictationPipeline::is_special_token_only("(silence)"));
        assert!(DictationPipeline::is_special_token_only("[SILENCE]"));

        // Case insensitive
        assert!(DictationPipeline::is_special_token_only("[blank_audio]"));
        assert!(DictationPipeline::is_special_token_only("[Blank_Audio]"));
        assert!(DictationPipeline::is_special_token_only("(MUSIC)"));

        // With whitespace
        assert!(DictationPipeline::is_special_token_only("  [BLANK_AUDIO]  "));
        assert!(DictationPipeline::is_special_token_only("\n[MUSIC]\n"));

        // These should NOT be special tokens
        assert!(!DictationPipeline::is_special_token_only("Hello world"));
        assert!(!DictationPipeline::is_special_token_only(""));
        assert!(!DictationPipeline::is_special_token_only("The [MUSIC] was great"));
        assert!(!DictationPipeline::is_special_token_only("[UNKNOWN]"));
    }

    #[test]
    fn test_pipeline_not_initialized() {
        let (tx, _rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        assert!(!pipeline.is_initialized());
    }

    #[test]
    fn test_cancel_handle() {
        let (tx, _rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        let cancel_handle = pipeline.get_cancel_handle();

        // Initially not cancelled
        assert!(!pipeline.is_cancelled());

        // Set cancel via handle
        cancel_handle.store(true, Ordering::SeqCst);
        assert!(pipeline.is_cancelled());

        // Reset
        pipeline.reset_cancel();
        assert!(!pipeline.is_cancelled());
    }

    #[test]
    fn test_discard_handle() {
        let (tx, _rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        let discard_handle = pipeline.get_discard_handle();

        // Initially not discarded
        assert!(!pipeline.is_discarded());

        // Set discard via handle
        discard_handle.store(true, Ordering::SeqCst);
        assert!(pipeline.is_discarded());

        // Reset (reset_cancel also resets discard)
        pipeline.reset_cancel();
        assert!(!pipeline.is_discarded());
    }

    #[test]
    fn test_pipeline_events_sent() {
        let (tx, mut rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        // Send state
        pipeline.send_state(PipelineState::Listening);

        // Verify event received
        let event = rx.try_recv().expect("Should receive event");
        match event {
            PipelineEvent::StateChanged(state) => {
                assert_eq!(state, PipelineState::Listening);
            }
            _ => panic!("Expected StateChanged event"),
        }
    }

    #[test]
    fn test_pipeline_run_without_initialization() {
        let (tx, mut rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        // Running without initialization should fail
        let result = pipeline.run_blocking();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Pipeline not initialized");

        // Should have sent error event
        let mut found_error_state = false;
        while let Ok(event) = rx.try_recv() {
            if let PipelineEvent::StateChanged(PipelineState::Error) = event {
                found_error_state = true;
            }
        }
        assert!(found_error_state);
    }

    #[test]
    fn test_pipeline_initialize_missing_model() {
        let (tx, _rx) = mpsc::channel(10);
        let mut pipeline = DictationPipeline::new(
            PipelineConfig {
                whisper_model_path: "/nonexistent/model.bin".into(),
                ..Default::default()
            },
            tx,
        );

        let result = pipeline.initialize();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }
}
