//! Dictation pipeline - coordinates audio capture, STT, LLM correction, and text output

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

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
    pub vad_threshold: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            whisper_model_path: PathBuf::new(),
            llm_model_path: None,
            use_llm_correction: false,
            output_method: OutputMethod::Type,
            vad_threshold: 0.01,
        }
    }
}

/// The dictation pipeline
pub struct DictationPipeline {
    config: PipelineConfig,
    is_cancelled: Arc<AtomicBool>,
    transcriber: Option<Arc<Transcriber>>,
    corrector: Option<Arc<Corrector>>,
    event_tx: mpsc::Sender<PipelineEvent>,
}

impl DictationPipeline {
    /// Create a new pipeline with the given configuration
    pub fn new(config: PipelineConfig, event_tx: mpsc::Sender<PipelineEvent>) -> Self {
        Self {
            config,
            is_cancelled: Arc::new(AtomicBool::new(false)),
            transcriber: None,
            corrector: None,
            event_tx,
        }
    }

    /// Initialize the pipeline (load models)
    pub fn initialize(&mut self) -> Result<(), String> {
        info!("Initializing dictation pipeline...");

        // Load Whisper model
        if !self.config.whisper_model_path.exists() {
            return Err(format!(
                "Whisper model not found: {:?}",
                self.config.whisper_model_path
            ));
        }

        self.transcriber = Some(Arc::new(
            Transcriber::new(&self.config.whisper_model_path)
                .map_err(|e| format!("Failed to load Whisper model: {}", e))?,
        ));
        info!("Whisper model loaded");

        // Load LLM model if configured
        if self.config.use_llm_correction {
            if let Some(ref llm_path) = self.config.llm_model_path {
                if !llm_path.exists() {
                    warn!("LLM model not found: {:?}, correction disabled", llm_path);
                } else {
                    self.corrector = Some(Arc::new(
                        Corrector::new(llm_path)
                            .map_err(|e| format!("Failed to load LLM model: {}", e))?,
                    ));
                    info!("LLM model loaded");
                }
            }
        }

        info!("Pipeline initialized successfully");
        Ok(())
    }

    /// Check if pipeline is initialized
    pub fn is_initialized(&self) -> bool {
        self.transcriber.is_some()
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
        let _ = self.event_tx.try_send(PipelineEvent::Error(error));
    }

    /// Cancel any in-progress operation
    pub fn cancel(&self) {
        info!("Pipeline cancel requested");
        self.is_cancelled.store(true, Ordering::SeqCst);
    }

    /// Reset cancelled flag
    fn reset_cancel(&self) {
        self.is_cancelled.store(false, Ordering::SeqCst);
    }

    fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::SeqCst)
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

        // 1. Listen for speech
        let samples = match self.listen_blocking() {
            Ok(s) => s,
            Err(e) => {
                if e == "Cancelled" {
                    self.send_state(PipelineState::Idle);
                } else {
                    self.send_error(e.clone());
                    self.send_state(PipelineState::Error);
                }
                return Err(e);
            }
        };

        // 2. Transcribe
        let text = match self.transcribe_blocking(&samples) {
            Ok(t) => t,
            Err(e) => {
                if e == "Cancelled" {
                    self.send_state(PipelineState::Idle);
                } else {
                    self.send_error(e.clone());
                    self.send_state(PipelineState::Error);
                }
                return Err(e);
            }
        };

        // 3. Correct (optional)
        let corrected = match self.correct_blocking(&text) {
            Ok(t) => t,
            Err(e) if e == "Cancelled" => {
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

        // Set up audio capture
        let capture = AudioCapture::new()
            .map_err(|e| format!("Failed to initialize audio: {}", e))?;

        let input_sample_rate = capture.sample_rate();
        info!("Audio capture initialized: {} Hz", input_sample_rate);

        // Start continuous capture (50ms chunks)
        let (_handle, mut rx) = capture
            .start_capture(50)
            .map_err(|e| format!("Failed to start capture: {}", e))?;

        // VAD processor
        let mut vad = VadProcessor::with_threshold(self.config.vad_threshold as f32);
        let mut all_samples: Vec<f32> = Vec::new();
        let mut speech_detected = false;

        // Process audio chunks in a blocking loop
        // Note: We use blocking_recv since we're in spawn_blocking
        while let Some(chunk) = rx.blocking_recv() {
            if self.is_cancelled() {
                info!("Listening cancelled");
                return Err("Cancelled".to_string());
            }

            // Calculate audio level for visualization
            let rms = Vad::calculate_rms(&chunk);
            self.send_audio_level(rms);

            // Process through VAD
            let events = vad.process(&chunk);

            for event in events {
                match event {
                    VadEvent::SpeechStart => {
                        debug!("Speech started");
                        speech_detected = true;
                    }
                    VadEvent::Speaking => {
                        // Accumulate samples during speech
                        all_samples.extend_from_slice(&chunk);
                    }
                    VadEvent::SpeechEnd { samples } => {
                        debug!("Speech ended, {} samples collected", samples.len());
                        all_samples.extend(samples);

                        if all_samples.is_empty() {
                            return Err("No speech detected".to_string());
                        }

                        info!(
                            "Captured {} samples ({:.2}s) at {} Hz",
                            all_samples.len(),
                            all_samples.len() as f32 / input_sample_rate as f32,
                            input_sample_rate
                        );

                        // Resample to Whisper format if needed
                        return self.maybe_resample(all_samples, input_sample_rate);
                    }
                    VadEvent::Silent => {
                        // If we haven't started speaking yet, that's fine
                    }
                }
            }

            // Safety timeout: if recording for too long, stop
            if all_samples.len() > input_sample_rate as usize * 30 {
                warn!("Recording timeout (30s max)");

                if all_samples.is_empty() {
                    return Err("No speech detected (timeout)".to_string());
                }

                return self.maybe_resample(all_samples, input_sample_rate);
            }
        }

        // Channel closed unexpectedly
        if speech_detected && !all_samples.is_empty() {
            self.maybe_resample(all_samples, input_sample_rate)
        } else {
            Err("Audio capture ended unexpectedly".to_string())
        }
    }

    fn maybe_resample(&self, samples: Vec<f32>, input_rate: u32) -> Result<Vec<f32>, String> {
        if input_rate != WHISPER_SAMPLE_RATE {
            info!("Resampling from {} Hz to {} Hz", input_rate, WHISPER_SAMPLE_RATE);
            resample_to_whisper(&samples, input_rate)
                .map_err(|e| format!("Resampling failed: {}", e))
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

        let transcript = transcriber
            .transcribe(samples)
            .map_err(|e| format!("Transcription failed: {}", e))?;

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

        let result = corrector
            .correct(text)
            .map_err(|e| format!("Correction failed: {}", e))?;

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

        let output = TextOutput::with_method(self.config.output_method);

        output
            .output(text)
            .map_err(|e| format!("Output failed: {}", e))?;

        self.send_state(PipelineState::Done);

        Ok(())
    }
}

// Make the pipeline thread-safe for sharing between threads
unsafe impl Send for DictationPipeline {}
unsafe impl Sync for DictationPipeline {}
