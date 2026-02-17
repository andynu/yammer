//! Dictation pipeline - coordinates audio capture, STT, LLM correction, and text output

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use yammer_audio::{AudioCapture, VadProcessor, VadEvent, resample_to_kyutai, Vad};
use yammer_output::{TextOutput, OutputMethod};
use yammer_stt::{KyutaiTranscriber, KYUTAI_CHUNK_SAMPLES};
use yammer_llm::Corrector;

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    // Idle and Reloading are used as string values emitted to the frontend
    // even though they are not explicitly constructed in Rust code paths.
    #[allow(dead_code)]
    Idle,
    Listening,
    Correcting,
    Done,
    Error,
    Discarded,
    #[allow(dead_code)]
    Reloading,
}

impl PipelineState {
    pub fn as_str(&self) -> &'static str {
        match self {
            PipelineState::Idle => "idle",
            PipelineState::Listening => "listening",
            PipelineState::Correcting => "correcting",
            PipelineState::Done => "done",
            PipelineState::Error => "error",
            PipelineState::Discarded => "discarded",
            PipelineState::Reloading => "reloading",
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
    /// HuggingFace repo ID for the STT model (e.g. "kyutai/stt-1b-en_fr-candle")
    pub stt_model_repo: String,
    pub llm_model_path: Option<std::path::PathBuf>,
    pub use_llm_correction: bool,
    pub output_method: OutputMethod,
    /// Delay between keystrokes in milliseconds (for Type output method)
    pub typing_delay_ms: u32,
    pub vad_threshold: f64,
    pub audio_device: Option<String>,
    /// Maximum recording duration in seconds (0 = no limit)
    pub max_recording_seconds: u32,
    /// Seconds of continuous silence before auto-stopping recording (0 = disabled)
    pub silence_timeout_seconds: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stt_model_repo: "kyutai/stt-1b-en_fr-candle".to_string(),
            llm_model_path: None,
            use_llm_correction: false,
            output_method: OutputMethod::Type,
            typing_delay_ms: 0,
            vad_threshold: 0.01,
            audio_device: None,
            max_recording_seconds: 0,
            silence_timeout_seconds: 5,
        }
    }
}

/// The dictation pipeline
pub struct DictationPipeline {
    config: PipelineConfig,
    is_cancelled: Arc<AtomicBool>,
    /// If true, cancel completely without processing/outputting
    is_discarded: Arc<AtomicBool>,
    transcriber: Option<KyutaiTranscriber>,
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
        info!("STT model repo: {}", self.config.stt_model_repo);

        // Load Kyutai STT model (auto-selects CUDA if available)
        let t0 = Instant::now();
        match KyutaiTranscriber::new(&self.config.stt_model_repo) {
            Ok(transcriber) => {
                self.transcriber = Some(transcriber);
                info!("STT model loaded in {:.2}s", t0.elapsed().as_secs_f64());
            }
            Err(e) => {
                let err = format!("Failed to load STT model: {}", e);
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
                    let t0 = Instant::now();
                    match Corrector::new(llm_path) {
                        Ok(corrector) => {
                            self.corrector = Some(Arc::new(corrector));
                            info!("LLM model loaded in {:.2}s", t0.elapsed().as_secs_f64());
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

    /// Get a handle to the cancel flag
    pub fn get_cancel_handle(&self) -> Arc<AtomicBool> {
        self.is_cancelled.clone()
    }

    /// Get a handle to the discard flag
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

    /// Run the complete dictation pipeline (blocking).
    /// Should be called from spawn_blocking.
    pub fn run_blocking(&mut self) -> Result<String, String> {
        info!("Starting dictation pipeline...");
        self.reset_cancel();

        if !self.is_initialized() {
            let err = "Pipeline not initialized".to_string();
            self.send_error(err.clone());
            self.send_state(PipelineState::Error);
            return Err(err);
        }

        // 1. Listen + transcribe concurrently (Kyutai streams words as user speaks)
        let text = match self.listen_and_transcribe_blocking() {
            Ok(t) => t,
            Err(e) => {
                if e == "No audio recorded" || e == "Discarded" {
                    self.send_state(PipelineState::Discarded);
                } else {
                    self.send_error(e.clone());
                    self.send_state(PipelineState::Error);
                }
                return Err(e);
            }
        };

        // 2. Correct (optional LLM pass)
        let corrected = match self.correct_blocking(&text) {
            Ok(t) => t,
            Err(e) if e == "Cancelled" => {
                if self.is_discarded() {
                    self.send_state(PipelineState::Discarded);
                    return Err("Discarded".to_string());
                }
                self.output_blocking(&text)?;
                return Ok(text);
            }
            Err(e) => {
                warn!("Correction failed, outputting uncorrected: {}", e);
                self.output_blocking(&text)?;
                return Ok(text);
            }
        };

        if self.is_discarded() {
            self.send_state(PipelineState::Discarded);
            return Err("Discarded".to_string());
        }

        // 3. Output text
        match self.output_blocking(&corrected) {
            Ok(()) => Ok(corrected),
            Err(e) => {
                self.send_error(e.clone());
                self.send_state(PipelineState::Error);
                Err(e)
            }
        }
    }

    /// Listen and transcribe concurrently.
    ///
    /// Audio is captured in chunks, resampled to 24 kHz, fed to Kyutai in
    /// 1920-sample (80 ms) steps, and words are emitted to the UI as they
    /// are recognised — while the user is still speaking.
    fn listen_and_transcribe_blocking(&mut self) -> Result<String, String> {
        self.send_state(PipelineState::Listening);
        info!("Starting audio capture + streaming transcription...");

        // Set up audio capture
        let fallback_capture: Option<AudioCapture>;
        let (capture, input_sample_rate, _handle, rx) =
            match self.try_start_capture(&self.audio_capture) {
                Ok((c, sr, h, r)) => (c, sr, h, r),
                Err(pre_init_err) => {
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

        let _ = capture;
        info!("Audio capture ready: {} Hz", input_sample_rate);
        let mut rx = rx;

        // VAD for audio level + speech detection
        let mut vad = VadProcessor::with_threshold(self.config.vad_threshold as f32);

        // Silence timeout tracking
        let silence_timeout = if self.config.silence_timeout_seconds > 0 {
            Some(std::time::Duration::from_secs(self.config.silence_timeout_seconds as u64))
        } else {
            None
        };
        let recording_start = Instant::now();
        let mut ever_had_speech = false;

        // Carry buffer: accumulates resampled 24 kHz samples between chunks
        let mut carry_buf: Vec<f32> = Vec::with_capacity(KYUTAI_CHUNK_SAMPLES * 4);

        // Accumulated words from STT
        let mut accumulated_words: Vec<String> = Vec::new();

        while let Some(chunk) = rx.blocking_recv() {
            // Check discard/cancel before doing any work
            if self.is_discarded() {
                info!("Recording discarded during listen+transcribe");
                return Err("Discarded".to_string());
            }

            if self.is_cancelled() {
                info!(
                    "Listening stopped by user, {} accumulated words",
                    accumulated_words.len()
                );
                // Flush remaining carry buffer as one last step
                if let Some(ref mut t) = self.transcriber {
                    Self::flush_carry_buf(t, &mut carry_buf, &mut accumulated_words);
                }
                break;
            }

            // Audio level for the waveform visualisation (on raw device-rate samples)
            let rms = Vad::calculate_rms(&chunk);
            self.send_audio_level(rms);

            // VAD for speech detection
            let events = vad.process(&chunk);
            for event in &events {
                if matches!(event, VadEvent::SpeechStart) {
                    debug!("Speech started");
                    ever_had_speech = true;
                }
            }

            // Silence timeout: discard if no speech detected at all within the timeout
            if !ever_had_speech {
                if let Some(timeout) = silence_timeout {
                    if recording_start.elapsed() >= timeout {
                        info!("Silence timeout with no speech detected, discarding");
                        return Err("No audio recorded".to_string());
                    }
                }
            }

            // Resample to 24 kHz and extend carry buffer
            let chunk_24k = resample_to_kyutai(&chunk, input_sample_rate)
                .map_err(|e| format!("Resampling failed: {}", e))?;
            carry_buf.extend_from_slice(&chunk_24k);

            // Drain carry buffer in 1920-sample steps
            while carry_buf.len() >= KYUTAI_CHUNK_SAMPLES {
                let slice: Vec<f32> = carry_buf.drain(..KYUTAI_CHUNK_SAMPLES).collect();

                let words = self.transcriber.as_mut().unwrap()
                    .step(&slice)
                    .map_err(|e| format!("STT step failed: {}", e))?;

                if !words.is_empty() {
                    accumulated_words.extend(words);
                    let partial_text = accumulated_words.join(" ");
                    debug!("Partial transcript: {:?}", partial_text);
                    self.send_transcript(partial_text, true);
                }
            }

            // Safety max-duration timeout
            let max_seconds = self.config.max_recording_seconds;
            if max_seconds > 0 {
                let estimated_seconds = accumulated_words.len() as f32 * 0.4; // rough estimate
                let elapsed = recording_start.elapsed().as_secs() as u32;
                if elapsed >= max_seconds {
                    warn!("Recording timeout ({}s max)", max_seconds);
                    if let Some(ref mut t) = self.transcriber {
                        Self::flush_carry_buf(t, &mut carry_buf, &mut accumulated_words);
                    }
                    let _ = estimated_seconds; // suppress warning
                    break;
                }
            }
        }

        let final_text = accumulated_words.join(" ");

        if final_text.trim().is_empty() {
            warn!("No speech recognised");
            return Err("No audio recorded".to_string());
        }

        info!("Transcription complete: {:?}", final_text);
        // Send final transcript (partial=true if LLM correction will follow)
        self.send_transcript(final_text.clone(), self.corrector.is_some());

        Ok(final_text)
    }

    /// Pad the carry buffer with zeros and run one final step to flush any
    /// partial word the model is accumulating.
    fn flush_carry_buf(
        transcriber: &mut KyutaiTranscriber,
        carry_buf: &mut Vec<f32>,
        accumulated_words: &mut Vec<String>,
    ) {
        if carry_buf.is_empty() {
            return;
        }
        // Pad to a full chunk
        carry_buf.resize(KYUTAI_CHUNK_SAMPLES, 0.0);
        if let Ok(words) = transcriber.step(carry_buf) {
            accumulated_words.extend(words);
        }
        carry_buf.clear();
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

        info!("Correcting text: {:?}", text);

        let start = Instant::now();

        let result = match corrector.correct(text) {
            Ok(r) => r,
            Err(e) => {
                let err = format!("Correction failed: {}", e);
                error!("{}", err);
                return Err(err);
            }
        };

        info!(
            "Correction complete in {:.2}s: {:?}",
            start.elapsed().as_secs_f32(),
            result.text
        );

        self.send_transcript(result.text.clone(), false);

        Ok(result.text)
    }

    /// Output text (blocking)
    fn output_blocking(&self, text: &str) -> Result<(), String> {
        if text.trim().is_empty() {
            warn!("Empty text, nothing to output");
            self.send_state(PipelineState::Done);
            return Ok(());
        }

        info!("Outputting text: {:?}", text);

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
        assert_eq!(PipelineState::Correcting.as_str(), "correcting");
        assert_eq!(PipelineState::Done.as_str(), "done");
        assert_eq!(PipelineState::Error.as_str(), "error");
        assert_eq!(PipelineState::Discarded.as_str(), "discarded");
        assert_eq!(PipelineState::Reloading.as_str(), "reloading");
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.stt_model_repo, "kyutai/stt-1b-en_fr-candle");
        assert!(config.llm_model_path.is_none());
        assert!(!config.use_llm_correction);
        assert_eq!(config.output_method, OutputMethod::Type);
        assert_eq!(config.typing_delay_ms, 0);
        assert!((config.vad_threshold - 0.01).abs() < f64::EPSILON);
        assert!(config.audio_device.is_none());
        assert_eq!(config.silence_timeout_seconds, 5);
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
        assert!(!pipeline.is_cancelled());

        cancel_handle.store(true, Ordering::SeqCst);
        assert!(pipeline.is_cancelled());

        pipeline.reset_cancel();
        assert!(!pipeline.is_cancelled());
    }

    #[test]
    fn test_discard_handle() {
        let (tx, _rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        let discard_handle = pipeline.get_discard_handle();
        assert!(!pipeline.is_discarded());

        discard_handle.store(true, Ordering::SeqCst);
        assert!(pipeline.is_discarded());

        pipeline.reset_cancel();
        assert!(!pipeline.is_discarded());
    }

    #[test]
    fn test_pipeline_events_sent() {
        let (tx, mut rx) = mpsc::channel(10);
        let pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        pipeline.send_state(PipelineState::Listening);

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
        let mut pipeline = DictationPipeline::new(PipelineConfig::default(), tx);

        let result = pipeline.run_blocking();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Pipeline not initialized");

        let mut found_error_state = false;
        while let Ok(event) = rx.try_recv() {
            if let PipelineEvent::StateChanged(PipelineState::Error) = event {
                found_error_state = true;
            }
        }
        assert!(found_error_state);
    }
}
