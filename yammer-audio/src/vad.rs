//! Voice Activity Detection
//!
//! Energy-based VAD with hysteresis to detect speech vs silence.

use tracing::debug;

/// Default RMS threshold for speech detection
pub const DEFAULT_SPEECH_THRESHOLD: f32 = 0.01;

/// Default number of consecutive speech frames to trigger speech start
pub const DEFAULT_SPEECH_START_FRAMES: usize = 3;

/// Default number of consecutive silence frames to trigger speech end
pub const DEFAULT_SPEECH_END_FRAMES: usize = 15;

/// Configuration for VAD
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// RMS threshold above which audio is considered speech
    pub speech_threshold: f32,
    /// Number of consecutive speech frames to confirm speech start
    pub speech_start_frames: usize,
    /// Number of consecutive silence frames to confirm speech end
    pub speech_end_frames: usize,
    /// Minimum speech duration in frames (shorter segments ignored)
    pub min_speech_frames: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: DEFAULT_SPEECH_THRESHOLD,
            speech_start_frames: DEFAULT_SPEECH_START_FRAMES,
            speech_end_frames: DEFAULT_SPEECH_END_FRAMES,
            min_speech_frames: 4, // ~250ms at typical frame sizes
        }
    }
}

/// Current state of VAD
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadState {
    /// Currently detecting silence
    Silence,
    /// Possibly starting speech (consecutive speech frames detected)
    MaybeSpeech,
    /// Confirmed speech
    Speech,
    /// Possibly ending speech (consecutive silence frames detected)
    MaybeSilence,
}

/// Voice activity detector with hysteresis
pub struct Vad {
    config: VadConfig,
    state: VadState,
    consecutive_speech: usize,
    consecutive_silence: usize,
    speech_frame_count: usize,
}

impl Vad {
    /// Create a new VAD instance with default configuration
    pub fn new() -> Self {
        Self::with_config(VadConfig::default())
    }

    /// Create a new VAD with custom configuration
    pub fn with_config(config: VadConfig) -> Self {
        Self {
            config,
            state: VadState::Silence,
            consecutive_speech: 0,
            consecutive_silence: 0,
            speech_frame_count: 0,
        }
    }

    /// Create a VAD with custom threshold
    pub fn with_threshold(threshold: f32) -> Self {
        let config = VadConfig {
            speech_threshold: threshold,
            speech_start_frames: DEFAULT_SPEECH_START_FRAMES,
            speech_end_frames: DEFAULT_SPEECH_END_FRAMES,
            min_speech_frames: 4, // ~250ms at typical frame sizes
        };
        Self::with_config(config)
    }

    /// Get the current VAD state
    pub fn state(&self) -> VadState {
        self.state
    }

    /// Check if currently in speech
    pub fn is_speech(&self) -> bool {
        matches!(self.state, VadState::Speech | VadState::MaybeSilence)
    }

    /// Get configuration
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Calculate RMS (root mean square) of audio samples
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
        (sum_squares / samples.len() as f32).sqrt()
    }

    /// Check if audio frame is above speech threshold
    pub fn frame_is_speech(&self, samples: &[f32]) -> bool {
        Self::calculate_rms(samples) > self.config.speech_threshold
    }

    /// Process an audio frame and return whether state changed
    /// Returns (current_state, state_changed)
    pub fn process_frame(&mut self, samples: &[f32]) -> (VadState, bool) {
        let frame_speech = self.frame_is_speech(samples);
        let old_state = self.state;

        if frame_speech {
            self.consecutive_speech += 1;
            self.consecutive_silence = 0;
        } else {
            self.consecutive_silence += 1;
            self.consecutive_speech = 0;
        }

        // State machine transitions
        match self.state {
            VadState::Silence => {
                if self.consecutive_speech >= self.config.speech_start_frames {
                    debug!("VAD: Silence -> MaybeSpeech");
                    self.state = VadState::MaybeSpeech;
                }
            }
            VadState::MaybeSpeech => {
                if self.consecutive_silence > 0 {
                    debug!("VAD: MaybeSpeech -> Silence (false start)");
                    self.state = VadState::Silence;
                } else if self.consecutive_speech >= self.config.speech_start_frames + 1 {
                    debug!("VAD: MaybeSpeech -> Speech");
                    self.state = VadState::Speech;
                    self.speech_frame_count = self.consecutive_speech;
                }
            }
            VadState::Speech => {
                self.speech_frame_count += 1;
                if self.consecutive_silence >= 1 {
                    debug!("VAD: Speech -> MaybeSilence");
                    self.state = VadState::MaybeSilence;
                }
            }
            VadState::MaybeSilence => {
                if self.consecutive_speech > 0 {
                    debug!("VAD: MaybeSilence -> Speech (speech resumed)");
                    self.state = VadState::Speech;
                } else if self.consecutive_silence >= self.config.speech_end_frames {
                    debug!("VAD: MaybeSilence -> Silence (speech ended)");
                    self.state = VadState::Silence;
                    self.speech_frame_count = 0;
                }
            }
        }

        (self.state, self.state != old_state)
    }

    /// Reset VAD state to silence
    pub fn reset(&mut self) {
        self.state = VadState::Silence;
        self.consecutive_speech = 0;
        self.consecutive_silence = 0;
        self.speech_frame_count = 0;
    }
}

impl Default for Vad {
    fn default() -> Self {
        Self::new()
    }
}

/// Event types emitted by VAD processor
#[derive(Debug, Clone)]
pub enum VadEvent {
    /// Speech started
    SpeechStart,
    /// Speech ended, contains the speech samples
    SpeechEnd { samples: Vec<f32> },
    /// Currently in speech (for continuous updates)
    Speaking,
    /// Currently silent
    Silent,
}

/// VAD processor that accumulates speech segments
pub struct VadProcessor {
    vad: Vad,
    buffer: Vec<f32>,
    was_speech: bool,
}

impl VadProcessor {
    /// Create a new VAD processor
    pub fn new() -> Self {
        Self::with_vad(Vad::new())
    }

    /// Create a new VAD processor with custom VAD
    pub fn with_vad(vad: Vad) -> Self {
        Self {
            vad,
            buffer: Vec::new(),
            was_speech: false,
        }
    }

    /// Create a new VAD processor with custom threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self::with_vad(Vad::with_threshold(threshold))
    }

    /// Process audio frame and return events
    pub fn process(&mut self, samples: &[f32]) -> Vec<VadEvent> {
        let mut events = Vec::new();
        let (_state, _changed) = self.vad.process_frame(samples);
        let is_speech = self.vad.is_speech();

        // Detect transitions
        if is_speech && !self.was_speech {
            events.push(VadEvent::SpeechStart);
        }

        if is_speech {
            self.buffer.extend_from_slice(samples);
            events.push(VadEvent::Speaking);
        } else {
            if self.was_speech {
                // Speech just ended
                if !self.buffer.is_empty() {
                    events.push(VadEvent::SpeechEnd {
                        samples: std::mem::take(&mut self.buffer),
                    });
                }
            }
            events.push(VadEvent::Silent);
        }

        self.was_speech = is_speech;
        events
    }

    /// Get the current speech buffer (if any)
    pub fn current_buffer(&self) -> &[f32] {
        &self.buffer
    }

    /// Check if currently in speech
    pub fn is_speech(&self) -> bool {
        self.vad.is_speech()
    }

    /// Get the underlying VAD
    pub fn vad(&self) -> &Vad {
        &self.vad
    }

    /// Reset the processor
    pub fn reset(&mut self) {
        self.vad.reset();
        self.buffer.clear();
        self.was_speech = false;
    }
}

impl Default for VadProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_calculation() {
        // Silent audio
        let silent: Vec<f32> = vec![0.0; 100];
        assert_eq!(Vad::calculate_rms(&silent), 0.0);

        // Constant value
        let constant: Vec<f32> = vec![0.5; 100];
        assert!((Vad::calculate_rms(&constant) - 0.5).abs() < 0.001);

        // Sine wave should have RMS of ~0.707 * amplitude
        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * i as f32 / 100.0).sin())
            .collect();
        let rms = Vad::calculate_rms(&samples);
        assert!((rms - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_vad_state_transitions() {
        let mut vad = Vad::with_config(VadConfig {
            speech_threshold: 0.1,
            speech_start_frames: 2,
            speech_end_frames: 3,
            min_speech_frames: 1,
        });

        // Start silent
        assert_eq!(vad.state(), VadState::Silence);

        // First loud frame
        let loud: Vec<f32> = vec![0.5; 100];
        let silent: Vec<f32> = vec![0.001; 100];

        vad.process_frame(&loud);
        // Need more frames for speech start
        assert!(matches!(vad.state(), VadState::Silence | VadState::MaybeSpeech));

        // Confirm speech with more loud frames
        vad.process_frame(&loud);
        vad.process_frame(&loud);
        vad.process_frame(&loud);
        assert!(vad.is_speech());

        // Start silence detection
        vad.process_frame(&silent);
        assert!(matches!(vad.state(), VadState::MaybeSilence));

        // Confirm silence
        vad.process_frame(&silent);
        vad.process_frame(&silent);
        vad.process_frame(&silent);
        assert_eq!(vad.state(), VadState::Silence);
    }

    #[test]
    fn test_vad_processor_events() {
        let mut processor = VadProcessor::with_threshold(0.1);

        let loud: Vec<f32> = vec![0.5; 100];
        let silent: Vec<f32> = vec![0.001; 100];

        // Process silent frames
        let events = processor.process(&silent);
        assert!(events.iter().any(|e| matches!(e, VadEvent::Silent)));

        // Process enough loud frames to trigger speech
        for _ in 0..5 {
            processor.process(&loud);
        }
        assert!(processor.is_speech());

        // Process silent frames until speech ends
        for _ in 0..20 {
            let events = processor.process(&silent);
            if events.iter().any(|e| matches!(e, VadEvent::SpeechEnd { .. })) {
                // Got speech end event
                return;
            }
        }
    }
}
