//! yammer-audio: audio capture, resampling, and voice activity detection
//!
//! This crate handles all audio input processing including:
//! - Audio capture via cpal
//! - Resampling to STT-compatible formats
//! - Voice activity detection (VAD)

pub mod capture;
pub mod resample;
pub mod vad;

pub use capture::{
    AudioCapture, AudioChunk, AudioError, AudioResult, CaptureHandle, InputDeviceInfo, StreamConfigInfo, write_wav,
};
pub use resample::{
    resample_to_kyutai, AudioResampler, KYUTAI_SAMPLE_RATE,
    resample_to_whisper, WHISPER_SAMPLE_RATE,
};
pub use vad::{Vad, VadConfig, VadEvent, VadProcessor, VadState, DEFAULT_SPEECH_THRESHOLD};
