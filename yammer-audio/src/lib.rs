//! yammer-audio: audio capture, resampling, and voice activity detection
//!
//! This crate handles all audio input processing including:
//! - Audio capture via cpal
//! - Resampling to whisper-compatible format
//! - Voice activity detection (VAD)

pub mod capture;
pub mod resample;
pub mod vad;

pub use capture::{
    AudioCapture, AudioChunk, AudioError, AudioResult, InputDeviceInfo, StreamConfigInfo, write_wav,
};
pub use resample::{resample_to_whisper, AudioResampler, WHISPER_SAMPLE_RATE};
