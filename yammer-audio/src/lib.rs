//! yammer-audio: audio capture, resampling, and voice activity detection
//!
//! This crate handles all audio input processing including:
//! - Audio capture via cpal
//! - Resampling to whisper-compatible format
//! - Voice activity detection (VAD)

pub mod capture;
pub mod vad;

pub use capture::AudioCapture;
