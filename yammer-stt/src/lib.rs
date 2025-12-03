//! yammer-stt: speech-to-text using whisper
//!
//! This crate provides whisper-rs integration for transcribing audio.

pub mod transcriber;

pub use transcriber::Transcriber;
