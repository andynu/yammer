//! yammer-stt: speech-to-text using whisper
//!
//! This crate provides whisper-rs integration for transcribing audio.

pub mod transcriber;

pub use transcriber::{
    load_wav_16k, TranscribeError, TranscribeResult, Transcriber, Transcript, TranscriptSegment,
    WHISPER_SAMPLE_RATE,
};
