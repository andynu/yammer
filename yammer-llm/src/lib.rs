//! yammer-llm: LLM integration for text correction
//!
//! This crate provides llama.cpp integration for post-processing
//! transcribed text to fix grammar, punctuation, and context.

pub mod corrector;

pub use corrector::{CorrectionResult, Corrector, CorrectorConfig, CorrectorError, CorrectorResult};
