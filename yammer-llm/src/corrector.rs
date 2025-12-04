//! LLM text correction using llama.cpp

use std::path::Path;
use std::time::Instant;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use llama_cpp::standard_sampler::StandardSampler;
use thiserror::Error;
use tracing::{debug, info};

/// Errors that can occur during text correction
#[derive(Error, Debug)]
pub enum CorrectorError {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Session creation failed: {0}")]
    SessionCreate(String),

    #[error("Generation failed: {0}")]
    Generation(String),
}

/// Result type for corrector operations
pub type CorrectorResult<T> = Result<T, CorrectorError>;

/// Configuration for the corrector
#[derive(Debug, Clone)]
pub struct CorrectorConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = deterministic)
    pub temperature: f32,
    /// Context size
    pub context_size: u32,
}

impl Default for CorrectorConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.1, // Low temperature for consistent corrections
            context_size: 2048,
        }
    }
}

/// Correction result with timing information
#[derive(Debug, Clone)]
pub struct CorrectionResult {
    /// The corrected text
    pub text: String,
    /// Time taken for correction in milliseconds
    pub latency_ms: u64,
}

/// The prompt template for text correction
const CORRECTION_PROMPT: &str = r#"Fix any transcription errors in the following dictation.
Only fix obvious mistakes, don't rephrase.
Add punctuation and capitalization.

Input: "#;

const CORRECTION_SUFFIX: &str = "\nOutput:";

/// LLM-based text corrector using llama.cpp
pub struct Corrector {
    model: LlamaModel,
    config: CorrectorConfig,
}

impl Corrector {
    /// Create a new corrector with the given model path
    pub fn new(model_path: &Path) -> CorrectorResult<Self> {
        Self::with_config(model_path, CorrectorConfig::default())
    }

    /// Create a new corrector with custom configuration
    pub fn with_config(model_path: &Path, config: CorrectorConfig) -> CorrectorResult<Self> {
        info!("Loading LLM model from {:?}", model_path);

        let params = LlamaParams::default();

        let model = LlamaModel::load_from_file(model_path, params)
            .map_err(|e| CorrectorError::ModelLoad(e.to_string()))?;

        info!("LLM model loaded successfully");
        Ok(Self { model, config })
    }

    /// Correct/improve transcribed text
    pub fn correct(&self, text: &str) -> CorrectorResult<CorrectionResult> {
        let start = Instant::now();

        // Build the prompt
        let prompt = format!("{}{}{}", CORRECTION_PROMPT, text, CORRECTION_SUFFIX);
        debug!("Correction prompt: {}", prompt);

        // Create a session for this correction with configured context size
        let mut session_params = SessionParams::default();
        session_params.n_ctx = self.config.context_size;

        let mut session = self.model.create_session(session_params)
            .map_err(|e| CorrectorError::SessionCreate(e.to_string()))?;

        // Feed the prompt
        session.advance_context(&prompt)
            .map_err(|e| CorrectorError::Generation(e.to_string()))?;

        // Generate the correction with configured temperature
        use llama_cpp::standard_sampler::SamplerStage;
        let sampler = StandardSampler::new_softmax(
            vec![
                SamplerStage::Temperature(self.config.temperature),
                SamplerStage::TopP(0.95),
                SamplerStage::MinP(0.05),
            ],
            1 // min_keep
        );

        let completions = session
            .start_completing_with(sampler, self.config.max_tokens)
            .map_err(|e| CorrectorError::Generation(e.to_string()))?;

        // Collect tokens into text
        let mut result = String::new();
        for token in completions.into_strings() {
            // Stop at newline (end of corrected output)
            if token.contains('\n') {
                let before_newline = token.split('\n').next().unwrap_or("");
                result.push_str(before_newline);
                break;
            }
            result.push_str(&token);
        }

        let latency_ms = start.elapsed().as_millis() as u64;
        debug!("Correction completed in {}ms: '{}' -> '{}'", latency_ms, text, result.trim());

        Ok(CorrectionResult {
            text: result.trim().to_string(),
            latency_ms,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &CorrectorConfig {
        &self.config
    }
}
