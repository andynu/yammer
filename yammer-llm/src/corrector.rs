//! LLM text correction using llama.cpp

use std::path::Path;
use std::time::Instant;
use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
use llama_cpp::standard_sampler::StandardSampler;
use thiserror::Error;
use tracing::{debug, info, warn};

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
/// Uses few-shot examples to guide the model on expected output format
const CORRECTION_PROMPT: &str = r#"Fix transcription errors. Add punctuation (periods, commas, apostrophes). Fix capitalization. Split words that are stuck together. Don't rephrase.

Input: im going to the store do you want anything
Output: I'm going to the store. Do you want anything?

Input: its raining outside we should bring an umbrella
Output: It's raining outside. We should bring an umbrella.

Input: Ihope multiple labswill contribute Ihave noidea howshared itwill be
Output: I hope multiple labs will contribute. I have no idea how shared it will be.

Input: "#;

const CORRECTION_SUFFIX: &str = "\nOutput:";

/// The placeholder for user text in custom prompts
const TEXT_PLACEHOLDER: &str = "{text}";

/// LLM-based text corrector using llama.cpp
pub struct Corrector {
    model: LlamaModel,
    config: CorrectorConfig,
}

/// Build a prompt from a custom template and input text.
/// If the template contains {text}, replace it with the input.
/// Otherwise, append the input at the end.
fn build_custom_prompt(template: &str, text: &str) -> String {
    if template.contains(TEXT_PLACEHOLDER) {
        template.replace(TEXT_PLACEHOLDER, text)
    } else {
        format!("{}{}", template, text)
    }
}

/// Edits allowed between input and output regardless of length, so short
/// inputs can still gain a word ending ("ok" -> "Okay.").
const GUARD_MIN_EDITS: usize = 3;

/// Fraction of the input's letters that the correction may change.
const GUARD_MAX_EDIT_RATIO: f64 = 0.3;

/// Lowercase alphanumerics only, so punctuation, capitalization and word
/// splitting cost nothing and only changes to the words themselves count.
fn letters(text: &str) -> String {
    text.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

/// Return the correction, or the raw input when the model produced nothing or
/// changed the words more than a correction should (a summary, a rephrase, an
/// invented continuation).
fn guard_correction<'a>(input: &'a str, output: &'a str) -> &'a str {
    if output.trim().is_empty() {
        warn!("LLM correction was empty, keeping raw transcript");
        return input;
    }

    let before = letters(input);
    let after = letters(output);
    let distance = strsim::levenshtein(&before, &after);
    let allowed = GUARD_MIN_EDITS
        .max((before.chars().count() as f64 * GUARD_MAX_EDIT_RATIO) as usize);

    if distance > allowed {
        warn!(
            "LLM correction changed {} of {} letters (limit {}), keeping raw transcript: '{}'",
            distance,
            before.chars().count(),
            allowed,
            output
        );
        return input;
    }
    output
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

    /// Correct/improve transcribed text using the built-in default prompt
    pub fn correct(&self, text: &str) -> CorrectorResult<CorrectionResult> {
        self.correct_with_prompt(text, None)
    }

    /// Correct/improve transcribed text with an optional custom prompt template
    pub fn correct_with_prompt(
        &self,
        text: &str,
        custom_prompt: Option<&str>,
    ) -> CorrectorResult<CorrectionResult> {
        let start = Instant::now();

        // Build the prompt - use custom if provided, otherwise use built-in default
        let prompt = match custom_prompt {
            Some(template) => build_custom_prompt(template, text),
            None => format!("{}{}{}", CORRECTION_PROMPT, text, CORRECTION_SUFFIX),
        };
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

        let output = result.trim();
        let corrected = match custom_prompt {
            // A custom prompt may legitimately ask for a rewrite, so only an
            // empty result is rejected there.
            Some(_) if !output.is_empty() => output,
            _ => guard_correction(text, output),
        };

        Ok(CorrectionResult {
            text: corrected.to_string(),
            latency_ms,
        })
    }

    /// Get the configuration
    pub fn config(&self) -> &CorrectorConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_custom_prompt_with_placeholder() {
        let template = "Fix this text: {text}\nResult:";
        let text = "hello world";
        let result = build_custom_prompt(template, text);
        assert_eq!(result, "Fix this text: hello world\nResult:");
    }

    #[test]
    fn test_build_custom_prompt_without_placeholder() {
        let template = "Fix this text: ";
        let text = "hello world";
        let result = build_custom_prompt(template, text);
        assert_eq!(result, "Fix this text: hello world");
    }

    #[test]
    fn test_guard_accepts_punctuation_and_capitalization() {
        let input = "im going to the store do you want anything";
        let output = "I'm going to the store. Do you want anything?";
        assert_eq!(guard_correction(input, output), output);
    }

    #[test]
    fn test_guard_accepts_split_words() {
        let input = "Ihope multiple labswill contribute Ihave noidea howshared itwill be";
        let output = "I hope multiple labs will contribute. I have no idea how shared it will be.";
        assert_eq!(guard_correction(input, output), output);
    }

    #[test]
    fn test_guard_accepts_homophone_fix() {
        let input = "their going to be late for the meeting";
        let output = "They're going to be late for the meeting.";
        assert_eq!(guard_correction(input, output), output);
    }

    #[test]
    fn test_guard_accepts_small_change_to_short_input() {
        assert_eq!(guard_correction("ok", "Okay."), "Okay.");
    }

    #[test]
    fn test_guard_rejects_empty_output() {
        assert_eq!(guard_correction("hello there", ""), "hello there");
        assert_eq!(guard_correction("ok", "  "), "ok");
    }

    #[test]
    fn test_guard_rejects_summary() {
        let input = "so i was thinking we could move the standup to ten because half the team is in another timezone now";
        let output = "The speaker suggests moving the meeting.";
        assert_eq!(guard_correction(input, output), input);
    }

    #[test]
    fn test_guard_rejects_invented_continuation() {
        let input = "send me the report when its done";
        let output = "Send me the report when it's done. Also, remember to book the flights for the conference next week.";
        assert_eq!(guard_correction(input, output), input);
    }

    #[test]
    fn test_build_custom_prompt_multiple_placeholders() {
        // If there are multiple {text} placeholders, all get replaced
        let template = "{text} -- {text}";
        let text = "hello";
        let result = build_custom_prompt(template, text);
        assert_eq!(result, "hello -- hello");
    }
}
