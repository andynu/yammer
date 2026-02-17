//! Audio resampling for Whisper compatibility (16kHz mono)

use rubato::{FftFixedIn, Resampler};
use tracing::debug;

/// Target sample rate for Whisper
pub const WHISPER_SAMPLE_RATE: u32 = 16000;

/// Audio resampler for converting to Whisper-compatible format
pub struct AudioResampler {
    resampler: FftFixedIn<f64>,
    input_rate: u32,
    output_rate: u32,
}

impl AudioResampler {
    /// Create a new resampler from source to target sample rate
    pub fn new(input_rate: u32, output_rate: u32) -> Result<Self, rubato::ResamplerConstructionError> {
        let ratio = output_rate as f64 / input_rate as f64;
        // Use chunk size that works well with both common rates
        let chunk_size = 1024;

        debug!(
            "Creating resampler: {} Hz -> {} Hz (ratio: {:.4})",
            input_rate, output_rate, ratio
        );

        let resampler = FftFixedIn::<f64>::new(
            input_rate as usize,
            output_rate as usize,
            chunk_size,
            2, // sub_chunks for better latency
            1, // mono channel
        )?;

        Ok(Self {
            resampler,
            input_rate,
            output_rate,
        })
    }

    /// Create a resampler configured for Whisper (16kHz output)
    pub fn for_whisper(input_rate: u32) -> Result<Self, rubato::ResamplerConstructionError> {
        Self::new(input_rate, WHISPER_SAMPLE_RATE)
    }

    /// Get the input sample rate
    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// Get the output sample rate
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    /// Get the number of input frames needed for next processing call
    pub fn input_frames_next(&self) -> usize {
        self.resampler.input_frames_next()
    }

    /// Resample audio samples
    /// Input: mono f32 samples at input_rate
    /// Output: mono f32 samples at output_rate
    pub fn resample(&mut self, input: &[f32]) -> Result<Vec<f32>, rubato::ResampleError> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Convert f32 to f64 for rubato
        let input_f64: Vec<f64> = input.iter().map(|&s| s as f64).collect();
        let input_frames = vec![input_f64];

        // Resample
        let output_frames = self.resampler.process(&input_frames, None)?;

        // Convert back to f32
        let output: Vec<f32> = output_frames[0].iter().map(|&s| s as f32).collect();

        Ok(output)
    }

    /// Process remaining samples and flush resampler
    pub fn flush(&mut self) -> Result<Vec<f32>, rubato::ResampleError> {
        // Create empty input for flushing
        let input_frames_needed = self.resampler.input_frames_next();
        let empty_input = vec![vec![0.0f64; input_frames_needed]];

        let output_frames = self.resampler.process(&empty_input, None)?;

        let output: Vec<f32> = output_frames[0].iter().map(|&s| s as f32).collect();

        Ok(output)
    }

    /// Convenience method to resample entire buffer at once
    pub fn resample_buffer(&mut self, input: &[f32]) -> Result<Vec<f32>, rubato::ResampleError> {
        let mut output = Vec::new();
        let chunk_size = self.resampler.input_frames_next();

        // Process in chunks
        for chunk in input.chunks(chunk_size) {
            // Pad last chunk if needed
            let padded: Vec<f32> = if chunk.len() < chunk_size {
                let mut padded = chunk.to_vec();
                padded.resize(chunk_size, 0.0);
                padded
            } else {
                chunk.to_vec()
            };

            let resampled = self.resample(&padded)?;
            output.extend(resampled);
        }

        Ok(output)
    }
}

/// Resample a complete buffer of audio in one call
/// Convenience function for simple use cases
pub fn resample_to_whisper(samples: &[f32], input_rate: u32) -> Result<Vec<f32>, String> {
    if input_rate == WHISPER_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let mut resampler = AudioResampler::for_whisper(input_rate)
        .map_err(|e| format!("Failed to create resampler: {}", e))?;

    resampler
        .resample_buffer(samples)
        .map_err(|e| format!("Resampling failed: {}", e))
}

/// Target sample rate for Kyutai STT (moshi/mimi codec)
pub const KYUTAI_SAMPLE_RATE: u32 = 24_000;

/// Resample a complete buffer of audio to Kyutai's 24kHz format
pub fn resample_to_kyutai(samples: &[f32], input_rate: u32) -> Result<Vec<f32>, String> {
    if input_rate == KYUTAI_SAMPLE_RATE {
        return Ok(samples.to_vec());
    }

    let mut resampler = AudioResampler::new(input_rate, KYUTAI_SAMPLE_RATE)
        .map_err(|e| format!("Failed to create resampler: {}", e))?;

    resampler
        .resample_buffer(samples)
        .map_err(|e| format!("Resampling failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_48k_to_16k() {
        // Create a simple sine wave at 48kHz
        let input_rate = 48000;
        let duration_secs = 1.0;
        let samples = (input_rate as f32 * duration_secs) as usize;

        let input: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f32 / input_rate as f32;
                (2.0 * std::f32::consts::PI * 440.0 * t).sin()
            })
            .collect();

        let output = resample_to_whisper(&input, input_rate).unwrap();

        // Output should be approximately 1/3 the size (16k/48k)
        let expected_samples = (WHISPER_SAMPLE_RATE as f32 * duration_secs) as usize;
        assert!(
            (output.len() as i32 - expected_samples as i32).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_samples,
            output.len()
        );
    }

    #[test]
    fn test_resample_44100_to_16k() {
        let input_rate = 44100;
        let duration_secs = 0.5;
        let samples = (input_rate as f32 * duration_secs) as usize;

        let input: Vec<f32> = (0..samples)
            .map(|i| {
                let t = i as f32 / input_rate as f32;
                (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
            })
            .collect();

        let output = resample_to_whisper(&input, input_rate).unwrap();

        let expected_samples = (WHISPER_SAMPLE_RATE as f32 * duration_secs) as usize;
        assert!(
            (output.len() as i32 - expected_samples as i32).abs() < 100,
            "Expected ~{} samples, got {}",
            expected_samples,
            output.len()
        );
    }

    #[test]
    fn test_no_resample_16k() {
        let input: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let output = resample_to_whisper(&input, 16000).unwrap();
        assert_eq!(input, output);
    }
}
