//! WAV file loading and resampling utilities shared across STT backends.

use anyhow::{Context, Result};
use rubato::{FftFixedIn, Resampler};

/// Load a WAV file and resample it to the given target sample rate (mono f32).
pub fn load_wav_resampled(path: &std::path::Path, target_rate: u32) -> Result<Vec<f32>> {
    let reader = hound::WavReader::open(path)
        .with_context(|| format!("Failed to open WAV: {}", path.display()))?;
    let spec = reader.spec();

    // Decode to f32 mono
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => {
            reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect()
        }
    };

    // Mix to mono if needed
    let mono: Vec<f32> = if spec.channels > 1 {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    // Resample if needed
    if spec.sample_rate == target_rate {
        return Ok(mono);
    }

    let chunk_size = 1024usize;
    let mut resampler = FftFixedIn::<f64>::new(
        spec.sample_rate as usize,
        target_rate as usize,
        chunk_size,
        2,
        1,
    )
    .context("Failed to create resampler")?;

    let mut output = Vec::new();
    for chunk in mono.chunks(chunk_size) {
        let mut padded = chunk.to_vec();
        padded.resize(chunk_size, 0.0);
        let input_f64: Vec<f64> = padded.iter().map(|&s| s as f64).collect();
        let out = resampler
            .process(&[input_f64], None)
            .context("Resampling failed")?;
        output.extend(out[0].iter().map(|&s| s as f32));
    }

    Ok(output)
}
