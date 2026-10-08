//! Detects an input device that delivers pure digital silence.
//!
//! A muted microphone (hardware capture switch off, or muted in PipeWire)
//! produces samples that are exactly zero. A live microphone always carries
//! a noise floor, even in a quiet room, so a sustained run of exact zeros
//! means the mic is muted rather than that the user is quiet.

/// Tracks consecutive exact-zero samples across chunks.
pub struct SilentInputDetector {
    threshold_samples: usize,
    zero_run: usize,
}

impl SilentInputDetector {
    /// `seconds` of uninterrupted exact zeros at `sample_rate` counts as muted.
    pub fn new(sample_rate: u32, seconds: f32) -> Self {
        Self {
            threshold_samples: (sample_rate as f32 * seconds) as usize,
            zero_run: 0,
        }
    }

    /// Feed a chunk; returns true while the zero run has reached the threshold.
    pub fn feed(&mut self, chunk: &[f32]) -> bool {
        for &s in chunk {
            if s == 0.0 {
                self.zero_run += 1;
            } else {
                self.zero_run = 0;
            }
        }
        self.zero_run >= self.threshold_samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_zeros_for_the_threshold_are_detected() {
        let mut d = SilentInputDetector::new(1000, 1.0);
        assert!(!d.feed(&[0.0; 600]));
        assert!(d.feed(&[0.0; 400]));
    }

    #[test]
    fn quiet_noise_floor_is_not_detected() {
        let mut d = SilentInputDetector::new(1000, 1.0);
        let quiet: Vec<f32> = (0..3000).map(|i| if i % 2 == 0 { 1e-5 } else { -1e-5 }).collect();
        assert!(!d.feed(&quiet));
    }

    #[test]
    fn a_nonzero_sample_resets_the_run() {
        let mut d = SilentInputDetector::new(1000, 1.0);
        assert!(!d.feed(&[0.0; 900]));
        let mut chunk = vec![0.0; 200];
        chunk[50] = 0.01;
        assert!(!d.feed(&chunk));
        assert!(!d.feed(&[0.0; 800]));
        assert!(d.feed(&[0.0; 100]));
    }
}
