//! Apple SpeechAnalyzer STT backend (macOS 26 Tahoe+)
//!
//! Wraps the Swift SpeechBridge via C FFI. The Swift side creates a
//! `SpeechAnalyzer` + `SpeechTranscriber`, feeds audio as
//! `AVAudioPCMBuffer`, and drains results into a thread-safe queue
//! that Rust polls via `apple_stt_poll_results`.
//!
//! ## Exit handling
//!
//! SpeechAnalyzer spawns tasks on Swift's cooperative thread pool that
//! cannot be cleanly cancelled from C. When the process exits, the
//! Swift runtime's cleanup triggers EXC_BREAKPOINT (__builtin_trap).
//! We work around this by installing a SIGTRAP handler early and
//! using _exit() in Drop to skip runtime cleanup.

use anyhow::{Result, bail};
use tracing::{debug, info};

use crate::SttBackend;

// C FFI declarations — implemented in swift/SpeechBridge.swift
unsafe extern "C" {
    fn apple_stt_init(locale: *const std::ffi::c_char) -> i32;
    fn apple_stt_get_format(rate: *mut u32, channels: *mut u32) -> i32;
    fn apple_stt_feed_pcm(samples: *const f32, count: u32) -> i32;
    fn apple_stt_poll_results(buf: *mut u8, len: u32) -> i32;
    fn apple_stt_stop() -> i32;
}

/// Install a SIGTRAP handler that calls _exit(0). This prevents
/// EXC_BREAKPOINT from Swift's cooperative pool cleanup from crashing
/// the process on exit. Only installed once.
fn install_exit_trap_handler() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        unsafe extern "C" fn handle_trap(_sig: libc::c_int) {
            // The Swift runtime is tearing down — just exit cleanly.
            unsafe { libc::_exit(0) };
        }
        unsafe {
            libc::signal(libc::SIGTRAP, handle_trap as libc::sighandler_t);
        }
    });
}

/// Apple SpeechAnalyzer-backed streaming transcriber.
pub struct AppleSttTranscriber {
    sample_rate: u32,
    chunk_samples: usize,
}

impl AppleSttTranscriber {
    /// Initialise the SpeechAnalyzer pipeline.
    ///
    /// `model_repo` is accepted for API compatibility but ignored on macOS
    /// (the system model is used).
    pub fn new(_model_repo: &str) -> Result<Self> {
        info!("Initializing Apple SpeechAnalyzer STT backend");

        install_exit_trap_handler();

        let locale = std::ffi::CString::new("en-US").unwrap();
        let rc = unsafe { apple_stt_init(locale.as_ptr()) };
        if rc != 0 {
            bail!("apple_stt_init failed (rc={})", rc);
        }

        let mut rate: u32 = 0;
        let mut channels: u32 = 0;
        let rc = unsafe { apple_stt_get_format(&mut rate, &mut channels) };
        if rc != 0 {
            bail!("apple_stt_get_format failed (rc={})", rc);
        }

        // Use 80ms chunks to match the Kyutai cadence
        let chunk_samples = (rate as usize * 80) / 1000;
        info!(
            "Apple STT ready: {}Hz, {} channels, chunk={}",
            rate, channels, chunk_samples
        );

        Ok(Self {
            sample_rate: rate,
            chunk_samples,
        })
    }

    fn poll_text(&self) -> Option<String> {
        let mut buf = vec![0u8; 4096];
        let n = unsafe { apple_stt_poll_results(buf.as_mut_ptr(), buf.len() as u32) };
        if n <= 0 {
            return None;
        }
        let text = String::from_utf8_lossy(&buf[..n as usize]).to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    }
}

impl Drop for AppleSttTranscriber {
    fn drop(&mut self) {
        unsafe {
            apple_stt_stop();
        }
    }
}

impl SttBackend for AppleSttTranscriber {
    fn step(&mut self, pcm: &[f32]) -> Result<Vec<String>> {
        let rc = unsafe { apple_stt_feed_pcm(pcm.as_ptr(), pcm.len() as u32) };
        if rc != 0 {
            bail!("apple_stt_feed_pcm failed (rc={})", rc);
        }

        let mut words = Vec::new();
        if let Some(text) = self.poll_text() {
            debug!("Apple STT: {:?}", text);
            // SpeechAnalyzer returns full segments; split into words
            for w in text.split_whitespace() {
                words.push(w.to_string());
            }
        }
        Ok(words)
    }

    fn transcribe_buffer(&mut self, pcm: &[f32]) -> Result<String> {
        let chunk = self.chunk_samples();
        let mut all_text = String::new();

        let mut offset = 0;
        while offset + chunk <= pcm.len() {
            let slice = &pcm[offset..offset + chunk];
            let rc = unsafe { apple_stt_feed_pcm(slice.as_ptr(), slice.len() as u32) };
            if rc != 0 {
                bail!("apple_stt_feed_pcm failed (rc={})", rc);
            }
            offset += chunk;
        }

        // Flush remainder
        if offset < pcm.len() {
            let mut padded = pcm[offset..].to_vec();
            padded.resize(chunk, 0.0);
            let rc = unsafe { apple_stt_feed_pcm(padded.as_ptr(), padded.len() as u32) };
            if rc != 0 {
                bail!("apple_stt_feed_pcm failed (rc={})", rc);
            }
        }

        // Give the analyzer a moment to process
        std::thread::sleep(std::time::Duration::from_millis(200));

        // Drain any accumulated text
        while let Some(text) = self.poll_text() {
            if !all_text.is_empty() {
                all_text.push(' ');
            }
            all_text.push_str(&text);
        }

        // Signal end of this batch — finishes the input continuation,
        // but doesn't deallocate (which would crash the cooperative pool).
        let rc = unsafe { apple_stt_stop() };
        if rc != 0 {
            bail!("apple_stt_stop failed (rc={})", rc);
        }

        // Re-initialise for next use
        let locale = std::ffi::CString::new("en-US").unwrap();
        let rc = unsafe { apple_stt_init(locale.as_ptr()) };
        if rc != 0 {
            bail!("apple_stt_init (re-init) failed (rc={})", rc);
        }

        Ok(all_text)
    }

    fn chunk_samples(&self) -> usize {
        self.chunk_samples
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
