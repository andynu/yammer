//! Audio capture functionality using cpal

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Temporarily suppress stderr to hide ALSA/JACK probe noise.
/// Returns a guard that restores stderr when dropped.
#[cfg(target_os = "linux")]
struct StderrSuppressor {
    saved_fd: i32,
}

#[cfg(target_os = "linux")]
impl StderrSuppressor {
    fn new() -> Option<Self> {
        unsafe {
            let saved_fd = libc::dup(2);
            if saved_fd < 0 {
                return None;
            }
            let devnull = libc::open(b"/dev/null\0".as_ptr() as *const _, libc::O_WRONLY);
            if devnull < 0 {
                libc::close(saved_fd);
                return None;
            }
            libc::dup2(devnull, 2);
            libc::close(devnull);
            Some(Self { saved_fd })
        }
    }
}

#[cfg(target_os = "linux")]
impl Drop for StderrSuppressor {
    fn drop(&mut self) {
        unsafe {
            libc::dup2(self.saved_fd, 2);
            libc::close(self.saved_fd);
        }
    }
}

/// Audio capture errors
#[derive(Error, Debug)]
pub enum AudioError {
    #[error("No input device available")]
    NoInputDevice,

    #[error("Device error: {0}")]
    Device(String),

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Unsupported sample format: {0:?}")]
    UnsupportedFormat(SampleFormat),

    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for audio operations
pub type AudioResult<T> = Result<T, AudioError>;

/// Information about an audio input device
#[derive(Debug, Clone)]
pub struct InputDeviceInfo {
    /// Device name
    pub name: String,
    /// Whether this is the default device
    pub is_default: bool,
    /// Supported configurations
    pub configs: Vec<StreamConfigInfo>,
}

/// Information about a supported stream configuration
#[derive(Debug, Clone)]
pub struct StreamConfigInfo {
    pub channels: u16,
    pub min_sample_rate: u32,
    pub max_sample_rate: u32,
    pub sample_format: String,
}

/// Audio samples collected from capture
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Audio samples (mono, f32, -1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

/// Audio capture handle
pub struct AudioCapture {
    #[allow(dead_code)]
    host: Host,
    device: Device,
    config: StreamConfig,
    is_recording: Arc<AtomicBool>,
}

impl AudioCapture {
    /// Create a new audio capture instance with default device
    pub fn new() -> AudioResult<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;

        let device_name = device
            .name()
            .unwrap_or_else(|_| "Unknown".to_string());
        info!("Using audio input device: {}", device_name);

        // Get supported config
        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        debug!(
            "Default config: {} channels, {} Hz, {:?}",
            supported_config.channels(),
            supported_config.sample_rate().0,
            supported_config.sample_format()
        );

        let config: StreamConfig = supported_config.into();

        Ok(Self {
            host,
            device,
            config,
            is_recording: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Create with a specific device by name
    pub fn with_device(device_name: &str) -> AudioResult<Self> {
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .map_err(|e| AudioError::Device(e.to_string()))?
            .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
            .ok_or_else(|| AudioError::Device(format!("Device not found: {}", device_name)))?;

        let supported_config = device
            .default_input_config()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        let config: StreamConfig = supported_config.into();

        Ok(Self {
            host,
            device,
            config,
            is_recording: Arc::new(AtomicBool::new(false)),
        })
    }

    /// List available input devices
    pub fn list_devices() -> AudioResult<Vec<InputDeviceInfo>> {
        // Suppress ALSA/JACK probe noise on stderr
        #[cfg(target_os = "linux")]
        let _suppress = StderrSuppressor::new();

        let host = cpal::default_host();
        let default_name = host
            .default_input_device()
            .and_then(|d| d.name().ok());

        let mut devices = Vec::new();
        for device in host.input_devices().map_err(|e| AudioError::Device(e.to_string()))? {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            let is_default = default_name.as_ref().map(|d| d == &name).unwrap_or(false);

            let mut configs = Vec::new();
            if let Ok(supported_configs) = device.supported_input_configs() {
                for config in supported_configs {
                    configs.push(StreamConfigInfo {
                        channels: config.channels(),
                        min_sample_rate: config.min_sample_rate().0,
                        max_sample_rate: config.max_sample_rate().0,
                        sample_format: format!("{:?}", config.sample_format()),
                    });
                }
            }

            devices.push(InputDeviceInfo {
                name,
                is_default,
                configs,
            });
        }

        Ok(devices)
    }

    /// Get the sample rate being used
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate.0
    }

    /// Get the number of channels being captured
    pub fn channels(&self) -> u16 {
        self.config.channels
    }

    /// Record for a specific duration and return samples
    pub async fn record_duration(&self, duration: std::time::Duration) -> AudioResult<AudioChunk> {
        let (tx, mut rx) = mpsc::channel::<Vec<f32>>(100);

        let is_recording = self.is_recording.clone();
        is_recording.store(true, Ordering::SeqCst);

        let channels = self.config.channels;
        let sample_rate = self.config.sample_rate.0;

        // Create error callback
        let err_fn = |err| error!("Audio stream error: {}", err);

        // Get the sample format and build appropriate stream
        let supported_config = self
            .device
            .default_input_config()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => self.build_stream::<f32>(tx.clone(), channels, err_fn)?,
            SampleFormat::I16 => self.build_stream_i16(tx.clone(), channels, err_fn)?,
            SampleFormat::U16 => self.build_stream_u16(tx.clone(), channels, err_fn)?,
            format => return Err(AudioError::UnsupportedFormat(format)),
        };

        stream.play().map_err(|e| AudioError::Stream(e.to_string()))?;
        info!("Recording started");

        // Record for the specified duration
        tokio::time::sleep(duration).await;

        is_recording.store(false, Ordering::SeqCst);
        drop(stream);
        drop(tx);

        info!("Recording stopped, collecting samples");

        // Collect all samples
        let mut all_samples = Vec::new();
        while let Some(chunk) = rx.recv().await {
            all_samples.extend(chunk);
        }

        info!("Collected {} samples", all_samples.len());

        Ok(AudioChunk {
            samples: all_samples,
            sample_rate,
        })
    }

    fn build_stream<T>(
        &self,
        tx: mpsc::Sender<Vec<f32>>,
        channels: u16,
        err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    ) -> AudioResult<Stream>
    where
        T: cpal::SizedSample + cpal::FromSample<f32> + Into<f32>,
    {
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                // Convert to mono f32
                let mono: Vec<f32> = if channels == 1 {
                    data.iter().map(|s| (*s).into()).collect()
                } else {
                    // Average channels to mono
                    data.chunks(channels as usize)
                        .map(|frame| {
                            let sum: f32 = frame.iter().map(|s| (*s).into()).sum();
                            sum / channels as f32
                        })
                        .collect()
                };

                if let Err(e) = tx.try_send(mono) {
                    warn!("Failed to send audio samples: {}", e);
                }
            },
            err_fn,
            None,
        ).map_err(|e| AudioError::Stream(e.to_string()))?;

        Ok(stream)
    }

    fn build_stream_i16(
        &self,
        tx: mpsc::Sender<Vec<f32>>,
        channels: u16,
        err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    ) -> AudioResult<Stream> {
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if channels == 1 {
                    data.iter().map(|s| *s as f32 / i16::MAX as f32).collect()
                } else {
                    data.chunks(channels as usize)
                        .map(|frame| {
                            let sum: f32 = frame.iter().map(|s| *s as f32 / i16::MAX as f32).sum();
                            sum / channels as f32
                        })
                        .collect()
                };

                if let Err(e) = tx.try_send(mono) {
                    warn!("Failed to send audio samples: {}", e);
                }
            },
            err_fn,
            None,
        ).map_err(|e| AudioError::Stream(e.to_string()))?;

        Ok(stream)
    }

    fn build_stream_u16(
        &self,
        tx: mpsc::Sender<Vec<f32>>,
        channels: u16,
        err_fn: impl Fn(cpal::StreamError) + Send + 'static,
    ) -> AudioResult<Stream> {
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[u16], _: &cpal::InputCallbackInfo| {
                let mono: Vec<f32> = if channels == 1 {
                    data.iter()
                        .map(|s| (*s as f32 - 32768.0) / 32768.0)
                        .collect()
                } else {
                    data.chunks(channels as usize)
                        .map(|frame| {
                            let sum: f32 = frame.iter()
                                .map(|s| (*s as f32 - 32768.0) / 32768.0)
                                .sum();
                            sum / channels as f32
                        })
                        .collect()
                };

                if let Err(e) = tx.try_send(mono) {
                    warn!("Failed to send audio samples: {}", e);
                }
            },
            err_fn,
            None,
        ).map_err(|e| AudioError::Stream(e.to_string()))?;

        Ok(stream)
    }
}

/// Start continuous audio capture, returning a receiver for audio chunks
///
/// Each chunk contains roughly `chunk_ms` milliseconds of audio.
/// The stream continues until the returned `CaptureHandle` is dropped.
pub struct CaptureHandle {
    _stream: Stream,
    is_running: Arc<AtomicBool>,
}

impl Drop for CaptureHandle {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
    }
}

impl AudioCapture {
    /// Start continuous audio capture
    ///
    /// Returns a handle (keep alive to continue capture) and a receiver for audio chunks.
    /// Each chunk contains approximately `chunk_ms` of audio samples.
    pub fn start_capture(&self, chunk_ms: u32) -> AudioResult<(CaptureHandle, mpsc::Receiver<Vec<f32>>)> {
        let samples_per_chunk = (self.config.sample_rate.0 as f32 * chunk_ms as f32 / 1000.0) as usize;
        let (tx, rx) = mpsc::channel::<Vec<f32>>(100);

        // Shared state for callbacks - use Arc to share across closure and CaptureHandle
        let is_running = Arc::new(AtomicBool::new(true));
        let channels = self.config.channels;

        let err_fn = |err| error!("Audio stream error: {}", err);

        let supported_config = self
            .device
            .default_input_config()
            .map_err(|e| AudioError::Device(e.to_string()))?;

        // Buffer to accumulate samples (shared between callback and chunking logic)
        let buffer = Arc::new(std::sync::Mutex::new(Vec::with_capacity(samples_per_chunk * 2)));

        // Clone references for F32 callback (I16 branch will use originals via move)
        let buffer_f32 = buffer.clone();
        let tx_f32 = tx.clone();
        let is_running_f32 = is_running.clone();

        // Clone for CaptureHandle - shares the same underlying AtomicBool
        let is_running_handle = is_running.clone();

        // F32 callback - uses _f32 suffixed clones
        let callback_f32 = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            if !is_running_f32.load(Ordering::SeqCst) {
                return;
            }

            // Convert to mono
            let mono: Vec<f32> = if channels == 1 {
                data.to_vec()
            } else {
                data.chunks(channels as usize)
                    .map(|frame| {
                        let sum: f32 = frame.iter().sum();
                        sum / channels as f32
                    })
                    .collect()
            };

            let mut buf = buffer_f32.lock().unwrap();
            buf.extend(mono);

            // Send chunks when we have enough samples
            while buf.len() >= samples_per_chunk {
                let chunk: Vec<f32> = buf.drain(..samples_per_chunk).collect();
                if let Err(e) = tx_f32.try_send(chunk) {
                    warn!("Failed to send audio chunk: {}", e);
                }
            }
        };

        let stream = match supported_config.sample_format() {
            SampleFormat::F32 => {
                self.device.build_input_stream(
                    &self.config,
                    callback_f32,
                    err_fn,
                    None,
                ).map_err(|e| AudioError::Stream(e.to_string()))?
            }
            SampleFormat::I16 => {
                // I16 callback - moves the original buffer/tx/is_running (not used by F32 branch)
                let callback_i16 = move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_running.load(Ordering::SeqCst) {
                        return;
                    }

                    let mono: Vec<f32> = if channels == 1 {
                        data.iter().map(|s| *s as f32 / i16::MAX as f32).collect()
                    } else {
                        data.chunks(channels as usize)
                            .map(|frame| {
                                let sum: f32 = frame.iter().map(|s| *s as f32 / i16::MAX as f32).sum();
                                sum / channels as f32
                            })
                            .collect()
                    };

                    let mut buf = buffer.lock().unwrap();
                    buf.extend(mono);

                    while buf.len() >= samples_per_chunk {
                        let chunk: Vec<f32> = buf.drain(..samples_per_chunk).collect();
                        if let Err(e) = tx.try_send(chunk) {
                            warn!("Failed to send audio chunk: {}", e);
                        }
                    }
                };

                self.device.build_input_stream(
                    &self.config,
                    callback_i16,
                    err_fn,
                    None,
                ).map_err(|e| AudioError::Stream(e.to_string()))?
            }
            format => return Err(AudioError::UnsupportedFormat(format)),
        };

        stream.play().map_err(|e| AudioError::Stream(e.to_string()))?;
        info!("Continuous capture started");

        // Use is_running_handle which shares the same underlying AtomicBool as:
        // - is_running_f32 (used by F32 callback)
        // - is_running (moved into I16 callback)
        // Setting this flag to false will stop whichever callback is active.
        Ok((
            CaptureHandle {
                _stream: stream,
                is_running: is_running_handle,
            },
            rx,
        ))
    }
}

/// Write audio samples to WAV file
pub fn write_wav(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?;
    for &sample in samples {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
    }
    writer.finalize().map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_error_display() {
        let err = AudioError::NoInputDevice;
        assert!(err.to_string().contains("No input device"));

        let err = AudioError::Device("test device error".into());
        assert!(err.to_string().contains("Device error"));
        assert!(err.to_string().contains("test device error"));

        let err = AudioError::Stream("test stream error".into());
        assert!(err.to_string().contains("Stream error"));
        assert!(err.to_string().contains("test stream error"));

        let err = AudioError::Config("test config error".into());
        assert!(err.to_string().contains("Configuration error"));
        assert!(err.to_string().contains("test config error"));
    }

    #[test]
    fn test_input_device_info_debug_clone() {
        let info = InputDeviceInfo {
            name: "Test Device".to_string(),
            is_default: true,
            configs: vec![],
        };

        let cloned = info.clone();
        assert_eq!(info.name, cloned.name);
        assert_eq!(info.is_default, cloned.is_default);

        let debug_str = format!("{:?}", info);
        assert!(debug_str.contains("Test Device"));
    }

    #[test]
    fn test_stream_config_info_debug_clone() {
        let config = StreamConfigInfo {
            channels: 2,
            min_sample_rate: 44100,
            max_sample_rate: 48000,
            sample_format: "F32".to_string(),
        };

        let cloned = config.clone();
        assert_eq!(config.channels, cloned.channels);
        assert_eq!(config.min_sample_rate, cloned.min_sample_rate);
        assert_eq!(config.max_sample_rate, cloned.max_sample_rate);
        assert_eq!(config.sample_format, cloned.sample_format);

        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("F32"));
    }

    #[test]
    fn test_audio_chunk_debug_clone() {
        let chunk = AudioChunk {
            samples: vec![0.0, 0.5, -0.5],
            sample_rate: 16000,
        };

        let cloned = chunk.clone();
        assert_eq!(chunk.samples, cloned.samples);
        assert_eq!(chunk.sample_rate, cloned.sample_rate);

        let debug_str = format!("{:?}", chunk);
        assert!(debug_str.contains("16000"));
    }

    #[test]
    fn test_write_wav_roundtrip() {
        use tempfile::NamedTempFile;

        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 0.25, -0.25];
        let sample_rate = 16000;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Write WAV
        write_wav(path, &samples, sample_rate).expect("Failed to write WAV");

        // Verify file exists and has content
        let metadata = std::fs::metadata(path).expect("Failed to get file metadata");
        assert!(metadata.len() > 0);

        // Read back with hound
        let reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, sample_rate);
        assert_eq!(spec.channels, 1);

        let read_samples: Vec<i16> = reader.into_samples().filter_map(Result::ok).collect();
        assert_eq!(read_samples.len(), samples.len());
    }

    #[test]
    fn test_write_wav_empty_samples() {
        use tempfile::NamedTempFile;

        let samples: Vec<f32> = vec![];
        let sample_rate = 48000;

        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let path = temp_file.path();

        // Writing empty samples should succeed
        write_wav(path, &samples, sample_rate).expect("Failed to write empty WAV");

        // Read back
        let reader = hound::WavReader::open(path).expect("Failed to open WAV");
        let read_samples: Vec<i16> = reader.into_samples().filter_map(Result::ok).collect();
        assert!(read_samples.is_empty());
    }

    // Integration tests that require audio hardware
    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when audio hardware available
    fn test_list_devices_integration() {
        let result = AudioCapture::list_devices();
        println!("list_devices result: {:?}", result);
        // May return empty list on headless systems
        assert!(result.is_ok());
    }

    #[test]
    #[ignore] // Run with `cargo test -- --ignored` when audio hardware available
    fn test_audio_capture_new_integration() {
        let result = AudioCapture::new();
        match result {
            Ok(capture) => {
                println!("Sample rate: {}", capture.sample_rate());
                println!("Channels: {}", capture.channels());
                assert!(capture.sample_rate() > 0);
                assert!(capture.channels() > 0);
            }
            Err(AudioError::NoInputDevice) => {
                println!("No input device available (expected on headless systems)");
            }
            Err(e) => {
                println!("Unexpected error: {:?}", e);
            }
        }
    }
}
