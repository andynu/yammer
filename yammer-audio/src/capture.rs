//! Audio capture functionality using cpal

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, Host, SampleFormat, Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

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
