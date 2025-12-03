//! yammer-cli: headless CLI for testing building blocks
//!
//! This binary provides a command-line interface for testing
//! individual components of the yammer system.

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use yammer_audio::{AudioCapture, resample_to_whisper, write_wav, Vad, VadEvent, VadProcessor, WHISPER_SAMPLE_RATE};
use yammer_core::{
    format_bytes, get_default_models, get_model_registry, DownloadManager, ModelStatus, ModelType,
};
use yammer_stt::Transcriber;

#[derive(Parser)]
#[command(name = "yammer")]
#[command(about = "Linux dictation app - local speech-to-text with LLM correction")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Download required AI models
    DownloadModels {
        /// Only show what would be downloaded, don't actually download
        #[arg(long)]
        dry_run: bool,

        /// Download all available models (not just defaults)
        #[arg(long)]
        all: bool,

        /// Specific model ID to download
        #[arg(long)]
        model: Option<String>,
    },

    /// List available and downloaded models
    ListModels,

    /// List available audio input devices
    ListDevices,

    /// Record audio from microphone
    Record {
        /// Recording duration in seconds
        #[arg(long, default_value = "5")]
        duration: u64,

        /// Output WAV file path
        #[arg(long, short)]
        output: PathBuf,

        /// Audio device to use (default: system default)
        #[arg(long)]
        device: Option<String>,

        /// Resample to 16kHz for Whisper compatibility
        #[arg(long)]
        resample: bool,
    },

    /// Test voice activity detection in real-time
    VadTest {
        /// VAD threshold (RMS level, default 0.01)
        #[arg(long, default_value = "0.01")]
        threshold: f32,

        /// Duration to run in seconds (0 = until Ctrl+C)
        #[arg(long, default_value = "30")]
        duration: u64,

        /// Audio device to use
        #[arg(long)]
        device: Option<String>,
    },

    /// Transcribe a WAV file using Whisper
    Transcribe {
        /// Path to WAV file (must be 16kHz mono)
        file: PathBuf,

        /// Path to Whisper model file (auto-detected from downloaded models if not specified)
        #[arg(long)]
        model: Option<PathBuf>,

        /// Show timestamps
        #[arg(long, short)]
        timestamps: bool,
    },

    /// Live dictation: speak and see text appear in real-time
    Dictate {
        /// Path to Whisper model file (auto-detected from downloaded models if not specified)
        #[arg(long)]
        model: Option<PathBuf>,

        /// VAD threshold (RMS level, default 0.01)
        #[arg(long, default_value = "0.01")]
        threshold: f32,

        /// Audio device to use
        #[arg(long)]
        device: Option<String>,

        /// Duration to run in seconds (0 = until Ctrl+C)
        #[arg(long, default_value = "0")]
        duration: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::DownloadModels { dry_run, all, model }) => {
            download_models(dry_run, all, model).await?;
        }
        Some(Commands::ListModels) => {
            list_models().await?;
        }
        Some(Commands::ListDevices) => {
            list_devices()?;
        }
        Some(Commands::Record { duration, output, device, resample }) => {
            record_audio(duration, output, device, resample).await?;
        }
        Some(Commands::VadTest { threshold, duration, device }) => {
            vad_test(threshold, duration, device).await?;
        }
        Some(Commands::Transcribe { file, model, timestamps }) => {
            transcribe_file(file, model, timestamps).await?;
        }
        Some(Commands::Dictate { model, threshold, device, duration }) => {
            dictate(model, threshold, device, duration).await?;
        }
        None => {
            println!("yammer - Linux dictation app");
            println!("Run with --help for usage");
        }
    }

    Ok(())
}

async fn download_models(dry_run: bool, all: bool, specific_model: Option<String>) -> Result<()> {
    let registry = get_model_registry();
    let defaults = get_default_models();
    let manager = DownloadManager::new(DownloadManager::default_model_dir());

    // Determine which models to download
    let models_to_check: Vec<_> = if let Some(ref model_id) = specific_model {
        registry
            .iter()
            .filter(|m| m.id == *model_id)
            .collect()
    } else if all {
        registry.iter().collect()
    } else {
        registry
            .iter()
            .filter(|m| defaults.contains(&m.id.as_str()))
            .collect()
    };

    if models_to_check.is_empty() {
        if let Some(ref model_id) = specific_model {
            println!("Unknown model: {}", model_id);
            println!("Available models:");
            for m in &registry {
                println!("  - {} ({})", m.id, m.name);
            }
        }
        return Ok(());
    }

    // Check status and collect models that need downloading
    let mut to_download = Vec::new();
    let mut already_downloaded = Vec::new();

    for model in models_to_check {
        match manager.check_status(model).await {
            ModelStatus::Ready { path } => {
                already_downloaded.push((model, path));
            }
            _ => {
                to_download.push(model);
            }
        }
    }

    // Report already downloaded
    if !already_downloaded.is_empty() {
        println!("Already downloaded:");
        for (model, path) in &already_downloaded {
            println!("  {} - {:?}", model.name, path);
        }
    }

    // Report what needs downloading
    if to_download.is_empty() {
        println!("\nAll requested models are already downloaded.");
        return Ok(());
    }

    let total_size: u64 = to_download.iter().map(|m| m.size_bytes).sum();
    println!("\nModels to download ({}):", format_bytes(total_size));
    for model in &to_download {
        println!(
            "  {} ({}) - {}",
            model.id,
            model.name,
            format_bytes(model.size_bytes)
        );
    }

    if dry_run {
        println!("\n[Dry run - no downloads performed]");
        return Ok(());
    }

    // Download each model
    println!("\nDownloading to {:?}...\n", manager.model_path(&to_download[0]).parent().unwrap());

    for model in to_download {
        let pb = ProgressBar::new(model.size_bytes);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
                .progress_chars("#>-"),
        );
        pb.set_message(format!("Downloading {}", model.name));

        let pb_clone = Arc::new(pb);
        let pb_callback = pb_clone.clone();

        let progress_callback: Box<dyn Fn(u64, u64) + Send + Sync> =
            Box::new(move |downloaded, _total| {
                pb_callback.set_position(downloaded);
            });

        match manager.download(model, Some(progress_callback)).await {
            Ok(path) => {
                pb_clone.finish_with_message(format!("{} downloaded to {:?}", model.name, path));
            }
            Err(e) => {
                pb_clone.abandon_with_message(format!("Failed to download {}: {}", model.name, e));
                return Err(e.into());
            }
        }
    }

    println!("\nAll models downloaded successfully!");
    Ok(())
}

async fn list_models() -> Result<()> {
    let registry = get_model_registry();
    let defaults = get_default_models();
    let manager = DownloadManager::new(DownloadManager::default_model_dir());

    println!("Available models:\n");

    for model in &registry {
        let status = manager.check_status(model).await;
        let is_default = defaults.contains(&model.id.as_str());

        let status_str = match status {
            ModelStatus::Ready { .. } => "[downloaded]".to_string(),
            ModelStatus::NotDownloaded => "[not downloaded]".to_string(),
            ModelStatus::Failed { error } => format!("[failed: {}]", error),
            ModelStatus::Downloading { progress } => format!("[downloading: {:.0}%]", progress * 100.0),
        };

        let default_marker = if is_default { " (default)" } else { "" };

        println!(
            "  {} - {}{}\n    Type: {:?}, Size: {}\n    Status: {}\n",
            model.id,
            model.name,
            default_marker,
            model.model_type,
            format_bytes(model.size_bytes),
            status_str
        );
    }

    // Show downloaded models
    let downloaded = manager.list_downloaded().await?;
    if !downloaded.is_empty() {
        println!("Downloaded model files:");
        for path in downloaded {
            println!("  {:?}", path);
        }
    }

    Ok(())
}

fn list_devices() -> Result<()> {
    let devices = AudioCapture::list_devices()?;

    if devices.is_empty() {
        println!("No audio input devices found.");
        println!("\nNote: On Linux, you may need to install ALSA dev libraries:");
        println!("  sudo apt install libasound2-dev");
        return Ok(());
    }

    println!("Available audio input devices:\n");

    for device in devices {
        let default_marker = if device.is_default { " (default)" } else { "" };
        println!("  {}{}", device.name, default_marker);

        for config in &device.configs {
            println!(
                "    - {} ch, {}-{} Hz, {}",
                config.channels,
                config.min_sample_rate,
                config.max_sample_rate,
                config.sample_format
            );
        }
        println!();
    }

    Ok(())
}

async fn record_audio(duration_secs: u64, output: PathBuf, device: Option<String>, resample: bool) -> Result<()> {
    let capture = if let Some(ref device_name) = device {
        println!("Using device: {}", device_name);
        AudioCapture::with_device(device_name)?
    } else {
        AudioCapture::new()?
    };

    let native_rate = capture.sample_rate();
    let target_rate = if resample { WHISPER_SAMPLE_RATE } else { native_rate };

    println!(
        "Recording {} seconds at {} Hz{}...",
        duration_secs,
        native_rate,
        if resample { format!(" (resampling to {} Hz)", WHISPER_SAMPLE_RATE) } else { String::new() }
    );

    let pb = ProgressBar::new(duration_secs);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len}s")?
            .progress_chars("#>-"),
    );
    pb.set_message("Recording");

    // Start a task to update progress
    let pb_clone = pb.clone();
    let progress_handle = tokio::spawn(async move {
        for i in 0..duration_secs {
            tokio::time::sleep(Duration::from_secs(1)).await;
            pb_clone.set_position(i + 1);
        }
    });

    let duration = Duration::from_secs(duration_secs);
    let audio = capture.record_duration(duration).await?;

    progress_handle.abort();
    pb.finish_with_message("Recording complete");

    // Resample if requested
    let (final_samples, final_rate) = if resample && native_rate != WHISPER_SAMPLE_RATE {
        println!("Resampling {} Hz -> {} Hz...", native_rate, WHISPER_SAMPLE_RATE);
        let resampled = resample_to_whisper(&audio.samples, native_rate)
            .map_err(|e| anyhow::anyhow!("Resampling failed: {}", e))?;
        println!("Resampled {} -> {} samples", audio.samples.len(), resampled.len());
        (resampled, WHISPER_SAMPLE_RATE)
    } else {
        (audio.samples, native_rate)
    };

    // Write to WAV file
    println!("Writing to {:?}...", output);
    write_wav(&output, &final_samples, final_rate)?;

    let file_size = std::fs::metadata(&output)?.len();
    println!(
        "Saved {} samples @ {} Hz ({}) to {:?}",
        final_samples.len(),
        target_rate,
        format_bytes(file_size),
        output
    );
    println!("\nPlay with: aplay {:?}", output);

    Ok(())
}

async fn vad_test(threshold: f32, duration_secs: u64, device: Option<String>) -> Result<()> {

    let capture = if let Some(ref device_name) = device {
        println!("Using device: {}", device_name);
        AudioCapture::with_device(device_name)?
    } else {
        AudioCapture::new()?
    };

    let sample_rate = capture.sample_rate();
    println!("VAD Test - {} Hz, threshold: {}", sample_rate, threshold);
    println!("Speak to test. Press Ctrl+C to stop.\n");

    // Frame size for VAD (~50ms)
    let frame_samples = (sample_rate as f32 * 0.05) as usize;

    let mut vad = Vad::with_threshold(threshold);
    let mut last_state = yammer_audio::VadState::Silence;
    let mut speech_start_time = std::time::Instant::now();

    let duration = Duration::from_secs(duration_secs);
    let audio = capture.record_duration(duration).await?;

    // Process in frames
    for (i, frame) in audio.samples.chunks(frame_samples).enumerate() {
        let (state, _changed) = vad.process_frame(frame);
        let rms = Vad::calculate_rms(frame);
        let time_ms = (i * frame_samples) as f32 / sample_rate as f32 * 1000.0;

        // Print state changes
        if state != last_state {
            let state_str = match state {
                yammer_audio::VadState::Silence => "QUIET",
                yammer_audio::VadState::MaybeSpeech => "maybe speech...",
                yammer_audio::VadState::Speech => ">>> SPEECH <<<",
                yammer_audio::VadState::MaybeSilence => "maybe quiet...",
            };

            if state == yammer_audio::VadState::Speech {
                speech_start_time = std::time::Instant::now();
            }

            let duration_info = if last_state == yammer_audio::VadState::MaybeSilence
                && state == yammer_audio::VadState::Silence
            {
                format!(" (speech lasted {:.1}s)", speech_start_time.elapsed().as_secs_f32())
            } else {
                String::new()
            };

            println!(
                "[{:8.1}ms] RMS: {:.4} -> {}{}",
                time_ms, rms, state_str, duration_info
            );

            last_state = state;
        }
    }

    println!("\nVAD test complete.");
    Ok(())
}

async fn transcribe_file(file: PathBuf, model_path: Option<PathBuf>, timestamps: bool) -> Result<()> {
    // Find model path
    let model = if let Some(path) = model_path {
        path
    } else {
        // Auto-detect from downloaded models
        let manager = DownloadManager::new(DownloadManager::default_model_dir());
        let registry = get_model_registry();

        // Find a downloaded Whisper model
        let whisper_model = registry
            .iter()
            .find(|m| m.model_type == ModelType::Whisper && {
                let status = tokio::runtime::Handle::current()
                    .block_on(manager.check_status(m));
                matches!(status, ModelStatus::Ready { .. })
            });

        match whisper_model {
            Some(m) => {
                let status = manager.check_status(m).await;
                if let ModelStatus::Ready { path } = status {
                    println!("Using model: {} ({:?})", m.name, path);
                    path
                } else {
                    anyhow::bail!("Model not ready");
                }
            }
            None => {
                println!("No Whisper model found. Download one first:");
                println!("  yammer download-models");
                anyhow::bail!("No Whisper model available");
            }
        }
    };

    println!("Loading Whisper model...");
    let transcriber = Transcriber::new(&model)?;

    println!("Transcribing {:?}...\n", file);
    let transcript = transcriber.transcribe_file(&file)?;

    if timestamps {
        for segment in &transcript.segments {
            println!("{}", segment);
        }
    } else {
        println!("{}", transcript.text());
    }

    Ok(())
}

async fn dictate(
    model_path: Option<PathBuf>,
    threshold: f32,
    device: Option<String>,
    duration_secs: u64,
) -> Result<()> {
    // Find model path
    let model = if let Some(path) = model_path {
        path
    } else {
        // Auto-detect from downloaded models
        let manager = DownloadManager::new(DownloadManager::default_model_dir());
        let registry = get_model_registry();

        let whisper_model = registry
            .iter()
            .find(|m| m.model_type == ModelType::Whisper && {
                let status = tokio::runtime::Handle::current()
                    .block_on(manager.check_status(m));
                matches!(status, ModelStatus::Ready { .. })
            });

        match whisper_model {
            Some(m) => {
                let status = manager.check_status(m).await;
                if let ModelStatus::Ready { path } = status {
                    println!("Using model: {}", m.name);
                    path
                } else {
                    anyhow::bail!("Model not ready");
                }
            }
            None => {
                println!("No Whisper model found. Download one first:");
                println!("  yammer download-models");
                anyhow::bail!("No Whisper model available");
            }
        }
    };

    println!("Loading Whisper model...");
    let transcriber = Arc::new(Transcriber::new(&model)?);

    // Set up audio capture
    let capture = if let Some(ref device_name) = device {
        println!("Using device: {}", device_name);
        AudioCapture::with_device(device_name)?
    } else {
        AudioCapture::new()?
    };

    let native_rate = capture.sample_rate();
    println!(
        "Dictation mode - {} Hz, VAD threshold: {}",
        native_rate, threshold
    );
    println!("Speak to transcribe. Press Ctrl+C to stop.\n");

    // Start continuous capture (50ms chunks for VAD processing)
    let (_handle, mut audio_rx) = capture.start_capture(50)?;

    // Set up VAD processor
    let mut vad_processor = VadProcessor::with_threshold(threshold);

    // Set up Ctrl+C handler
    let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let running_clone = running.clone();
    ctrlc::set_handler(move || {
        running_clone.store(false, std::sync::atomic::Ordering::SeqCst);
    })?;

    let start_time = std::time::Instant::now();
    let duration = if duration_secs == 0 {
        None
    } else {
        Some(Duration::from_secs(duration_secs))
    };

    while running.load(std::sync::atomic::Ordering::SeqCst) {
        // Check duration limit
        if let Some(max_duration) = duration {
            if start_time.elapsed() >= max_duration {
                break;
            }
        }

        // Get next audio chunk with timeout
        let chunk = tokio::select! {
            Some(chunk) = audio_rx.recv() => chunk,
            _ = tokio::time::sleep(Duration::from_millis(100)) => continue,
        };

        // Process through VAD
        let events = vad_processor.process(&chunk);

        for event in events {
            if let VadEvent::SpeechEnd { samples } = event {
                // Speech segment complete, transcribe it
                let speech_duration_ms = (samples.len() as f32 / native_rate as f32 * 1000.0) as u32;

                // Skip very short segments (likely noise)
                if speech_duration_ms < 200 {
                    continue;
                }

                // Resample to 16kHz if needed
                let samples_16k = if native_rate != WHISPER_SAMPLE_RATE {
                    match resample_to_whisper(&samples, native_rate) {
                        Ok(resampled) => resampled,
                        Err(e) => {
                            eprintln!("Resample error: {}", e);
                            continue;
                        }
                    }
                } else {
                    samples
                };

                // Transcribe in a blocking task (whisper-rs is synchronous)
                let transcriber_clone = transcriber.clone();
                let result = tokio::task::spawn_blocking(move || {
                    transcriber_clone.transcribe(&samples_16k)
                }).await?;

                match result {
                    Ok(transcript) => {
                        let text = transcript.text();
                        if !text.trim().is_empty() {
                            print!("{} ", text.trim());
                            // Flush stdout to show text immediately
                            use std::io::Write;
                            std::io::stdout().flush()?;
                        }
                    }
                    Err(e) => {
                        eprintln!("\n[Transcription error: {}]", e);
                    }
                }
            }
        }
    }

    println!("\n\nDictation stopped.");
    Ok(())
}
