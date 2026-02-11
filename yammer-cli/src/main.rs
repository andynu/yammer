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
    format_bytes, get_default_models, get_model_registry, Config, DownloadManager, ModelStatus,
    ModelType, VerifiedHashes,
};
use yammer_llm::Corrector;
use yammer_output::{TextOutput, OutputMethod};
use yammer_stt::Transcriber;

/// Action to perform for config command
enum ConfigAction {
    /// Show overview of config status
    Overview,
    /// Show current configuration as TOML
    Show,
    /// Print path to config file
    Path,
    /// Initialize default config file
    Init,
}

/// Action to perform for hashes command
enum HashesAction {
    /// List all verified hashes
    List,
    /// Print path to hashes file
    Path,
    /// Clear all verified hashes
    Clear,
    /// Clear hash for a specific model
    ClearModel(String),
}

#[derive(Parser)]
#[command(name = "yammer")]
#[command(about = "Linux dictation app - local speech-to-text with LLM correction")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Launch the GUI application
    Gui {
        /// Toggle dictation in existing instance (for keyboard shortcut integration)
        #[arg(long)]
        toggle: bool,
    },

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
    ListDevices {
        /// Show all supported configurations (verbose)
        #[arg(long, short)]
        verbose: bool,
    },

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

    /// Correct text using LLM
    Correct {
        /// Text to correct
        text: String,

        /// Path to LLM model file (auto-detected from downloaded models if not specified)
        #[arg(long)]
        model: Option<PathBuf>,
    },

    /// Show GPU information and VRAM usage
    GpuInfo {
        /// Watch mode: continuously update VRAM usage
        #[arg(long, short)]
        watch: bool,
    },

    /// Type text into the focused application via xdotool
    TypeText {
        /// Text to type
        text: String,

        /// Use clipboard paste instead of simulated keystrokes
        #[arg(long)]
        clipboard: bool,

        /// Delay before typing (seconds, to switch focus)
        #[arg(long, default_value = "2")]
        delay: u64,
    },

    /// Show or initialize configuration
    Config {
        /// Show current configuration as TOML
        #[arg(long)]
        show: bool,

        /// Path to the config file (show path if not provided)
        #[arg(long)]
        path: bool,

        /// Initialize default config file if it doesn't exist
        #[arg(long)]
        init: bool,
    },

    /// Manage verified model checksums
    Hashes {
        /// Clear all verified hashes (forces re-verification on next download)
        #[arg(long)]
        clear: bool,

        /// Clear hash for a specific model ID
        #[arg(long)]
        clear_model: Option<String>,

        /// Show path to hashes file
        #[arg(long)]
        path: bool,
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
        Some(Commands::Gui { toggle }) => {
            launch_gui(toggle)?;
        }
        Some(Commands::DownloadModels { dry_run, all, model }) => {
            download_models(dry_run, all, model).await?;
        }
        Some(Commands::ListModels) => {
            list_models().await?;
        }
        Some(Commands::ListDevices { verbose }) => {
            list_devices(verbose)?;
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
        Some(Commands::Correct { text, model }) => {
            correct_text(text, model).await?;
        }
        Some(Commands::GpuInfo { watch }) => {
            gpu_info(watch).await?;
        }
        Some(Commands::TypeText { text, clipboard, delay }) => {
            type_text_cmd(text, clipboard, delay).await?;
        }
        Some(Commands::Config { show, path, init }) => {
            let action = if path {
                ConfigAction::Path
            } else if init {
                ConfigAction::Init
            } else if show {
                ConfigAction::Show
            } else {
                ConfigAction::Overview
            };
            config_cmd(action)?;
        }
        Some(Commands::Hashes {
            clear,
            clear_model,
            path,
        }) => {
            let action = if path {
                HashesAction::Path
            } else if clear {
                HashesAction::Clear
            } else if let Some(model_id) = clear_model {
                HashesAction::ClearModel(model_id)
            } else {
                HashesAction::List
            };
            hashes_cmd(action)?;
        }
        None => {
            println!("yammer - Linux dictation app");
            println!("Run with --help for usage");
        }
    }

    Ok(())
}

/// Find a downloaded model of the specified type
///
/// Returns the path to the first ready model of the given type, or an error
/// if no model is found. Uses stderr for output when `use_stderr` is true
/// (for dictate mode where stdout is used for transcription).
async fn find_downloaded_model(model_type: ModelType, use_stderr: bool) -> Result<PathBuf> {
    let manager = DownloadManager::new(DownloadManager::default_model_dir());
    let registry = get_model_registry();

    for m in registry.iter() {
        if m.model_type == model_type {
            let status = manager.check_status(m).await;
            if let ModelStatus::Ready { path } = status {
                if use_stderr {
                    eprintln!("Using model: {}", m.name);
                } else {
                    println!("Using model: {}", m.name);
                }
                return Ok(path);
            }
        }
    }

    let model_type_str = match model_type {
        ModelType::Whisper => "Whisper",
        ModelType::Llm => "LLM",
    };
    let msg = format!("No {} model found. Download one first:\n  yammer download-models", model_type_str);
    if use_stderr {
        eprintln!("{}", msg);
    } else {
        println!("{}", msg);
    }
    anyhow::bail!("No {} model available", model_type_str)
}

fn launch_gui(toggle: bool) -> Result<()> {
    use std::process::Command;

    // Find the GUI binary - look next to current executable first
    let current_exe = std::env::current_exe()?;
    let exe_dir = current_exe.parent().ok_or_else(|| anyhow::anyhow!("Cannot find executable directory"))?;

    // Try yammer-app in same directory as current binary
    let gui_binary = exe_dir.join("yammer-app");

    if !gui_binary.exists() {
        // Provide helpful error message
        eprintln!("GUI binary not found at: {:?}", gui_binary);
        eprintln!("\nTo build the GUI:");
        eprintln!("  cd yammer-app && npm run tauri build");
        eprintln!("\nOr for development:");
        eprintln!("  cd yammer-app && npm run tauri dev");
        anyhow::bail!("yammer-app binary not found");
    }

    // Build command with optional --toggle flag
    let mut cmd = Command::new(&gui_binary);
    if toggle {
        cmd.arg("--toggle");
    }

    // Spawn detached so CLI can exit
    cmd.spawn()
        .map_err(|e| anyhow::anyhow!("Failed to launch GUI: {}", e))?;

    if toggle {
        println!("Toggling dictation...");
    } else {
        println!("Launching Yammer GUI...");
    }

    Ok(())
}

async fn download_models(dry_run: bool, all: bool, specific_model: Option<String>) -> Result<()> {
    let registry = get_model_registry();
    let defaults = get_default_models();
    let mut manager = DownloadManager::new(DownloadManager::default_model_dir());

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
            ModelStatus::NotDownloaded => {
                to_download.push(model);
            }
            ModelStatus::Downloading { .. } => {
                // Currently downloading - still add to download queue to resume/restart
                to_download.push(model);
            }
            ModelStatus::Failed { error } => {
                eprintln!("Warning: {} previously failed: {}", model.name, error);
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
    let first_model = to_download.first().expect("to_download verified non-empty above");
    println!("\nDownloading to {:?}...\n", manager.model_path(first_model).parent().unwrap());

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

fn list_devices(verbose: bool) -> Result<()> {
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

        if verbose {
            // Verbose mode: show all configs
            for config in &device.configs {
                println!(
                    "    - {} ch, {}-{} Hz, {}",
                    config.channels,
                    config.min_sample_rate,
                    config.max_sample_rate,
                    config.sample_format
                );
            }
        } else if !device.configs.is_empty() {
            // Simplified mode: show single summary line
            let min_channels = device.configs.iter().map(|c| c.channels).min().unwrap_or(0);
            let max_channels = device.configs.iter().map(|c| c.channels).max().unwrap_or(0);
            let min_rate = device.configs.iter().map(|c| c.min_sample_rate).min().unwrap_or(0);
            let max_rate = device.configs.iter().map(|c| c.max_sample_rate).max().unwrap_or(0);

            let channels_str = if min_channels == max_channels {
                format!("{} ch", min_channels)
            } else {
                format!("{}-{} ch", min_channels, max_channels)
            };

            println!("    {} {}-{} Hz", channels_str, min_rate, max_rate);
        }
        println!();
    }

    if !verbose {
        println!("Tip: Use --verbose to see all supported configurations");
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
    let model = match model_path {
        Some(path) => path,
        None => find_downloaded_model(ModelType::Whisper, false).await?,
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

/// Dictation state for UI display
#[derive(Clone, Copy, PartialEq)]
enum DictateState {
    Listening,
    Recording,
    Processing,
}

impl DictateState {
    fn display(&self) -> &'static str {
        match self {
            DictateState::Listening => "○ Listening",
            DictateState::Recording => "● Recording",
            DictateState::Processing => "◐ Processing",
        }
    }

    fn color_code(&self) -> &'static str {
        match self {
            DictateState::Listening => "\x1b[90m",  // Gray
            DictateState::Recording => "\x1b[91m",  // Red
            DictateState::Processing => "\x1b[93m", // Yellow
        }
    }
}

/// Build ASCII audio level meter
fn audio_meter(rms: f32, width: usize) -> String {
    // Scale RMS (typically 0.0-0.3) to meter width
    let level = (rms * 10.0).min(1.0);
    let filled = (level * width as f32) as usize;

    let bar: String = (0..width)
        .map(|i| if i < filled { '█' } else { '░' })
        .collect();

    // Color based on level
    let color = if level > 0.7 {
        "\x1b[91m" // Red (loud)
    } else if level > 0.3 {
        "\x1b[92m" // Green (good)
    } else {
        "\x1b[90m" // Gray (quiet)
    };

    format!("{}{}│\x1b[0m", color, bar)
}

/// Update the status line without newline
fn update_status(state: DictateState, rms: f32) {
    use std::io::Write;
    let meter = audio_meter(rms, 20);
    // \r = carriage return, \x1b[K = clear to end of line
    print!(
        "\r\x1b[K{}{}\x1b[0m  {}",
        state.color_code(),
        state.display(),
        meter
    );
    let _ = std::io::stdout().flush();
}

/// Clear status line and print transcribed text
fn print_transcript(text: &str) {
    use std::io::Write;
    // Clear status line, print text, then newline
    print!("\r\x1b[K{}\n", text);
    let _ = std::io::stdout().flush();
}

async fn dictate(
    model_path: Option<PathBuf>,
    threshold: f32,
    device: Option<String>,
    duration_secs: u64,
) -> Result<()> {
    let model = match model_path {
        Some(path) => path,
        None => find_downloaded_model(ModelType::Whisper, true).await?,
    };

    eprintln!("Loading Whisper model...");
    let transcriber = Arc::new(Transcriber::new(&model)?);

    // Set up audio capture
    let capture = if let Some(ref device_name) = device {
        eprintln!("Using device: {}", device_name);
        AudioCapture::with_device(device_name)?
    } else {
        AudioCapture::new()?
    };

    let native_rate = capture.sample_rate();
    eprintln!(
        "Dictation mode - {} Hz, VAD threshold: {}",
        native_rate, threshold
    );
    eprintln!("Speak to transcribe. Press Ctrl+C to stop.\n");

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

    // Track current state and audio level for UI
    let mut current_state = DictateState::Listening;
    let mut current_rms: f32 = 0.0;

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
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Update display even when no audio (keeps meter visible)
                update_status(current_state, current_rms);
                continue;
            }
        };

        // Calculate RMS for audio meter
        current_rms = yammer_audio::Vad::calculate_rms(&chunk);

        // Process through VAD
        let events = vad_processor.process(&chunk);

        // Update state based on VAD
        let vad_state = vad_processor.vad().state();
        current_state = match vad_state {
            yammer_audio::VadState::Speech | yammer_audio::VadState::MaybeSilence => {
                DictateState::Recording
            }
            yammer_audio::VadState::Silence | yammer_audio::VadState::MaybeSpeech => {
                DictateState::Listening
            }
        };

        // Update status display
        update_status(current_state, current_rms);

        for event in events {
            if let VadEvent::SpeechEnd { samples } = event {
                // Speech segment complete - show processing state
                current_state = DictateState::Processing;
                update_status(current_state, current_rms);

                let speech_duration_ms = (samples.len() as f32 / native_rate as f32 * 1000.0) as u32;

                // Skip very short segments (likely noise)
                if speech_duration_ms < 200 {
                    current_state = DictateState::Listening;
                    continue;
                }

                // Resample to 16kHz if needed
                let samples_16k = if native_rate != WHISPER_SAMPLE_RATE {
                    match resample_to_whisper(&samples, native_rate) {
                        Ok(resampled) => resampled,
                        Err(e) => {
                            eprintln!("\r\x1b[KResample error: {}", e);
                            current_state = DictateState::Listening;
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
                            print_transcript(text.trim());
                        }
                    }
                    Err(e) => {
                        eprintln!("\r\x1b[K[Transcription error: {}]", e);
                    }
                }

                current_state = DictateState::Listening;
            }
        }
    }

    // Clear status line on exit
    print!("\r\x1b[K");
    eprintln!("Dictation stopped.");
    Ok(())
}

async fn correct_text(text: String, model_path: Option<PathBuf>) -> Result<()> {
    let model = match model_path {
        Some(path) => path,
        None => find_downloaded_model(ModelType::Llm, false).await?,
    };

    println!("Loading LLM model...");
    let corrector = Corrector::new(&model)?;

    println!("Correcting: \"{}\"", text);
    let result = corrector.correct(&text)?;

    println!("\nCorrected: \"{}\"", result.text);
    println!("Latency: {}ms", result.latency_ms);

    Ok(())
}

/// GPU information from nvidia-smi
#[derive(Debug)]
struct GpuMemory {
    device_name: String,
    total_mb: u64,
    used_mb: u64,
    free_mb: u64,
}

fn query_gpu_memory() -> Result<Vec<GpuMemory>> {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("nvidia-smi failed: {}", stderr);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut gpus = Vec::new();

    for line in stdout.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() >= 4 {
            gpus.push(GpuMemory {
                device_name: parts[0].to_string(),
                total_mb: parts[1].parse().unwrap_or(0),
                used_mb: parts[2].parse().unwrap_or(0),
                free_mb: parts[3].parse().unwrap_or(0),
            });
        }
    }

    Ok(gpus)
}

async fn gpu_info(watch: bool) -> Result<()> {
    // Check if nvidia-smi is available
    if std::process::Command::new("nvidia-smi")
        .arg("--version")
        .output()
        .is_err()
    {
        println!("nvidia-smi not found. Is NVIDIA driver installed?");
        println!("\nNote: GPU info requires NVIDIA GPU with proprietary drivers.");
        return Ok(());
    }

    let print_gpu_info = || -> Result<()> {
        let gpus = query_gpu_memory()?;

        if gpus.is_empty() {
            println!("No NVIDIA GPUs detected.");
            return Ok(());
        }

        println!("GPU Memory Usage:\n");
        for (i, gpu) in gpus.iter().enumerate() {
            let usage_percent = if gpu.total_mb > 0 {
                (gpu.used_mb as f64 / gpu.total_mb as f64 * 100.0) as u64
            } else {
                0
            };

            println!("  GPU {}: {}", i, gpu.device_name);
            println!(
                "    Total:  {:>6} MiB",
                gpu.total_mb
            );
            println!(
                "    Used:   {:>6} MiB ({}%)",
                gpu.used_mb, usage_percent
            );
            println!(
                "    Free:   {:>6} MiB",
                gpu.free_mb
            );
            println!();
        }

        // Print model size reference
        println!("Model VRAM estimates:");
        println!("  Whisper base.en:     ~142 MiB");
        println!("  Whisper small.en:    ~466 MiB");
        println!("  LLM TinyLlama Q4_K_M: ~637 MiB");
        println!("  KV cache (2048 ctx):  ~500-1000 MiB");
        println!("  CUDA overhead:        ~300-500 MiB");

        Ok(())
    };

    if watch {
        println!("Watching GPU memory (Ctrl+C to stop)...\n");

        // Set up Ctrl+C handler
        let running = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let running_clone = running.clone();
        ctrlc::set_handler(move || {
            running_clone.store(false, std::sync::atomic::Ordering::SeqCst);
        })?;

        while running.load(std::sync::atomic::Ordering::SeqCst) {
            // Clear screen and move cursor to top
            print!("\x1B[2J\x1B[H");
            println!("GPU Memory Monitor (Ctrl+C to stop)\n");
            print_gpu_info()?;
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    } else {
        print_gpu_info()?;
    }

    Ok(())
}

async fn type_text_cmd(text: String, clipboard: bool, delay: u64) -> Result<()> {
    // Check text output backend is available
    TextOutput::check_backend()?;

    let method = if clipboard {
        OutputMethod::Clipboard
    } else {
        OutputMethod::Type
    };

    let output = TextOutput::with_method(method);

    if delay > 0 {
        println!(
            "Will type {} characters in {} seconds. Switch to target window...",
            text.len(),
            delay
        );
        for i in (1..=delay).rev() {
            println!("  {}...", i);
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    }

    println!("Typing now!");
    output.output(&text)?;
    println!("Done.");

    Ok(())
}

fn config_cmd(action: ConfigAction) -> Result<()> {
    let config_path = Config::default_path();

    match action {
        ConfigAction::Overview => {
            println!("Configuration file: {:?}", config_path);
            if config_path.exists() {
                println!("Status: exists");
                println!("\nUse --show to display contents");
                println!("Use --path to print path only");
            } else {
                println!("Status: not found (using defaults)");
                println!("\nUse --init to create default config");
            }
        }
        ConfigAction::Path => {
            println!("{}", config_path.display());
        }
        ConfigAction::Init => {
            if config_path.exists() {
                println!("Config file already exists: {:?}", config_path);
                println!("Remove it first if you want to reset to defaults.");
            } else {
                Config::ensure_default_exists()
                    .map_err(|e| anyhow::anyhow!("Failed to create config: {}", e))?;
                println!("Created default config: {:?}", config_path);
            }
        }
        ConfigAction::Show => {
            let config = Config::load();
            let toml_str = config
                .to_toml()
                .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;
            println!("{}", toml_str);
        }
    }

    Ok(())
}

fn hashes_cmd(action: HashesAction) -> Result<()> {
    let hashes_path = VerifiedHashes::default_path();

    match action {
        HashesAction::Path => {
            println!("{}", hashes_path.display());
        }
        HashesAction::Clear => {
            let mut hashes = VerifiedHashes::load();
            hashes.clear();
            hashes
                .save()
                .map_err(|e| anyhow::anyhow!("Failed to save hashes: {}", e))?;
            println!("Cleared all verified hashes.");
        }
        HashesAction::ClearModel(model_id) => {
            let mut hashes = VerifiedHashes::load();
            if hashes.remove(&model_id).is_some() {
                hashes
                    .save()
                    .map_err(|e| anyhow::anyhow!("Failed to save hashes: {}", e))?;
                println!("Cleared hash for model: {}", model_id);
            } else {
                println!("No hash found for model: {}", model_id);
            }
        }
        HashesAction::List => {
            let hashes = VerifiedHashes::load();
            println!("Verified model checksums:\n");
            println!("File: {}\n", hashes_path.display());

            if hashes.hashes.is_empty() {
                println!("No verified hashes yet. Hashes are stored on first download of each model.");
            } else {
                let mut sorted: Vec<_> = hashes.hashes.iter().collect();
                sorted.sort_by_key(|(k, _)| k.as_str());

                for (model_id, sha256) in sorted {
                    println!("{}", model_id);
                    println!("  SHA256: {}", sha256);
                }
            }
        }
    }

    Ok(())
}
