//! yammer-cli: headless CLI for testing building blocks
//!
//! This binary provides a command-line interface for testing
//! individual components of the yammer system.

use anyhow::Result;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};
use yammer_core::{
    format_bytes, get_default_models, get_model_registry, DownloadManager, ModelStatus,
};

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
