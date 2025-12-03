//! yammer-cli: headless CLI for testing building blocks
//!
//! This binary provides a command-line interface for testing
//! individual components of the yammer system.

use anyhow::Result;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    tracing::info!("yammer-cli starting");

    // Placeholder - will add actual CLI commands
    println!("yammer - Linux dictation app");
    println!("Run with --help for usage");

    Ok(())
}
