//! Yammer Tauri application

mod pipeline;

use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
#[cfg(debug_assertions)]
use tauri::Manager;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};

use pipeline::{DictationPipeline, PipelineConfig, PipelineEvent};
use yammer_core::Config;
use yammer_output::OutputMethod;

/// Application state holding the pipeline and related resources
pub struct AppState {
    pipeline: Arc<Mutex<Option<DictationPipeline>>>,
    is_running: Arc<Mutex<bool>>,
    event_tx: mpsc::Sender<PipelineEvent>,
}

/// Initialize the pipeline with model paths
#[tauri::command]
async fn initialize_pipeline(
    state: State<'_, AppState>,
    whisper_model: Option<String>,
    llm_model: Option<String>,
    use_correction: bool,
) -> Result<(), String> {
    info!("Initializing pipeline...");

    // Load configuration
    let app_config = Config::load();
    info!("Loaded config from {:?}", Config::default_path());

    // Determine whisper model path
    let whisper_path = whisper_model
        .map(|m| app_config.models.model_dir.join(&m))
        .unwrap_or_else(|| app_config.whisper_model_path());

    // Determine LLM model path if correction enabled
    let llm_path = if use_correction {
        llm_model
            .map(|m| app_config.models.model_dir.join(&m))
            .or_else(|| app_config.llm_model_path())
    } else {
        None
    };

    // Determine output method from config
    let output_method = match app_config.output.method.as_str() {
        "clipboard" => OutputMethod::Clipboard,
        _ => OutputMethod::Type,
    };

    let pipeline_config = PipelineConfig {
        whisper_model_path: whisper_path,
        llm_model_path: llm_path,
        use_llm_correction: use_correction && app_config.llm_enabled(),
        output_method,
        vad_threshold: app_config.audio.vad_threshold,
        audio_device: app_config.audio.device.clone(),
    };

    let mut pipeline = DictationPipeline::new(pipeline_config, state.event_tx.clone());
    pipeline.initialize()?;

    let mut pipeline_guard = state.pipeline.lock().await;
    *pipeline_guard = Some(pipeline);

    info!("Pipeline initialized successfully");
    Ok(())
}

/// Start the dictation pipeline (called from frontend when user starts dictation)
#[tauri::command]
async fn start_dictation(
    state: State<'_, AppState>,
    _app: AppHandle,
) -> Result<(), String> {
    info!("Start dictation command received");

    // Check if already running
    {
        let is_running = state.is_running.lock().await;
        if *is_running {
            warn!("Dictation already in progress");
            return Err("Dictation already in progress".to_string());
        }
    }

    // Verify pipeline is initialized
    {
        let pipeline_guard = state.pipeline.lock().await;
        match pipeline_guard.as_ref() {
            Some(p) if p.is_initialized() => {}
            _ => return Err("Pipeline not initialized. Call initialize_pipeline first.".to_string()),
        }
    }

    // Mark as running
    {
        let mut is_running = state.is_running.lock().await;
        *is_running = true;
    }

    // Get pipeline reference for blocking task
    let pipeline_state = state.pipeline.clone();
    let is_running_state = state.is_running.clone();

    // Run pipeline in blocking task (audio capture isn't Send-safe)
    tokio::task::spawn_blocking(move || {
        let result = {
            // We need to block on the mutex since we're in a blocking context
            let rt = tokio::runtime::Handle::current();
            let pipeline_guard = rt.block_on(pipeline_state.lock());
            if let Some(ref pipeline) = *pipeline_guard {
                pipeline.run_blocking()
            } else {
                Err("Pipeline not available".to_string())
            }
        };

        // Mark as no longer running
        {
            let rt = tokio::runtime::Handle::current();
            let mut is_running = rt.block_on(is_running_state.lock());
            *is_running = false;
        }

        match result {
            Ok(text) => {
                info!("Dictation completed successfully: \"{}\"", text);
            }
            Err(ref e) if e == "Cancelled" => {
                info!("Dictation was cancelled");
            }
            Err(ref e) => {
                error!("Dictation failed: {}", e);
            }
        }
    });

    Ok(())
}

/// Stop/cancel the current dictation
#[tauri::command]
async fn stop_dictation(state: State<'_, AppState>) -> Result<(), String> {
    info!("Stop dictation command received");

    let pipeline_guard = state.pipeline.lock().await;
    if let Some(ref pipeline) = *pipeline_guard {
        pipeline.cancel();
    }

    Ok(())
}

/// Toggle dictation (start if idle, stop if running)
#[tauri::command]
async fn toggle_dictation(
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<bool, String> {
    let is_running = *state.is_running.lock().await;

    if is_running {
        stop_dictation(state).await?;
        Ok(false) // Now stopped
    } else {
        start_dictation(state, app).await?;
        Ok(true) // Now running
    }
}

/// Get current pipeline state (simplified - relies on events for detailed state)
#[tauri::command]
async fn get_pipeline_state(state: State<'_, AppState>) -> Result<String, String> {
    let pipeline_guard = state.pipeline.lock().await;
    if pipeline_guard.is_none() {
        return Ok("uninitialized".to_string());
    }

    let is_running = *state.is_running.lock().await;
    if is_running {
        Ok("running".to_string())
    } else {
        Ok("idle".to_string())
    }
}

/// Check if models are downloaded
#[tauri::command]
async fn check_models() -> Result<serde_json::Value, String> {
    let app_config = Config::load();

    let whisper_path = app_config.whisper_model_path();
    let whisper_exists = whisper_path.exists();

    let llm_path = app_config.llm_model_path();
    let llm_exists = llm_path.as_ref().is_some_and(|p| p.exists());

    Ok(serde_json::json!({
        "models_dir": app_config.models.model_dir.to_string_lossy(),
        "whisper": {
            "exists": whisper_exists,
            "path": whisper_path.to_string_lossy()
        },
        "llm": {
            "exists": llm_exists,
            "path": llm_path.map(|p| p.to_string_lossy().to_string()).unwrap_or_default(),
            "enabled": app_config.llm_enabled()
        }
    }))
}

/// Quit the application
#[tauri::command]
fn quit_app(app: AppHandle) {
    info!("Quit command received");
    app.exit(0);
}

/// Save current window position to config
#[tauri::command]
async fn save_window_position(x: i32, y: i32) -> Result<(), String> {
    debug!("Saving window position: ({}, {})", x, y);
    let mut config = Config::load();
    config.set_window_position(x, y)?;
    Ok(())
}

/// Get saved window position (validated against screen bounds)
#[tauri::command]
async fn get_saved_window_position(
    screen_width: u32,
    screen_height: u32,
    window_width: u32,
    window_height: u32,
) -> Result<Option<(i32, i32)>, String> {
    let config = Config::load();
    Ok(config.validated_window_position(screen_width, screen_height, window_width, window_height))
}

/// Simulate audio waveform data for testing (legacy command, kept for compatibility)
#[tauri::command]
fn simulate_audio(app: AppHandle) -> Result<(), String> {
    let samples: Vec<f32> = (0..40)
        .map(|i| {
            let t = i as f32 * 0.1;
            (t.sin() * 0.5 + 0.5) * (1.0 - (i as f32 / 40.0))
        })
        .collect();

    app.emit("audio-samples", samples)
        .map_err(|e| e.to_string())?;

    Ok(())
}

/// Initialize and run the Tauri application
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("yammer=debug".parse().unwrap())
                .add_directive("yammer_app=debug".parse().unwrap())
        )
        .init();

    info!("Starting Yammer application");

    // Create event channel for pipeline events
    let (event_tx, mut event_rx) = mpsc::channel::<PipelineEvent>(100);

    // Create app state
    let app_state = AppState {
        pipeline: Arc::new(Mutex::new(None)),
        is_running: Arc::new(Mutex::new(false)),
        event_tx,
    };

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    // Only trigger on key press, not release
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }

                    // Check for our dictation toggle shortcut (Ctrl+Shift+Space)
                    let dictate_shortcut =
                        Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
                    if shortcut == &dictate_shortcut {
                        debug!("Dictation hotkey pressed (Ctrl+Shift+Space)");

                        // Emit event to frontend to toggle dictation
                        if let Err(e) = app.emit("dictation-toggle", ()) {
                            error!("Failed to emit dictation-toggle: {}", e);
                        }
                    }
                })
                .build(),
        )
        .manage(app_state)
        .setup(|app| {
            #[cfg(debug_assertions)]
            {
                let window = app.get_webview_window("main").expect("Main window not found");
                info!("Window created: {:?}", window.label());
            }

            // Register global hotkey: Ctrl+Shift+Space for dictation toggle
            let dictate_shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
            match app.global_shortcut().register(dictate_shortcut) {
                Ok(_) => {
                    info!("Registered global shortcut: Ctrl+Shift+Space");
                }
                Err(e) => {
                    error!(
                        "Failed to register global shortcut Ctrl+Shift+Space: {}. \
                         Another application may have grabbed this key.",
                        e
                    );
                }
            }

            // Spawn task to forward pipeline events to frontend
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = event_rx.recv().await {
                    match event {
                        PipelineEvent::StateChanged(state) => {
                            let _ = app_handle.emit("pipeline-state", state.as_str());
                        }
                        PipelineEvent::AudioLevel(level) => {
                            // Convert to waveform samples (spread across 40 bars)
                            let samples: Vec<f32> = (0..40)
                                .map(|i| {
                                    let phase = i as f32 * 0.15;
                                    let variation = (phase.sin() * 0.3 + 0.7) * level;
                                    variation.min(1.0)
                                })
                                .collect();
                            let _ = app_handle.emit("audio-samples", samples);
                        }
                        PipelineEvent::Transcript { text, is_partial } => {
                            let _ = app_handle.emit("transcript", serde_json::json!({
                                "text": text,
                                "isPartial": is_partial
                            }));
                        }
                        PipelineEvent::Error(err) => {
                            let _ = app_handle.emit("pipeline-error", err);
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_pipeline,
            start_dictation,
            stop_dictation,
            toggle_dictation,
            get_pipeline_state,
            check_models,
            simulate_audio,
            quit_app,
            save_window_position,
            get_saved_window_position,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
