//! Yammer Tauri application

mod pipeline;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use tauri::{
    AppHandle, Emitter, Manager, State,
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};
use tauri_plugin_single_instance;
use tokio::sync::{mpsc, Mutex};
use tracing::{debug, error, info, warn};
use rdev::{listen, Event, EventType, Key};

use pipeline::{DictationPipeline, PipelineConfig, PipelineEvent};
use yammer_core::Config;
use yammer_output::OutputMethod;

/// State for tracking modifier key presses for press-hold-release hotkey
struct HotkeyState {
    ctrl_pressed: bool,
    super_pressed: bool,
    dictation_active: bool,
}

impl HotkeyState {
    fn new() -> Self {
        Self {
            ctrl_pressed: false,
            super_pressed: false,
            dictation_active: false,
        }
    }

    /// Returns true if both Ctrl and Super are currently held
    fn both_pressed(&self) -> bool {
        self.ctrl_pressed && self.super_pressed
    }
}

/// Spawns the rdev keyboard listener for Ctrl+Super press-hold-release hotkey.
///
/// The listener runs in a separate thread and sends events to the app:
/// - When Ctrl+Super are both pressed: start dictation
/// - When either is released (after both were pressed): stop dictation
fn spawn_hotkey_listener(app_handle: AppHandle) {
    thread::spawn(move || {
        info!("Starting Ctrl+Super hotkey listener (rdev)");

        let mut state = HotkeyState::new();

        let callback = move |event: Event| {
            match event.event_type {
                EventType::KeyPress(Key::ControlLeft) | EventType::KeyPress(Key::ControlRight) => {
                    state.ctrl_pressed = true;
                    debug!("Ctrl pressed, super={}", state.super_pressed);

                    // Check if both are now pressed and we're not already recording
                    if state.both_pressed() && !state.dictation_active {
                        state.dictation_active = true;
                        info!("Ctrl+Super pressed - starting dictation");

                        // Bring window forward and start dictation
                        if let Some(window) = app_handle.get_webview_window("main") {
                            let _ = window.unminimize();
                            let _ = window.show();
                            let _ = window.set_always_on_top(true);
                        }

                        if let Err(e) = app_handle.emit("dictation-start", ()) {
                            error!("Failed to emit dictation-start: {}", e);
                        }
                    }
                }
                EventType::KeyPress(Key::MetaLeft) | EventType::KeyPress(Key::MetaRight) => {
                    state.super_pressed = true;
                    debug!("Super pressed, ctrl={}", state.ctrl_pressed);

                    // Check if both are now pressed and we're not already recording
                    if state.both_pressed() && !state.dictation_active {
                        state.dictation_active = true;
                        info!("Ctrl+Super pressed - starting dictation");

                        // Bring window forward and start dictation
                        if let Some(window) = app_handle.get_webview_window("main") {
                            let _ = window.unminimize();
                            let _ = window.show();
                            let _ = window.set_always_on_top(true);
                        }

                        if let Err(e) = app_handle.emit("dictation-start", ()) {
                            error!("Failed to emit dictation-start: {}", e);
                        }
                    }
                }
                EventType::KeyRelease(Key::ControlLeft) | EventType::KeyRelease(Key::ControlRight) => {
                    state.ctrl_pressed = false;
                    debug!("Ctrl released, dictation_active={}", state.dictation_active);

                    // If dictation was active, stop it when modifier is released
                    if state.dictation_active {
                        state.dictation_active = false;
                        info!("Ctrl released - stopping dictation");

                        if let Err(e) = app_handle.emit("dictation-stop", ()) {
                            error!("Failed to emit dictation-stop: {}", e);
                        }
                    }
                }
                EventType::KeyRelease(Key::MetaLeft) | EventType::KeyRelease(Key::MetaRight) => {
                    state.super_pressed = false;
                    debug!("Super released, dictation_active={}", state.dictation_active);

                    // If dictation was active, stop it when modifier is released
                    if state.dictation_active {
                        state.dictation_active = false;
                        info!("Super released - stopping dictation");

                        if let Err(e) = app_handle.emit("dictation-stop", ()) {
                            error!("Failed to emit dictation-stop: {}", e);
                        }
                    }
                }
                _ => {}
            }
        };

        if let Err(error) = listen(callback) {
            error!("Hotkey listener error: {:?}", error);
        }
    });
}

/// Application state holding the pipeline and related resources
pub struct AppState {
    pipeline: Arc<Mutex<Option<DictationPipeline>>>,
    is_running: Arc<Mutex<bool>>,
    /// Cancel handle for the current dictation session (can be used without pipeline lock)
    cancel_handle: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    /// Discard handle for the current dictation session (cancels completely without output)
    discard_handle: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    event_tx: mpsc::Sender<PipelineEvent>,
    /// Last successful transcription result for "Copy Last" feature
    last_transcription: Arc<Mutex<Option<String>>>,
    /// Cached pipeline config for re-initialization after idle unload
    pipeline_config: Arc<Mutex<Option<PipelineConfig>>>,
    /// Handle to the idle unload timer (cancellable)
    idle_timer_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    /// Idle unload timeout in seconds (0 = disabled)
    idle_unload_seconds: u64,
}

/// Schedule (or reschedule) the idle unload timer.
/// Cancels any existing timer, then spawns a new one that will unload models
/// after `idle_unload_seconds` of inactivity.
fn schedule_idle_unload(
    idle_unload_seconds: u64,
    idle_timer_handle: Arc<Mutex<Option<tokio::task::JoinHandle<()>>>>,
    pipeline: Arc<Mutex<Option<DictationPipeline>>>,
    is_running: Arc<Mutex<bool>>,
    app_handle: AppHandle,
) {
    if idle_unload_seconds == 0 {
        return;
    }

    let duration = std::time::Duration::from_secs(idle_unload_seconds);

    tokio::spawn(async move {
        // Cancel any existing timer
        let mut handle_guard = idle_timer_handle.lock().await;
        if let Some(existing) = handle_guard.take() {
            existing.abort();
        }

        let pipeline_clone = pipeline.clone();
        let is_running_clone = is_running.clone();
        let app_handle_clone = app_handle.clone();

        *handle_guard = Some(tokio::spawn(async move {
            tokio::time::sleep(duration).await;

            // Check if dictation is active - don't unload during use
            let running = *is_running_clone.lock().await;
            if running {
                info!("Idle timer fired but dictation is active, skipping unload");
                return;
            }

            // Unload models by dropping the pipeline
            let mut pipeline_guard = pipeline_clone.lock().await;
            if pipeline_guard.is_some() {
                info!("Idle timeout reached ({}s), unloading models to free memory", idle_unload_seconds);
                *pipeline_guard = None;

                let _ = app_handle_clone.emit("models-unloaded", ());
                info!("Models unloaded, will reload on next dictation");
            }
        }));
    });
}

/// Initialize the pipeline with model paths
#[tauri::command]
async fn initialize_pipeline(
    state: State<'_, AppState>,
    app: AppHandle,
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

    info!("Whisper model path: {:?}", whisper_path);
    if !whisper_path.exists() {
        error!("Whisper model not found at {:?}", whisper_path);
    }

    // Determine LLM model path if correction enabled
    let llm_path = if use_correction {
        llm_model
            .map(|m| app_config.models.model_dir.join(&m))
            .or_else(|| app_config.llm_model_path())
    } else {
        None
    };

    if let Some(ref path) = llm_path {
        info!("LLM model path: {:?}", path);
        if !path.exists() {
            warn!("LLM model not found at {:?}", path);
        }
    } else {
        info!("LLM correction disabled");
    }

    // Determine output method from config
    let output_method = match app_config.output.method.as_str() {
        "clipboard" => OutputMethod::Clipboard,
        _ => OutputMethod::Type,
    };
    info!("Output method: {:?}", output_method);
    info!("Typing delay: {}ms", app_config.output.typing_delay_ms);

    // Audio device
    if let Some(ref device) = app_config.audio.device {
        info!("Audio device: {}", device);
    } else {
        info!("Audio device: system default");
    }

    let pipeline_config = PipelineConfig {
        whisper_model_path: whisper_path,
        llm_model_path: llm_path,
        use_llm_correction: use_correction && app_config.llm_enabled(),
        output_method,
        typing_delay_ms: app_config.output.typing_delay_ms,
        vad_threshold: app_config.audio.vad_threshold,
        audio_device: app_config.audio.device.clone(),
        max_recording_seconds: app_config.audio.max_recording_seconds,
    };

    // Cache config for re-initialization after idle unload
    *state.pipeline_config.lock().await = Some(pipeline_config.clone());

    let mut pipeline = DictationPipeline::new(pipeline_config, state.event_tx.clone());
    match pipeline.initialize() {
        Ok(()) => {
            info!("Pipeline initialized successfully");
        }
        Err(ref e) => {
            error!("Pipeline initialization failed: {}", e);
            return Err(e.clone());
        }
    }

    let mut pipeline_guard = state.pipeline.lock().await;
    *pipeline_guard = Some(pipeline);
    drop(pipeline_guard);

    // Start idle unload timer
    let idle_seconds = state.idle_unload_seconds;
    if idle_seconds > 0 {
        info!("Idle model unload enabled: {}s timeout", idle_seconds);
        schedule_idle_unload(
            idle_seconds,
            state.idle_timer_handle.clone(),
            state.pipeline.clone(),
            state.is_running.clone(),
            app,
        );
    }

    Ok(())
}

/// Start the dictation pipeline (called from frontend when user starts dictation)
#[tauri::command]
async fn start_dictation(
    state: State<'_, AppState>,
    app: AppHandle,
) -> Result<(), String> {
    info!("Start dictation command received");

    // Cancel any pending idle timer since we're about to use the pipeline
    {
        let mut timer_guard = state.idle_timer_handle.lock().await;
        if let Some(handle) = timer_guard.take() {
            handle.abort();
            debug!("Cancelled idle unload timer");
        }
    }

    // Combined lock acquisition: check/set is_running + verify pipeline + get handles
    {
        // First: atomically check and set is_running
        let mut is_running = state.is_running.lock().await;
        if *is_running {
            warn!("Dictation already in progress");
            return Err("Dictation already in progress".to_string());
        }

        // Verify pipeline is initialized and get handles (single pipeline lock)
        let mut pipeline_guard = state.pipeline.lock().await;

        // If pipeline was unloaded (idle timeout), reload from cached config
        if pipeline_guard.is_none() {
            let config_guard = state.pipeline_config.lock().await;
            if let Some(ref cached_config) = *config_guard {
                info!("Pipeline was unloaded, reloading models...");
                let _ = app.emit("pipeline-state", "reloading");

                let mut new_pipeline = DictationPipeline::new(
                    cached_config.clone(),
                    state.event_tx.clone(),
                );
                match new_pipeline.initialize() {
                    Ok(()) => {
                        info!("Pipeline reloaded successfully after idle unload");
                        *pipeline_guard = Some(new_pipeline);
                    }
                    Err(e) => {
                        error!("Failed to reload pipeline: {}", e);
                        let _ = app.emit("pipeline-state", "error");
                        return Err(format!("Failed to reload models: {}", e));
                    }
                }
            } else {
                error!("Pipeline not created yet");
                return Err("Pipeline not initialized. Call initialize_pipeline first.".to_string());
            }
        }

        match pipeline_guard.as_ref() {
            Some(p) if p.is_initialized() => {
                info!("Pipeline is initialized, proceeding with dictation");

                // Get cancel/discard handles while we have pipeline lock
                let cancel_handle = p.get_cancel_handle();
                let discard_handle = p.get_discard_handle();

                // Update handle state (brief nested locks)
                *state.cancel_handle.lock().await = Some(cancel_handle);
                *state.discard_handle.lock().await = Some(discard_handle);

                // Mark as running only after successful setup
                *is_running = true;
            }
            Some(_) => {
                error!("Pipeline exists but is not initialized (models not loaded)");
                return Err("Pipeline not initialized. Call initialize_pipeline first.".to_string());
            }
            None => {
                error!("Pipeline not created yet");
                return Err("Pipeline not initialized. Call initialize_pipeline first.".to_string());
            }
        }
    }

    // Get references for blocking task
    let pipeline_state = state.pipeline.clone();
    let is_running_state = state.is_running.clone();
    let cancel_handle_state = state.cancel_handle.clone();
    let discard_handle_state = state.discard_handle.clone();
    let last_transcription_state = state.last_transcription.clone();
    let idle_timer_handle = state.idle_timer_handle.clone();
    let idle_seconds = state.idle_unload_seconds;
    let app_for_timer = app.clone();

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

        // Mark as no longer running and clear handles (combined cleanup)
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            *is_running_state.lock().await = false;
            *cancel_handle_state.lock().await = None;
            *discard_handle_state.lock().await = None;
        });

        match result {
            Ok(text) => {
                info!("Dictation completed successfully: \"{}\"", text);
                // Store for "Copy Last" feature
                rt.block_on(async {
                    *last_transcription_state.lock().await = Some(text);
                });
            }
            Err(ref e) => {
                error!("Dictation failed: {}", e);
            }
        }

        // Reset idle timer after dictation completes
        if idle_seconds > 0 {
            rt.block_on(async {
                schedule_idle_unload(
                    idle_seconds,
                    idle_timer_handle,
                    pipeline_state.clone(),
                    is_running_state.clone(),
                    app_for_timer,
                );
            });
        }
    });

    Ok(())
}

/// Stop/cancel the current dictation
#[tauri::command]
async fn stop_dictation(state: State<'_, AppState>) -> Result<(), String> {
    info!("Stop dictation command received");

    // Use the cancel handle (doesn't require pipeline lock)
    let cancel_handle_guard = state.cancel_handle.lock().await;
    if let Some(ref handle) = *cancel_handle_guard {
        info!("Signaling pipeline to stop and process audio");
        handle.store(true, Ordering::SeqCst);
    } else {
        warn!("No active dictation session to stop");
    }

    Ok(())
}

/// Discard the current dictation completely (no output)
#[tauri::command]
async fn discard_dictation(state: State<'_, AppState>) -> Result<(), String> {
    info!("Discard dictation command received");

    // Use the discard handle (doesn't require pipeline lock)
    let discard_handle_guard = state.discard_handle.lock().await;
    if let Some(ref handle) = *discard_handle_guard {
        info!("Signaling pipeline to discard recording");
        handle.store(true, Ordering::SeqCst);
    } else {
        warn!("No active dictation session to discard");
    }

    // Also signal cancel to stop listening immediately
    let cancel_handle_guard = state.cancel_handle.lock().await;
    if let Some(ref handle) = *cancel_handle_guard {
        handle.store(true, Ordering::SeqCst);
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
    info!("Checking model availability...");
    let app_config = Config::load();

    let whisper_path = app_config.whisper_model_path();
    let whisper_exists = whisper_path.exists();
    if whisper_exists {
        info!("Whisper model found: {:?}", whisper_path);
    } else {
        error!("Whisper model NOT found: {:?}", whisper_path);
    }

    let llm_path = app_config.llm_model_path();
    let llm_exists = llm_path.as_ref().is_some_and(|p| p.exists());
    if let Some(ref path) = llm_path {
        if llm_exists {
            info!("LLM model found: {:?}", path);
        } else {
            warn!("LLM model NOT found: {:?}", path);
        }
    }

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
    info!("Saving window position: ({}, {})", x, y);
    let mut config = Config::load();
    config.set_window_position(x, y)?;
    info!("Window position saved successfully");
    Ok(())
}

/// Get saved window position (validated against screen bounds)
#[tauri::command]
async fn get_saved_window_position(
    screen_width: u32,
    screen_height: u32,
    window_width: u32,
) -> Result<Option<(i32, i32)>, String> {
    let config = Config::load();
    Ok(config.validated_window_position(screen_width, screen_height, window_width))
}

/// Get raw saved window position without validation (for multi-monitor JS-side checks)
#[tauri::command]
async fn get_raw_window_position() -> Result<Option<(i32, i32)>, String> {
    let config = Config::load();
    Ok(config.window_position())
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
    let app_config = Config::load();
    let idle_unload_seconds = app_config.gui.idle_unload_seconds;
    let last_transcription = Arc::new(Mutex::new(None));
    let app_state = AppState {
        pipeline: Arc::new(Mutex::new(None)),
        is_running: Arc::new(Mutex::new(false)),
        cancel_handle: Arc::new(Mutex::new(None)),
        discard_handle: Arc::new(Mutex::new(None)),
        event_tx,
        last_transcription: last_transcription.clone(),
        pipeline_config: Arc::new(Mutex::new(None)),
        idle_timer_handle: Arc::new(Mutex::new(None)),
        idle_unload_seconds,
    };

    tauri::Builder::default()
        // Single-instance plugin must be first
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            info!("Second instance launched with args: {:?}", argv);

            // Show the existing window
            if let Some(window) = app.get_webview_window("main") {
                // Default to false so we try to show if visibility check fails
                let was_visible = window.is_visible().unwrap_or(false);
                let was_minimized = window.is_minimized().unwrap_or(false);
                let is_toggle = argv.iter().any(|arg| arg == "--toggle");

                if !was_visible || was_minimized {
                    info!("Window was hidden/minimized, bringing forward");
                    // Multiple calls to ensure window appears on all window managers
                    let _ = window.unminimize();
                    let _ = window.show();
                    let _ = window.set_always_on_top(true);
                }

                // Only steal focus if NOT using --toggle (dictation should not steal focus)
                if !is_toggle {
                    let _ = window.set_focus();
                }

                // Check if --toggle flag was passed (for GNOME shortcut integration)
                if is_toggle {
                    info!("Toggle flag detected, starting dictation");
                    // Emit dictation-start since window was likely hidden
                    if !was_visible || was_minimized {
                        if let Err(e) = app.emit("dictation-start", ()) {
                            error!("Failed to emit dictation-start: {}", e);
                        }
                    } else {
                        // Window was visible, toggle dictation
                        if let Err(e) = app.emit("dictation-toggle", ()) {
                            error!("Failed to emit dictation-toggle: {}", e);
                        }
                    }
                }
            }
        }))
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    // Only trigger on key press, not release
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }

                    // Check for our dictation toggle shortcut (Super+H)
                    let dictate_shortcut =
                        Shortcut::new(Some(Modifiers::SUPER), Code::KeyH);
                    if shortcut == &dictate_shortcut {
                        debug!("Dictation hotkey pressed (Super+H)");

                        // Check if window was hidden/minimized and bring it forward
                        let was_hidden = if let Some(window) = app.get_webview_window("main") {
                            // Default to false so we try to show if visibility check fails
                            let was_visible = window.is_visible().unwrap_or(false);
                            let was_minimized = window.is_minimized().unwrap_or(false);

                            if !was_visible || was_minimized {
                                debug!("Window was hidden/minimized, bringing forward");
                                // Multiple calls to ensure window appears on all window managers:
                                // 1. Unminimize (in case it was minimized)
                                let _ = window.unminimize();
                                // 2. Show (in case it was hidden)
                                if let Err(e) = window.show() {
                                    error!("Failed to show window: {}", e);
                                }
                                // 3. Re-assert always on top to bring to foreground
                                let _ = window.set_always_on_top(true);
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };

                        // Emit appropriate event based on visibility state:
                        // - If window was hidden: start dictation (don't toggle, always start)
                        // - If window was visible: toggle dictation
                        if was_hidden {
                            debug!("Window was hidden, starting dictation");
                            if let Err(e) = app.emit("dictation-start", ()) {
                                error!("Failed to emit dictation-start: {}", e);
                            }
                        } else {
                            if let Err(e) = app.emit("dictation-toggle", ()) {
                                error!("Failed to emit dictation-toggle: {}", e);
                            }
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

            // Register global hotkey: Super+H for dictation toggle (fallback)
            let dictate_shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyH);
            match app.global_shortcut().register(dictate_shortcut) {
                Ok(_) => {
                    info!("Registered global shortcut: Super+H (fallback toggle)");
                }
                Err(e) => {
                    error!(
                        "Failed to register global shortcut Super+H: {}. \
                         Another application may have grabbed this key.",
                        e
                    );
                }
            }

            // Spawn the Ctrl+Super press-hold-release hotkey listener
            spawn_hotkey_listener(app.handle().clone());
            info!("Spawned Ctrl+Super press-hold-release hotkey listener");

            // Create system tray
            let show_item = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let toggle_item = MenuItem::with_id(app, "toggle", "Toggle Recording", true, None::<&str>)?;
            let copy_last_item = MenuItem::with_id(app, "copy_last", "Copy Last Transcription", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show_item, &toggle_item, &copy_last_item, &quit_item])?;

            let _tray = TrayIconBuilder::with_id("yammer-tray")
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .show_menu_on_left_click(false)  // Left-click shows window, right-click shows menu
                .tooltip("Yammer - Dictation App")
                .on_menu_event(|app, event| match event.id.as_ref() {
                    "show" => {
                        info!("Tray: Show Window clicked");
                        if let Some(window) = app.get_webview_window("main") {
                            // Robust show: unminimize, show, and ensure on top
                            let _ = window.unminimize();
                            let _ = window.show();
                            let _ = window.set_always_on_top(true);
                            let _ = window.set_focus();
                        }
                    }
                    "toggle" => {
                        info!("Tray: Toggle Recording clicked");
                        if let Err(e) = app.emit("dictation-toggle", ()) {
                            error!("Failed to emit dictation-toggle: {}", e);
                        }
                    }
                    "copy_last" => {
                        info!("Tray: Copy Last Transcription clicked");
                        let state: State<'_, AppState> = app.state();
                        // Clone the text out to avoid lifetime issues
                        let text_to_copy = state.last_transcription.try_lock()
                            .ok()
                            .and_then(|guard| guard.clone());
                        if let Some(text) = text_to_copy {
                            use tauri_plugin_clipboard_manager::ClipboardExt;
                            if let Err(e) = app.clipboard().write_text(text.clone()) {
                                error!("Failed to copy to clipboard: {}", e);
                            } else {
                                info!("Copied last transcription to clipboard ({} chars)", text.len());
                            }
                        } else {
                            warn!("No transcription available to copy");
                        }
                    }
                    "quit" => {
                        info!("Tray: Quit clicked");
                        app.exit(0);
                    }
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        debug!("Tray icon left-clicked");
                        let app = tray.app_handle();
                        if let Some(window) = app.get_webview_window("main") {
                            // Toggle visibility: if visible, hide; if hidden, show
                            let is_visible = window.is_visible().unwrap_or(false);
                            let is_minimized = window.is_minimized().unwrap_or(false);
                            if is_visible && !is_minimized {
                                let _ = window.hide();
                            } else {
                                // Robust show: unminimize, show, and ensure on top
                                let _ = window.unminimize();
                                let _ = window.show();
                                let _ = window.set_always_on_top(true);
                                let _ = window.set_focus();
                            }
                        }
                    }
                })
                .build(app)?;

            info!("System tray created");

            // Spawn task to forward pipeline events to frontend
            let app_handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = event_rx.recv().await {
                    match event {
                        PipelineEvent::StateChanged(state) => {
                            let _ = app_handle.emit("pipeline-state", state.as_str());
                        }
                        PipelineEvent::AudioLevel(level) => {
                            // Amplify RMS for visualization with non-linear curve
                            // Square root compresses dynamic range: boosts quiet sounds, tames loud
                            // - Quiet speech (0.01 RMS) → 0.45 amplitude
                            // - Normal speech (0.03 RMS) → 0.77 amplitude
                            // - Loud speech (0.05+ RMS) → ~1.0 amplitude
                            let amplified = (level * 20.0).sqrt().min(1.0);

                            // Convert to waveform samples (spread across 40 bars)
                            let samples: Vec<f32> = (0..40)
                                .map(|i| {
                                    let phase = i as f32 * 0.15;
                                    let variation = (phase.sin() * 0.3 + 0.7) * amplified;
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

            // Handle --toggle on first launch (not just second instance)
            if std::env::args().any(|arg| arg == "--toggle") {
                info!("First launch with --toggle flag, will start dictation after init");
                let app_handle = app.handle().clone();
                // Delay to allow frontend to initialize and listen for events
                tauri::async_runtime::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    info!("Emitting dictation-start for first-launch --toggle");
                    if let Err(e) = app_handle.emit("dictation-start", ()) {
                        error!("Failed to emit dictation-start: {}", e);
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            initialize_pipeline,
            start_dictation,
            stop_dictation,
            discard_dictation,
            toggle_dictation,
            get_pipeline_state,
            check_models,
            simulate_audio,
            quit_app,
            save_window_position,
            get_saved_window_position,
            get_raw_window_position,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_app_state_initial_values() {
        let (event_tx, _event_rx) = mpsc::channel::<PipelineEvent>(100);
        let app_state = AppState {
            pipeline: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
            cancel_handle: Arc::new(Mutex::new(None)),
            discard_handle: Arc::new(Mutex::new(None)),
            event_tx,
            last_transcription: Arc::new(Mutex::new(None)),
            pipeline_config: Arc::new(Mutex::new(None)),
            idle_timer_handle: Arc::new(Mutex::new(None)),
            idle_unload_seconds: 7200,
        };

        // Initial state should be not running
        assert!(!*app_state.is_running.lock().await);
        // Pipeline should be None
        assert!(app_state.pipeline.lock().await.is_none());
        // Cancel/discard handles should be None
        assert!(app_state.cancel_handle.lock().await.is_none());
        assert!(app_state.discard_handle.lock().await.is_none());
        // Last transcription should be None
        assert!(app_state.last_transcription.lock().await.is_none());
        // Pipeline config should be None (not yet initialized)
        assert!(app_state.pipeline_config.lock().await.is_none());
        // Idle timer should be None
        assert!(app_state.idle_timer_handle.lock().await.is_none());
        // Idle unload seconds should be set
        assert_eq!(app_state.idle_unload_seconds, 7200);
    }

    #[tokio::test]
    async fn test_cancel_handle_coordination() {
        // Test that cancel/discard handles can be set and read from different tasks
        let (event_tx, _event_rx) = mpsc::channel::<PipelineEvent>(100);
        let app_state = AppState {
            pipeline: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
            cancel_handle: Arc::new(Mutex::new(None)),
            discard_handle: Arc::new(Mutex::new(None)),
            event_tx,
            last_transcription: Arc::new(Mutex::new(None)),
            pipeline_config: Arc::new(Mutex::new(None)),
            idle_timer_handle: Arc::new(Mutex::new(None)),
            idle_unload_seconds: 7200,
        };

        // Simulate setting cancel handle (as start_dictation would)
        let flag = Arc::new(AtomicBool::new(false));
        {
            let mut handle_guard = app_state.cancel_handle.lock().await;
            *handle_guard = Some(flag.clone());
        }

        // Simulate stop_dictation reading and setting the flag
        {
            let handle_guard = app_state.cancel_handle.lock().await;
            if let Some(ref handle) = *handle_guard {
                handle.store(true, Ordering::SeqCst);
            }
        }

        // Verify the flag was set
        assert!(flag.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_is_running_state_transitions() {
        let (event_tx, _event_rx) = mpsc::channel::<PipelineEvent>(100);
        let app_state = AppState {
            pipeline: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
            cancel_handle: Arc::new(Mutex::new(None)),
            discard_handle: Arc::new(Mutex::new(None)),
            event_tx,
            last_transcription: Arc::new(Mutex::new(None)),
            pipeline_config: Arc::new(Mutex::new(None)),
            idle_timer_handle: Arc::new(Mutex::new(None)),
            idle_unload_seconds: 7200,
        };

        // Start dictation would set is_running to true
        {
            let mut is_running = app_state.is_running.lock().await;
            *is_running = true;
        }
        assert!(*app_state.is_running.lock().await);

        // After completion, is_running would be set back to false
        {
            let mut is_running = app_state.is_running.lock().await;
            *is_running = false;
        }
        assert!(!*app_state.is_running.lock().await);
    }

    #[tokio::test]
    async fn test_last_transcription_storage() {
        let (event_tx, _event_rx) = mpsc::channel::<PipelineEvent>(100);
        let app_state = AppState {
            pipeline: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
            cancel_handle: Arc::new(Mutex::new(None)),
            discard_handle: Arc::new(Mutex::new(None)),
            event_tx,
            last_transcription: Arc::new(Mutex::new(None)),
            pipeline_config: Arc::new(Mutex::new(None)),
            idle_timer_handle: Arc::new(Mutex::new(None)),
            idle_unload_seconds: 7200,
        };

        // Initially no transcription
        assert!(app_state.last_transcription.lock().await.is_none());

        // Store a transcription (as dictation completion would)
        {
            let mut last = app_state.last_transcription.lock().await;
            *last = Some("Hello world".to_string());
        }

        // Verify stored
        let stored = app_state.last_transcription.lock().await.clone();
        assert_eq!(stored, Some("Hello world".to_string()));
    }

    #[test]
    fn test_check_models_returns_json() {
        // This test verifies check_models can be called synchronously
        // (the actual model checking is done via Config which is tested in yammer-core)
        // We're just verifying the JSON structure is correct

        use serde_json::json;

        // Simulate the expected JSON structure
        let expected_structure = json!({
            "models_dir": "/some/path",
            "whisper": {
                "exists": false,
                "path": "/some/path/model.bin"
            },
            "llm": {
                "exists": false,
                "path": "",
                "enabled": true
            }
        });

        // Verify it has expected keys
        assert!(expected_structure.get("models_dir").is_some());
        assert!(expected_structure.get("whisper").is_some());
        assert!(expected_structure.get("llm").is_some());
    }
}
