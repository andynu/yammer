//! Yammer Tauri application

use tauri::{AppHandle, Emitter, Manager};

/// Simulate audio waveform data for testing
/// In production, this will be replaced with real audio from yammer-audio
#[tauri::command]
fn simulate_audio(app: AppHandle) -> Result<(), String> {
    // Generate fake audio samples (40 bars of random amplitudes)
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
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // Get the main window
            let window = app.get_webview_window("main").expect("Main window not found");

            // Log window info
            #[cfg(debug_assertions)]
            {
                println!("Window created: {:?}", window.label());
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![simulate_audio])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
