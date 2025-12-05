//! Yammer Tauri application

use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

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
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    // Only trigger on key press, not release
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }

                    // Check for our dictation toggle shortcut (Super+D)
                    let dictate_shortcut =
                        Shortcut::new(Some(Modifiers::SUPER), Code::KeyD);
                    if shortcut == &dictate_shortcut {
                        #[cfg(debug_assertions)]
                        println!("Dictation hotkey pressed (Super+D)");

                        // Emit event to frontend to toggle dictation
                        if let Err(e) = app.emit("dictation-toggle", ()) {
                            eprintln!("Failed to emit dictation-toggle: {}", e);
                        }
                    }
                })
                .build(),
        )
        .setup(|app| {
            // Get the main window
            let window = app.get_webview_window("main").expect("Main window not found");

            // Log window info
            #[cfg(debug_assertions)]
            {
                println!("Window created: {:?}", window.label());
            }

            // Register global hotkey: Super+D for dictation toggle
            let dictate_shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyD);
            match app.global_shortcut().register(dictate_shortcut) {
                Ok(_) => {
                    #[cfg(debug_assertions)]
                    println!("Registered global shortcut: Super+D");
                }
                Err(e) => {
                    eprintln!(
                        "Failed to register global shortcut Super+D: {}. \
                         Another application may have grabbed this key.",
                        e
                    );
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![simulate_audio])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
