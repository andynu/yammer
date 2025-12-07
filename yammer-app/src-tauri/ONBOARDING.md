# yammer-app Onboarding Guide

Welcome to yammer-app, the Tauri desktop application for the Yammer dictation system. This guide covers how the frontend and backend integrate to create the complete user experience.

## Table of Contents

1. [Conceptual Background](#conceptual-background)
2. [Library Choices](#library-choices)
3. [Architecture Overview](#architecture-overview)
4. [Code Walkthrough](#code-walkthrough)
5. [Common Tasks](#common-tasks)

---

## Conceptual Background

### What is Tauri?

[Tauri](https://tauri.app/) is a framework for building desktop applications with web frontends and Rust backends. Think of it as Electron but with Rust instead of Node.js.

```
┌───────────────────────────────────────────────┐
│                 Your Application              │
├─────────────────────┬─────────────────────────┤
│    Web Frontend     │      Rust Backend       │
│   (HTML/CSS/JS)     │        (Tauri)          │
│                     │                         │
│ • UI components     │ • System APIs           │
│ • User interactions │ • File system           │
│ • Visual feedback   │ • Native features       │
│                     │ • Heavy computation     │
└─────────────────────┴─────────────────────────┘
                   ↑↓ IPC (JSON)
```

### Tauri vs Electron

| Aspect | Tauri | Electron |
|--------|-------|----------|
| Backend | Rust | Node.js |
| Bundle size | ~10MB | ~150MB |
| Memory usage | Lower | Higher |
| Security | Stronger sandbox | Weaker |
| Ecosystem | Growing | Mature |
| Platform support | Windows/macOS/Linux | Same |

We chose Tauri because:
- **Small binaries**: Users don't need to download Chromium
- **Rust performance**: Audio processing, ML inference benefit from Rust
- **Memory efficiency**: Important for always-running background app
- **Single ecosystem**: Everything in Rust (workspace with other crates)

### Inter-Process Communication (IPC)

Tauri uses JSON-based IPC between frontend and backend:

**Frontend → Backend (Commands)**:
```typescript
// Frontend calls Rust function
const result = await invoke('start_dictation');
```

```rust
// Rust handles the call
#[tauri::command]
async fn start_dictation(...) -> Result<(), String> { ... }
```

**Backend → Frontend (Events)**:
```rust
// Rust emits event
app.emit("pipeline-state", "listening");
```

```typescript
// Frontend listens
listen('pipeline-state', (event) => {
  console.log(event.payload); // "listening"
});
```

### System Integration

Desktop apps need OS features beyond the web:

| Feature | What It Does | How We Use It |
|---------|--------------|---------------|
| **Global Shortcuts** | Keyboard shortcuts work even when app isn't focused | Super+H triggers dictation |
| **System Tray** | Icon in taskbar/menubar | Quick access, show/hide window |
| **Single Instance** | Prevent multiple app copies | Second launch shows existing window |
| **Clipboard** | System clipboard access | "Copy Last" feature |

### The tokio Runtime

Tauri uses [tokio](https://tokio.rs/) for async operations. Key concepts:

| Concept | Description |
|---------|-------------|
| **async/await** | Non-blocking operations |
| **spawn** | Run task concurrently |
| **spawn_blocking** | Run blocking code without blocking async runtime |
| **Mutex** | Async-safe locks for shared state |
| **mpsc channels** | Message passing between tasks |

---

## Library Choices

### Why Tauri 2?

Tauri 2.0 (released 2024) brings significant improvements:

| Feature | Tauri 1 | Tauri 2 |
|---------|---------|---------|
| Plugin system | Monolithic | Modular plugins |
| Mobile support | No | Yes (Android/iOS) |
| API stability | Changing | Stable |

### Tauri Plugins We Use

| Plugin | Purpose |
|--------|---------|
| `tauri-plugin-global-shortcut` | Register Super+H hotkey |
| `tauri-plugin-single-instance` | Prevent multiple instances |
| `tauri-plugin-clipboard-manager` | Copy transcription to clipboard |
| `tauri-plugin-shell` | Open external links |

### Why Svelte?

The frontend uses [Svelte](https://svelte.dev/) (via SvelteKit):

| Alternative | Why Not |
|-------------|---------|
| **React** | Larger bundle, virtual DOM overhead |
| **Vue** | Good but Svelte is simpler |
| **Vanilla JS** | No component model, harder maintenance |

Svelte advantages:
- **Small bundle**: Compiles to vanilla JS
- **Reactive**: Updates are automatic
- **Simple**: Less boilerplate than React

---

## Architecture Overview

### Project Structure

```
yammer-app/
├── src-tauri/          # Rust backend
│   ├── src/
│   │   ├── main.rs     # Entry point
│   │   ├── lib.rs      # Tauri commands and setup
│   │   └── pipeline.rs # Dictation pipeline
│   ├── Cargo.toml
│   └── tauri.conf.json
└── src/                # Svelte frontend
    ├── routes/
    │   └── +page.svelte
    └── lib/
        └── components/
```

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         yammer-app                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Frontend (Svelte)                   Backend (Rust)              │
│  ┌─────────────────┐                 ┌────────────────┐          │
│  │   Main Window   │   invoke()      │  AppState      │          │
│  │                 │────────────────▶│                │          │
│  │ • Waveform      │                 │ • pipeline     │          │
│  │ • State display │◀────────────────│ • is_running   │          │
│  │ • Controls      │   emit()        │ • cancel_handle│          │
│  └─────────────────┘                 └────────┬───────┘          │
│                                               │                  │
│  ┌─────────────────┐                 ┌────────▼───────┐          │
│  │   System Tray   │                 │DictationPipeline│         │
│  │                 │                 │                │          │
│  │ • Show/Hide     │                 │ • AudioCapture │          │
│  │ • Toggle Record │                 │ • VadProcessor │          │
│  │ • Copy Last     │                 │ • Transcriber  │          │
│  │ • Quit          │                 │ • Corrector    │          │
│  └─────────────────┘                 │ • TextOutput   │          │
│                                      └────────────────┘          │
│                                                                  │
│  ┌─────────────────┐                 ┌────────────────┐          │
│  │ Global Shortcut │                 │  Event Channel │          │
│  │                 │   emit()        │                │          │
│  │   Super+H ──────────────────────▶│ • StateChanged │          │
│  │                 │                 │ • AudioLevel   │          │
│  └─────────────────┘                 │ • Transcript   │          │
│                                      │ • Error        │          │
│                                      └────────────────┘          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### State Management

```rust
pub struct AppState {
    // The dictation pipeline (Option because not initialized at start)
    pipeline: Arc<Mutex<Option<DictationPipeline>>>,

    // Is dictation currently running?
    is_running: Arc<Mutex<bool>>,

    // Cancel/discard handles (can be used without pipeline lock)
    cancel_handle: Arc<Mutex<Option<Arc<AtomicBool>>>>,
    discard_handle: Arc<Mutex<Option<Arc<AtomicBool>>>>,

    // Channel to send events to frontend forwarder
    event_tx: mpsc::Sender<PipelineEvent>,

    // Last transcription for "Copy Last" feature
    last_transcription: Arc<Mutex<Option<String>>>,
}
```

Key design decisions:
- **Arc<Mutex<...>>**: Shared state across async tasks
- **Cancel handles**: Allow stopping without holding pipeline lock
- **Event channel**: Decouple pipeline from Tauri event emission

### Pipeline States

```
┌────────────────────────────────────────────────────────────┐
│                     Pipeline State Machine                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│    ┌──────┐   start   ┌───────────┐   speech   ┌────────┐ │
│    │ Idle │──────────▶│ Listening │───────────▶│Process.│ │
│    └──────┘           └───────────┘             └────────┘ │
│        ▲                    │                      │       │
│        │                    │ cancel               │       │
│        │                    │ (discard)            ▼       │
│        │                    ▼               ┌───────────┐  │
│        │              ┌───────────┐         │ Correcting│  │
│        │              │ Discarded │         └───────────┘  │
│        │              └───────────┘              │         │
│        │                                         ▼         │
│        │  ┌──────┐                         ┌────────┐      │
│        └──│ Done │◀────────────────────────│ Output │      │
│           └──────┘                         └────────┘      │
│                                                            │
│    Error can occur at any stage → PipelineState::Error     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Key Types

```rust
// Pipeline configuration
pub struct PipelineConfig {
    pub whisper_model_path: PathBuf,
    pub llm_model_path: Option<PathBuf>,
    pub use_llm_correction: bool,
    pub output_method: OutputMethod,
    pub typing_delay_ms: u32,
    pub vad_threshold: f64,
    pub audio_device: Option<String>,
}

// Events sent to frontend
pub enum PipelineEvent {
    StateChanged(PipelineState),
    AudioLevel(f32),
    Transcript { text: String, is_partial: bool },
    Error(String),
}

// Pipeline states
pub enum PipelineState {
    Idle,
    Listening,
    Processing,
    Correcting,
    Done,
    Error,
    Discarded,
}
```

---

## Code Walkthrough

### Tauri Command: Start Dictation (lib.rs:119-207)

```rust
#[tauri::command]
async fn start_dictation(
    state: State<'_, AppState>,
    _app: AppHandle,
) -> Result<(), String> {
    // 1. Check if already running
    {
        let mut is_running = state.is_running.lock().await;
        if *is_running {
            return Err("Dictation already in progress".to_string());
        }

        // Verify pipeline is initialized
        let pipeline_guard = state.pipeline.lock().await;
        match pipeline_guard.as_ref() {
            Some(p) if p.is_initialized() => {
                // Get cancel/discard handles
                *state.cancel_handle.lock().await = Some(p.get_cancel_handle());
                *state.discard_handle.lock().await = Some(p.get_discard_handle());
                *is_running = true;
            }
            _ => return Err("Pipeline not initialized".to_string()),
        }
    }

    // 2. Run pipeline in blocking task (audio capture isn't Send-safe)
    let pipeline_state = state.pipeline.clone();
    let is_running_state = state.is_running.clone();
    // ... clone other state handles

    tokio::task::spawn_blocking(move || {
        // Block on async mutex
        let rt = tokio::runtime::Handle::current();
        let pipeline_guard = rt.block_on(pipeline_state.lock());

        let result = if let Some(ref pipeline) = *pipeline_guard {
            pipeline.run_blocking()
        } else {
            Err("Pipeline not available".to_string())
        };

        // Cleanup: mark as not running
        rt.block_on(async {
            *is_running_state.lock().await = false;
        });
    });

    Ok(())
}
```

Key patterns:
- **Combined lock acquisition**: Check + set in single lock scope
- **spawn_blocking**: Audio capture needs sync context
- **Handle cloning**: Pass handles to spawned task

### Event Forwarding (lib.rs:597-629)

```rust
// In setup()
tauri::async_runtime::spawn(async move {
    while let Some(event) = event_rx.recv().await {
        match event {
            PipelineEvent::StateChanged(state) => {
                let _ = app_handle.emit("pipeline-state", state.as_str());
            }
            PipelineEvent::AudioLevel(level) => {
                // Non-linear amplification: sqrt compresses dynamic range
                // Boosts quiet sounds, tames loud ones
                let amplified = (level * 20.0).sqrt().min(1.0);

                // Convert to waveform samples
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
```

Design:
- **Separate task**: Event forwarding runs independently
- **Transform data**: RMS levels converted to waveform for UI
- **Non-blocking emit**: Uses `try_send` / ignores errors

### Global Hotkey Handler (lib.rs:437-492)

```rust
.plugin(
    tauri_plugin_global_shortcut::Builder::new()
        .with_handler(|app, shortcut, event| {
            // Only trigger on key press, not release
            if event.state() != ShortcutState::Pressed {
                return;
            }

            let dictate_shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyH);
            if shortcut == &dictate_shortcut {
                // Check if window was hidden
                let was_hidden = if let Some(window) = app.get_webview_window("main") {
                    let was_visible = window.is_visible().unwrap_or(false);
                    let was_minimized = window.is_minimized().unwrap_or(false);

                    if !was_visible || was_minimized {
                        let _ = window.unminimize();
                        let _ = window.show();
                        let _ = window.set_always_on_top(true);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };

                // Emit appropriate event
                if was_hidden {
                    let _ = app.emit("dictation-start", ());
                } else {
                    let _ = app.emit("dictation-toggle", ());
                }
            }
        })
        .build(),
)
```

Behavior:
- **Super+H pressed**: Check window visibility
- **Window hidden**: Show window + start dictation
- **Window visible**: Toggle dictation state

### Pipeline Blocking Run (pipeline.rs:294-391)

```rust
pub fn run_blocking(&self) -> Result<String, String> {
    self.reset_cancel();

    // 1. Listen for speech
    let samples = match self.listen_blocking() {
        Ok(s) => s,
        Err(e) if e == "Discarded" => {
            self.send_state(PipelineState::Discarded);
            return Err(e);
        }
        Err(e) => {
            self.send_error(e.clone());
            self.send_state(PipelineState::Error);
            return Err(e);
        }
    };

    // 2. Transcribe
    let text = self.transcribe_blocking(&samples)?;

    // 3. Correct (optional)
    let corrected = match self.correct_blocking(&text) {
        Ok(t) => t,
        Err(e) if e == "Cancelled" => {
            // On cancel during correction, output uncorrected
            self.output_blocking(&text)?;
            return Ok(text);
        }
        Err(e) => {
            // On error, output uncorrected
            self.output_blocking(&text)?;
            return Ok(text);
        }
    };

    // 4. Output text
    self.output_blocking(&corrected)?;
    Ok(corrected)
}
```

Graceful degradation:
- Cancel during correction → output raw transcription
- Correction error → output raw transcription
- Discard → no output at all

---

## Common Tasks

### Add a New Tauri Command

1. Define the command function in `lib.rs`:

```rust
/// Description of what this does
#[tauri::command]
async fn my_new_command(
    state: State<'_, AppState>,
    app: AppHandle,
    param: String,
) -> Result<String, String> {
    // Implementation
    Ok("result".to_string())
}
```

2. Register in the invoke handler:

```rust
.invoke_handler(tauri::generate_handler![
    // ... existing commands
    my_new_command,
])
```

3. Call from frontend:

```typescript
import { invoke } from '@tauri-apps/api/core';

const result = await invoke('my_new_command', { param: 'value' });
```

### Add a New Event

1. Add to PipelineEvent enum (if pipeline-related):

```rust
pub enum PipelineEvent {
    // ... existing events
    MyNewEvent { data: String },
}
```

2. Send from pipeline:

```rust
let _ = self.event_tx.try_send(PipelineEvent::MyNewEvent {
    data: "value".to_string()
});
```

3. Forward to frontend in the event loop:

```rust
PipelineEvent::MyNewEvent { data } => {
    let _ = app_handle.emit("my-new-event", data);
}
```

4. Listen in frontend:

```typescript
import { listen } from '@tauri-apps/api/event';

await listen('my-new-event', (event) => {
    console.log(event.payload);
});
```

### Modify the Pipeline Flow

The pipeline runs in stages in `run_blocking()`:

1. **listen_blocking()**: Capture audio until cancel
2. **transcribe_blocking()**: Convert audio to text
3. **correct_blocking()**: LLM improvement (optional)
4. **output_blocking()**: Type or paste text

To add a stage:

```rust
pub fn run_blocking(&self) -> Result<String, String> {
    // ... existing stages

    // New stage after transcription
    let enhanced = self.enhance_blocking(&text)?;

    // Continue with correction
    let corrected = self.correct_blocking(&enhanced)?;

    // ...
}

fn enhance_blocking(&self, text: &str) -> Result<String, String> {
    self.send_state(PipelineState::Enhancing);

    if self.is_cancelled() {
        return Err("Cancelled".to_string());
    }

    // Your logic here

    Ok(enhanced_text)
}
```

Remember to add the new state to `PipelineState` enum.

### Debug IPC Issues

1. **Check Rust logs**:
   ```bash
   RUST_LOG=debug cargo tauri dev
   ```

2. **Check frontend console** (DevTools F12):
   ```typescript
   import { listen } from '@tauri-apps/api/event';

   // Log all events
   listen('*', (event) => {
       console.log('Event:', event.event, event.payload);
   });
   ```

3. **Verify command registration**:
   - Ensure command is in `generate_handler![]`
   - Check parameter names match frontend `invoke()` call
   - Verify return type is `Result<T, String>`

4. **Test commands directly**:
   ```typescript
   // In browser console
   __TAURI__.invoke('get_pipeline_state')
       .then(console.log)
       .catch(console.error);
   ```

### Add System Tray Menu Item

```rust
// In setup() where tray is created:
let new_item = MenuItem::with_id(app, "my_action", "My Action", true, None::<&str>)?;
let menu = Menu::with_items(app, &[
    // ... existing items
    &new_item,
])?;

// In on_menu_event handler:
.on_menu_event(|app, event| match event.id.as_ref() {
    // ... existing handlers
    "my_action" => {
        info!("Tray: My Action clicked");
        // Your logic here
    }
    _ => {}
})
```

### Handle Single-Instance Arguments

```rust
.plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
    info!("Second instance launched with args: {:?}", argv);

    // Check for specific flags
    if argv.iter().any(|arg| arg == "--my-flag") {
        // Handle the flag
        if let Err(e) = app.emit("my-flag-event", ()) {
            error!("Failed to emit event: {}", e);
        }
    }
}))
```

---

## Configuration

### tauri.conf.json Key Settings

```json
{
  "app": {
    "windows": [{
      "title": "Yammer",
      "width": 320,
      "height": 200,
      "resizable": false,
      "alwaysOnTop": true,
      "decorations": false
    }],
    "trayIcon": {
      "iconPath": "icons/icon.png"
    }
  },
  "bundle": {
    "identifier": "com.yammer.app"
  }
}
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `RUST_LOG` | Logging level (debug, info, warn, error) |
| `DISPLAY` | X11 display (for Wayland compatibility) |

---

## Summary

yammer-app is the Tauri desktop application that:

1. **Integrates all crates**: audio, STT, LLM, output
2. **Provides UI**: Web frontend with Svelte
3. **System integration**: Hotkeys, tray, single-instance
4. **Event-driven**: Pipeline events forwarded to frontend

Key files:
- `src-tauri/src/lib.rs`: Tauri commands, setup, event forwarding
- `src-tauri/src/pipeline.rs`: Dictation pipeline orchestration
- `src/routes/+page.svelte`: Main UI component

Architecture patterns:
- **Arc<Mutex<...>>**: Shared state across async tasks
- **Cancel handles**: Stop operations without holding locks
- **spawn_blocking**: Run sync code in async context
- **Event channels**: Decouple pipeline from UI updates
