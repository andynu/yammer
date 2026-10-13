# IBus Input Engine — Replace xdotool/xclip Output with a Real Input Method

## Executive Summary

Replace Yammer's X11-only output path (`xdotool type`, `xclip` + uinput Shift+Insert,
`xdotool windowactivate` refocus) with a proper IBus input-method engine written in Rust
over `zbus`. Motivation: the xdotool path is X11-native and only works on Wayland via
portal dialogs or the uinput workaround already in `virtual_keyboard.rs` — it does not
give reliable Wayland support. Becoming a real IBus engine means the compositor/toolkit
hands us focus natively; there is no window to refocus and no keystroke simulation.

**Key decisions:**
- Depend on `librush` (crates.io, LGPL-2.1-or-later OR GPL-3.0-or-later) for the
  zbus/IBus protocol plumbing — connection setup, the `Engine`/`Factory` D-Bus
  interfaces, and the GVariant serialization of `IBusText`. Verified directly against
  IBus's own C source (`ibus/src/ibusengine.c`, `ibus/src/ibusserializable.c`,
  `ibus/bus/inputcontext.c`) — see Decision Log.
- **Hard constraint** (recorded in `CLAUDE.local.md`): `librush` is used unmodified and
  only ever compiled into a dedicated `cdylib` (`yammer-ibus-shim`), loaded at runtime
  via `libloading`. No crate outside `yammer-ibus-shim` may depend on `librush` directly.
  This keeps Yammer's MIT code from being a combined work under LGPL/GPL reciprocity.
- The engine's `ProcessKeyEvent` always returns `false` (pure passthrough). Confirmed
  from `bus/inputcontext.c`: the daemon relays our `false` straight back to the client,
  which then handles the keystroke itself exactly as if no IME were active. Yammer's
  existing Super+H global shortcut is a compositor-level grab, outside the IBus key
  pipeline entirely, so it needs no changes.
- Activation model: Yammer becomes a resident/default IBus input source (added once via
  GNOME Input Sources, same kind of one-time setup as today's `GNOME_SETUP.md` custom
  shortcut). While active, it tracks `FocusIn`/`FocusOut` to know which engine instance
  — i.e. which text field — last had focus, exactly standing in for today's
  `capture_active_window()`. The hotkey toggle, recording UI, and STT pipeline are
  unchanged; only the output seam changes.
- Keep the existing xdotool/xclip/uinput path as a fallback, not a parallel supported
  method: used only when the shim fails to load, IBus isn't reachable, or a commit call
  errors. Logged as a warning, not surfaced as a user-facing option.

---

## Technical Approach

### Architecture

```
yammer-app (hotkey, recording UI, STT pipeline)      — unchanged
        │
        ▼
yammer-ibus (MIT, ordinary rlib)                      — new
  - dlopen's libyammer_ibus_shim.so via `libloading`
  - safe Rust API: try_start(), capture_target(), commit_text()
  - IbusTarget is an opaque token; falls back to None if the shim
    isn't present/loadable so callers degrade gracefully
        │  (runtime dlopen, NOT a Cargo dependency edge)
        ▼
yammer-ibus-shim (MIT, cdylib)                        — new
  - the ONLY crate that names `librush` in Cargo.toml
  - implements librush's IBusEngine + IBusFactory traits for Yammer
  - owns a background Tokio runtime + the focus-token registry
  - exposes a 3-function extern "C" surface
        │  (zbus / D-Bus, real IBus protocol)
        ▼
ibus-daemon  ──────────────────────────────────────────  the focused app's
                                                           GtkIMContext / etc.
```

`yammer-output` keeps its current Linux backend (`xdotool`/`xclip`/`virtual_keyboard.rs`)
untouched, as the fallback path.

### Crate additions

| Crate | Type | Depends on `librush`? | Purpose |
|---|---|---|---|
| `yammer-ibus` | rlib | No (dlopen only) | Safe wrapper the rest of Yammer calls |
| `yammer-ibus-shim` | cdylib | Yes — only place it's named | Engine/Factory impl, owns D-Bus connection |

Add both to the workspace `[workspace] members` in the root `Cargo.toml`.

---

## Implementation Steps

### 1. `yammer-ibus-shim` — the engine itself

#### 1.1 `yammer-ibus-shim/Cargo.toml`
```toml
[package]
name = "yammer-ibus-shim"
version.workspace = true
edition.workspace = true
license = "MIT"

[lib]
crate-type = ["cdylib"]

[dependencies]
librush = "0.2"
tokio = { version = "1", features = ["rt-multi-thread"] }
```

#### 1.2 `yammer-ibus-shim/src/engine.rs` — `YammerEngine` / `YammerFactory`
- `struct YammerEngine`: holds a shared `Arc<FocusRegistry>` (`Mutex<HashMap<u64,
  ObjectPath>>` + `AtomicU64` for `current_focus_token` and `next_token`)
- `impl IBusEngine for YammerEngine`:
  - `focus_in`: allocate a token, record this instance's object path, set
    `current_focus_token`; invoke the focus-change callback registered by `start()`
  - `focus_out`: if this instance is the current focus, clear it; invoke the callback
  - `process_key_event`: always `Ok(false)` — no exceptions. This is the whole
    passthrough contract; do not add key interception here even for the toggle hotkey.
  - `enable`/`disable`/`reset`: no-ops for now (log at debug)
- `impl IBusFactory<YammerEngine> for YammerFactory`: `create_engine("yammer")` returns a
  fresh `YammerEngine` sharing the same `FocusRegistry`

#### 1.3 `yammer-ibus-shim/src/ffi.rs` — the `extern "C"` boundary
Three functions, kept deliberately minimal:
```rust
#[no_mangle]
pub extern "C" fn yammer_ibus_shim_start(
    focus_cb: extern "C" fn(bool),
    log_cb: extern "C" fn(*const c_char),
) -> i32;   // 0 = ok; negative = error (no IBUS_ADDRESS, connect failed, name taken, ...)

#[no_mangle]
pub extern "C" fn yammer_ibus_shim_capture_target() -> u64;  // 0 = nothing focused

#[no_mangle]
pub extern "C" fn yammer_ibus_shim_commit_text(
    target: u64, text_ptr: *const u8, text_len: usize,
) -> i32;   // 0 = ok; negative = stale target, dbus error, etc.
```
`start()` spawns a dedicated `tokio::runtime::Runtime` on a background thread, connects
via `librush`'s address lookup (`ibus::addr::get_ibus_addr()`), registers the factory,
and calls `request_name("org.freedesktop.IBus.Yammer")`. `commit_text` looks up the
token in the registry and fires `Engine::<YammerEngine>::commit_text` on that specific
object path — regardless of whether it is the *current* focus, matching today's
capture-once/use-later semantics for `target_window`.

### 2. `yammer-ibus` — the safe wrapper

#### 2.1 `yammer-ibus/Cargo.toml` — depends on `libloading`, NOT `librush`

#### 2.2 `yammer-ibus/src/lib.rs`
```rust
pub struct IbusEngine { /* holds the loaded Library + resolved symbols */ }
pub struct IbusTarget(u64);

impl IbusEngine {
    pub fn try_start() -> Option<Self>;              // None if .so missing/load fails
    pub fn capture_target(&self) -> Option<IbusTarget>;
    pub fn commit_text(&self, target: &IbusTarget, text: &str) -> Result<(), IbusError>;
}
```
Locate `libyammer_ibus_shim.so` next to the running executable first, then fall back to
the normal dynamic linker search path — mirrors how the app already locates its own
bundled resources.

### 3. Component registration

#### 3.1 `yammer-ibus-shim/yammer.xml` (or a shared `data/` dir)
```xml
<?xml version="1.0" encoding="utf-8"?>
<component>
    <name>org.freedesktop.IBus.Yammer</name>
    <description>Yammer dictation input method</description>
    <exec>/usr/bin/yammer-app --ibus-engine</exec>
    <version>0.3.0</version>
    <author>Andy</author>
    <license>MIT</license>
    <homepage>https://github.com/andy/yammer</homepage>
    <textdomain>yammer</textdomain>
    <engines>
        <engine>
            <name>yammer</name>
            <longname>Yammer Dictation</longname>
            <description>Passes typing through unchanged; commits dictated text on toggle</description>
            <language>en</language>
            <license>MIT</license>
            <author>Andy</author>
            <layout>us</layout>
            <rank>0</rank>
        </engine>
    </engines>
</component>
```
Install to `/usr/share/ibus/component/yammer.xml` — add it to the Tauri bundler's deb
resources list alongside the existing `.desktop` file and icons (check
`yammer-app/src-tauri/tauri.conf.json`'s `bundle` section for the exact mechanism used
today).

### 4. Wiring into `yammer-app/src-tauri/src/lib.rs`

Four call sites currently call `capture_active_window()` (hotkey handler, menu "Toggle
Recording" item, the two `--toggle`/first-launch paths — confirmed via grep, approx.
lines 111, 767, 827, 929). Replace with a combined capture that tries IBus first:

```rust
struct DictationTarget {
    window_id: Option<String>,   // xdotool fallback, captured as today
    ibus_target: Option<yammer_ibus::IbusTarget>,
}
fn capture_dictation_target(ibus: &Option<yammer_ibus::IbusEngine>) -> DictationTarget { ... }
```

- Add a `--ibus-engine` CLI flag parsed the same way `--toggle` is today (around line
  1029). On a fresh launch with this flag: start the full singleton (hotkey, tray, etc.)
  exactly as a normal launch would, in addition to calling `IbusEngine::try_start()`.
- On a *second-instance* launch with `--ibus-engine` (i.e. `ibus-daemon` exec'd us while
  Yammer was already running): forward it to the running singleton over the existing
  `tauri_plugin_single_instance` IPC, same as `--toggle` does, so the already-running
  process starts the IBus connection if it hasn't already.
- Attempt `IbusEngine::try_start()` unconditionally at startup (not only when
  `--ibus-engine` is passed) so Super+H works correctly regardless of which path first
  launched the process. Store the `Option<IbusEngine>` in Tauri managed state.

### 5. Wiring into `yammer-app/src-tauri/src/pipeline.rs`

- `DictationPipeline.target_window: Option<String>` becomes `target: DictationTarget`
  (both fields from step 4), set via a renamed `set_target()`.
- `output_blocking` (line 738): try `ibus_target` + `IbusEngine::commit_text()` first; on
  `None`/`Err`, fall through to the existing `TextOutput::with_options(...)
  .with_target_window(...)` path unchanged. Log at `warn!` on fallback so degraded
  operation is visible without being noisy on every successful commit.

### 6. Setup documentation

- New `docs/IBUS_SETUP.md` (mirrors `docs/GNOME_SETUP.md`'s style): adding "Yammer
  Dictation" via Settings → Keyboard → Input Sources, and — important — setting it as
  the sole/default source rather than one of several, since FocusIn only fires for the
  currently-selected source. Note the one-time nature, same as the Super+H shortcut
  setup.

---

## Testing Strategy

### Unit
- `YammerEngine::process_key_event` always returns `false` for arbitrary keyval/keycode/
  state combinations (property-style test, not just one example)
- Focus registry: `focus_in` → token issued → `focus_out` on a *different* token is a
  no-op on `current_focus_token`; `focus_out` on the current one clears it
- `yammer-ibus`: `try_start()` returns `None` (not a panic/error) when the `.so` is
  absent — exercise by pointing the search path at an empty temp dir

### Integration (needs a real `ibus-daemon` — already installed on iris)
- Register the component, switch input source to "Yammer" via `ibus engine
  org.freedesktop.IBus.Yammer` or the gsettings input-sources key, focus a scratch GTK
  text entry, call `commit_text` via the FFI surface directly, assert the text lands
- **Open question to verify empirically, not assume:** commit behavior when the
  previously-focused context has since lost focus (user alt-tabbed mid-dictation). Test
  by focusing entry A, capturing its target, switching focus to entry B, then
  committing to A's captured token — does GTK's IBus im-context still insert it?
- Normal typing while Yammer is the active source is completely unaffected (type a
  sentence with Yammer selected, compare to typing with it not selected)

### Manual
1. Set Yammer as the sole input source, dictate via Super+H into a GNOME Terminal
   (no IBus support in some terminals — confirms whether the xdotool fallback kicks in
   as expected there) and into a GTK app (e.g. gedit/Text Editor)
2. Repeat under Wayland and under X11 — this is the actual motivating scenario
3. Kill `ibus-daemon` entirely, confirm `IbusEngine::try_start()` fails gracefully and
   dictation still works via the xdotool fallback

---

## Acceptance Criteria

- [ ] Dictation output lands correctly in a GTK app under a Wayland session, with no
      portal approval dialog
- [ ] Normal typing is unaffected while Yammer is the active/default input source
- [ ] `librush` appears in exactly one `Cargo.toml` (`yammer-ibus-shim`), and that
      crate is the only `cdylib` depending on it — verify with `cargo tree -i librush`
- [ ] Killing `ibus-daemon` (or never installing the component) degrades to the
      existing xdotool/xclip path without a crash
- [ ] `cargo test` passes across the workspace, including the new crates

---

## Known Risks and Open Questions

| Risk | Mitigation |
|---|---|
| Rust has no stable ABI across a `cdylib` boundary even within one crate graph | Keep the `extern "C"` surface tiny (3 functions, primitive types only) and build both sides from the same workspace/toolchain in one build, rather than treating the `.so` as independently versioned |
| `librush` is a young, single-maintainer crate (2 releases) | Pin an exact version; re-verify the GVariant serialization test (`ibus_text_zvariant_signature`) still matches upstream IBus on any version bump |
| GTK commit-to-unfocused-context behavior is unverified | Test explicitly (see Testing Strategy); if it silently drops text, the mitigation is capturing a *fresh* target at commit time instead of invocation time — a real UX change from today's "paste where I was when I started," worth a follow-up conversation with Andy if it comes to that |
| Some clients don't implement an IBus input context at all | xdotool/xclip fallback stays in the tree for exactly this |
| Setting Yammer as the *sole* input source risks losing dead-key/compose support if the user relies on a non-`us` base layout | Document the `<layout>` field's role clearly in `docs/IBUS_SETUP.md`; this is a one-time setup choice Andy makes, not something the code needs to handle |

---

## Decision Log

- **2026-10-13** — Motivation confirmed: current xdotool/xclip/uinput path does not
  give real Wayland support; that's the reason for this change, not a general dislike
  of the old code.
- **2026-10-13** — Protocol verified directly against IBus's own C source (cloned
  `github.com/ibus/ibus` to scratch for reference): `ProcessKeyEvent` returning `false`
  is relayed to the client as-is (`bus/inputcontext.c`), and `CommitText` is a plain
  D-Bus signal carrying a serialized `IBusText` GVariant tuple
  (`src/ibusengine.c`, `src/ibusserializable.c`, `src/ibustext.c`).
- **2026-10-13** — Chose to depend on `librush` (cloned `github.com/fm-elpac/librush`)
  rather than hand-roll the protocol glue: it already implements the `IBusEngine`/
  `IBusFactory` traits and the `IBusText` GVariant encoding, with a unit test
  (`ibus_text_zvariant_signature`) pinning the exact signature we independently derived
  from the C source.
- **2026-10-13** — `librush` is LGPL-2.1-or-later OR GPL-3.0-or-later; Yammer is MIT.
  Decided: use it unmodified, dynamic-linking only, via a dedicated `cdylib` +
  `libloading` rather than a normal Cargo dependency edge from any MIT-licensed crate.
  Recorded as a hard constraint in `CLAUDE.local.md` and in `THIRD_PARTY_NOTICES.md`.
- **2026-10-13** — Activation model: resident/default input source, Super+H unchanged
  (it's a compositor-level global shortcut, not an IBus key event), capture-target-at-
  invocation semantics carried over unchanged from `capture_active_window()`.
