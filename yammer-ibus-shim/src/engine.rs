//! Yammer's IBus engine: a pure passthrough for normal typing, plus a way to
//! commit dictated text into whichever input context last had focus.
//!
//! `process_key_event` is never overridden — `IBusEngine`'s default
//! implementation already returns `Ok(false)` for every key, which IBus
//! relays straight back to the client as "handle this yourself" (verified
//! against `bus/inputcontext.c` in upstream IBus). Yammer's dictation hotkey
//! is a compositor-level global shortcut, outside this pipeline entirely, so
//! there is nothing else for this engine to intercept.
//!
//! librush 0.2.3's `Engine::<T>::new` registers every instance at the same
//! hardcoded object path (its own `bus.rs` doc comment: "only support 1
//! engine now"). So in practice there is at most one live engine object no
//! matter how many input contexts enable "yammer" — `FocusTracker` reflects
//! that: it is last-writer-wins, not a per-context map. Fine for the common
//! case (one focused context at a time); documented as a known limitation in
//! plans/002-ibus-input-engine.md rather than worked around, since fixing it
//! would mean modifying librush, which CLAUDE.local.md forbids.

use std::sync::{Arc, Mutex, OnceLock};

use librush::ibus::{IBusEngine, IBusEngineBackend, IBusFactory};
use zbus::object_server::SignalEmitter;
use zbus::zvariant::OwnedObjectPath;
use zbus::{Connection, ObjectServer, fdo};

#[derive(Default)]
struct FocusState {
    /// Updated live by `focus_in`/`focus_out` as contexts gain/lose focus.
    current: Option<OwnedObjectPath>,
    /// Frozen by an explicit `capture()` call and left alone afterwards —
    /// this is "where the cursor was when dictation was invoked", which may
    /// differ from `current` by the time `commit_text` runs.
    captured: Option<OwnedObjectPath>,
    /// Bumped by every `focus_in`. GNOME Shell routes every client through
    /// one engine path, so "focus came back" can't be read from the path.
    focus_generation: u64,
}

/// Shared handle to "whichever engine instance last had focus". Cheap to
/// clone; every `YammerEngine`/`YammerFactory` instance holds one so they all
/// observe the same state.
///
/// There's exactly one D-Bus `Connection` for the process's whole lifetime
/// (set once via `set_connection` right after connecting), so only the
/// object *path* needs to be tracked per focus event — which also makes the
/// focus/capture state machine testable with no real D-Bus connection.
#[derive(Clone, Default)]
pub struct FocusTracker {
    conn: Arc<OnceLock<Connection>>,
    state: Arc<Mutex<FocusState>>,
}

impl FocusTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set once, right after connecting to `ibus-daemon`. Calling this more
    /// than once is a logic error (ignored past the first call).
    pub fn set_connection(&self, conn: Connection) {
        let _ = self.conn.set(conn);
    }

    /// Snapshot whatever currently has focus into the `captured` slot. Call
    /// this once, at dictation-invocation time — the counterpart of today's
    /// `capture_active_window()`. Returns `false` if nothing is focused.
    pub fn capture(&self) -> bool {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.captured = guard.current.clone();
        guard.captured.is_some()
    }

    /// The path captured by the most recent `capture()` call, plus the
    /// process's single connection — used at commit time regardless of what
    /// currently has focus. `None` if `set_connection` was never called or
    /// nothing was ever captured.
    fn captured(&self) -> Option<(Connection, OwnedObjectPath)> {
        let path = self.state.lock().unwrap_or_else(|e| e.into_inner()).captured.clone()?;
        let conn = self.conn.get()?.clone();
        Some((conn, path))
    }

    /// Number of `focus_in` events seen so far.
    pub fn focus_generation(&self) -> u64 {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).focus_generation
    }

    fn set_focused(&self, path: OwnedObjectPath) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        guard.current = Some(path);
        guard.focus_generation += 1;
    }

    fn clear_if(&self, path: &OwnedObjectPath) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if guard.current.as_ref() == Some(path) {
            guard.current = None;
        }
    }
}

#[derive(Clone)]
pub struct YammerEngine {
    focus: FocusTracker,
}

impl YammerEngine {
    pub fn new(focus: FocusTracker) -> Self {
        Self { focus }
    }
}

impl IBusEngine for YammerEngine {
    async fn focus_in(&mut self, se: SignalEmitter<'_>, _server: &ObjectServer) -> fdo::Result<()> {
        self.focus.set_focused(se.path().to_owned().into());
        Ok(())
    }

    async fn focus_out(&mut self, se: SignalEmitter<'_>, _server: &ObjectServer) -> fdo::Result<()> {
        self.focus.clear_if(&se.path().to_owned().into());
        Ok(())
    }

    // process_key_event, reset, enable, disable: intentionally left as the
    // trait's defaults.
}

#[derive(Clone)]
pub struct YammerFactory {
    focus: FocusTracker,
}

impl YammerFactory {
    pub fn new(focus: FocusTracker) -> Self {
        Self { focus }
    }
}

impl IBusFactory<YammerEngine> for YammerFactory {
    fn create_engine(&mut self, name: String) -> Result<YammerEngine, String> {
        if name != crate::bus::ENGINE_NAME {
            return Err(format!("unknown engine name: {name}"));
        }
        Ok(YammerEngine::new(self.focus.clone()))
    }
}

/// Commit `text` into the instance snapshotted by the most recent
/// `FocusTracker::capture()` call, regardless of whether that instance still
/// has focus right now — matching the capture-at-invocation,
/// use-at-completion semantics of the legacy `target_window` path.
pub async fn commit_text(focus: &FocusTracker, text: String) -> Result<(), String> {
    let Some((conn, path)) = focus.captured() else {
        return Err("no target was captured before this dictation session".to_string());
    };
    let se = SignalEmitter::from_parts(conn, path.into());
    YammerEngine::commit_text(&se, text)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path(s: &str) -> OwnedObjectPath {
        OwnedObjectPath::try_from(s).unwrap()
    }

    #[test]
    fn capture_returns_false_when_nothing_focused() {
        let focus = FocusTracker::new();
        assert!(!focus.capture());
    }

    #[test]
    fn focus_in_then_capture_snapshots_the_path() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        assert!(focus.capture());
    }

    #[test]
    fn focus_out_for_a_different_path_does_not_clear_current() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        focus.clear_if(&path("/org/freedesktop/IBus/Engine/2"));
        assert!(focus.capture(), "focus_out for an unrelated path must not clear the real one");
    }

    #[test]
    fn focus_out_for_the_current_path_clears_it() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        focus.clear_if(&path("/org/freedesktop/IBus/Engine/1"));
        assert!(!focus.capture());
    }

    #[test]
    fn capture_is_frozen_even_if_focus_moves_on_afterwards() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        assert!(focus.capture());

        // Focus moves elsewhere after the capture — e.g. the user alt-tabbed
        // mid-dictation. The captured snapshot must not change.
        focus.clear_if(&path("/org/freedesktop/IBus/Engine/1"));
        focus.set_focused(path("/org/freedesktop/IBus/Engine/2"));

        let captured = focus.state.lock().unwrap().captured.clone();
        assert_eq!(captured, Some(path("/org/freedesktop/IBus/Engine/1")));
    }

    #[test]
    fn focus_generation_advances_on_every_focus_in() {
        let focus = FocusTracker::new();
        let start = focus.focus_generation();

        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        let after_first = focus.focus_generation();
        assert!(after_first > start);

        // Under GNOME Shell every context shares one engine path, so a
        // refocus to the same path must still count as new focus.
        focus.clear_if(&path("/org/freedesktop/IBus/Engine/1"));
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        assert!(focus.focus_generation() > after_first);
    }

    #[test]
    fn focus_out_does_not_advance_focus_generation() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        let before = focus.focus_generation();
        focus.clear_if(&path("/org/freedesktop/IBus/Engine/1"));
        assert_eq!(focus.focus_generation(), before);
    }

    #[test]
    fn commit_without_a_connection_fails_even_after_capture() {
        let focus = FocusTracker::new();
        focus.set_focused(path("/org/freedesktop/IBus/Engine/1"));
        focus.capture();
        // No set_connection() call — captured() must return None, not panic.
        assert!(focus.captured().is_none());
    }
}
