//! Safe wrapper around `yammer-ibus-shim`, dlopen'd at runtime via
//! `libloading`.
//!
//! `yammer-ibus-shim` is a separate `cdylib` because it's the only place in
//! the workspace allowed to depend on `librush` (LGPL) — see
//! CLAUDE.local.md. This crate deliberately never names `librush` and never
//! statically links the shim; `IbusEngine::try_start()` returns `None`
//! whenever the `.so` is missing, mismatched, or `ibus-daemon` isn't
//! reachable, so callers can fall back to the legacy xdotool/xclip path.

use std::ffi::c_char;
use std::sync::OnceLock;

use libloading::Library;
use tracing::warn;

const SHIM_LIB_NAME: &str = "libyammer_ibus_shim.so";

type StartFn = unsafe extern "C" fn(extern "C" fn(*const c_char)) -> i32;
type CaptureTargetFn = unsafe extern "C" fn() -> i32;
type CommitTextFn = unsafe extern "C" fn(*const u8, usize) -> i32;

/// A loaded, started shim. Symbols are re-resolved on each call rather than
/// cached — this is called at most once or twice per dictation toggle, not a
/// hot path, and re-resolving avoids a self-referential-struct lifetime
/// transmute for what would be a negligible saving.
pub struct IbusEngine {
    lib: Library,
}

/// Target captured by [`IbusEngine::capture_target`], to be redeemed later by
/// [`IbusEngine::commit_text`]. Carries no data of its own — the shim keeps
/// the real state — it only proves a capture happened.
pub struct IbusTarget(());

#[derive(Debug)]
pub enum IbusError {
    NothingCaptured,
    CommitFailed,
    Other(i32),
}

impl std::fmt::Display for IbusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IbusError::NothingCaptured => write!(f, "no target was captured before this session"),
            IbusError::CommitFailed => write!(f, "ibus commit_text call failed"),
            IbusError::Other(code) => write!(f, "yammer-ibus-shim returned error code {code}"),
        }
    }
}

impl std::error::Error for IbusError {}

extern "C" fn log_from_shim(msg: *const c_char) {
    if msg.is_null() {
        return;
    }
    // SAFETY: the shim only ever passes a valid, null-terminated C string
    // that it owns for the duration of this call.
    let msg = unsafe { std::ffi::CStr::from_ptr(msg) }.to_string_lossy();
    tracing::info!(target: "yammer_ibus_shim", "{msg}");
}

impl IbusEngine {
    /// Load the shim and connect to `ibus-daemon`. Returns `None` on any
    /// failure — missing `.so`, symbol mismatch, no `ibus-daemon` reachable
    /// — logging the reason at `warn` so it's visible without being fatal.
    pub fn try_start() -> Option<Self> {
        // SAFETY: we only call functions resolved by name below, matching
        // the exact signatures declared in yammer-ibus-shim/src/ffi.rs, and
        // we keep `Library` alive for this struct's whole lifetime.
        let lib = match unsafe { Self::load_library() } {
            Ok(lib) => lib,
            Err(e) => {
                warn!("yammer-ibus: could not load {SHIM_LIB_NAME}: {e}");
                return None;
            }
        };

        let rc = match unsafe { lib.get::<StartFn>(b"yammer_ibus_shim_start\0") } {
            Ok(start) => unsafe { start(log_from_shim) },
            Err(e) => {
                warn!("yammer-ibus: {SHIM_LIB_NAME} missing yammer_ibus_shim_start: {e}");
                return None;
            }
        };
        if rc != 0 {
            warn!("yammer-ibus: yammer_ibus_shim_start failed with code {rc}");
            return None;
        }

        Some(Self { lib })
    }

    /// Snapshot whatever currently has focus. Call once per dictation
    /// session, at invocation time — the IBus-backed equivalent of today's
    /// `capture_active_window()`.
    pub fn capture_target(&self) -> Option<IbusTarget> {
        let rc = match unsafe { self.lib.get::<CaptureTargetFn>(b"yammer_ibus_shim_capture_target\0") } {
            Ok(f) => unsafe { f() },
            Err(e) => {
                warn!("yammer-ibus: capture_target symbol lookup failed: {e}");
                return None;
            }
        };
        if rc == 0 { Some(IbusTarget(())) } else { None }
    }

    /// Commit `text` into the instance captured by `target`, regardless of
    /// what currently has focus.
    pub fn commit_text(&self, target: &IbusTarget, text: &str) -> Result<(), IbusError> {
        let _ = target; // proof a capture happened; the shim holds the real state
        let rc = match unsafe { self.lib.get::<CommitTextFn>(b"yammer_ibus_shim_commit_text\0") } {
            Ok(f) => unsafe { f(text.as_ptr(), text.len()) },
            Err(_) => return Err(IbusError::Other(i32::MIN)),
        };
        match rc {
            0 => Ok(()),
            -4 => Err(IbusError::NothingCaptured),
            -5 => Err(IbusError::CommitFailed),
            other => Err(IbusError::Other(other)),
        }
    }

    /// # Safety
    /// Must only resolve `libyammer_ibus_shim.so` built from this same
    /// workspace — the `extern "C"` surface is a private contract between
    /// the two crates, not a stable public ABI.
    unsafe fn load_library() -> Result<Library, libloading::Error> {
        // Prefer a copy next to the running executable (how it will ship),
        // fall back to the normal dynamic linker search path (useful in
        // `cargo test`/`cargo run` from the workspace, where it lands in
        // target/<profile>/ alongside the test binary).
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let candidate = dir.join(SHIM_LIB_NAME);
                if candidate.exists() {
                    return unsafe { Library::new(candidate) };
                }
            }
        }
        unsafe { Library::new(SHIM_LIB_NAME) }
    }
}

static WARNED_NO_ENGINE: OnceLock<()> = OnceLock::new();

/// Log once (not on every dictation) that Yammer is running without the
/// IBus engine, for callers that want a single startup-time note rather than
/// checking `IbusEngine::try_start()`'s `None` themselves.
pub fn warn_unavailable_once() {
    WARNED_NO_ENGINE.get_or_init(|| {
        warn!("yammer-ibus: IBus engine unavailable this session — falling back to xdotool/xclip output");
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `libyammer_ibus_shim.so` is not next to this test binary nor on the
    /// normal dynamic linker search path in a plain `cargo test` run, so
    /// this exercises the real "shim unavailable" case — the same one that
    /// should trigger the xdotool/xclip fallback in production. The
    /// contract under test is "never panics, always degrades to None".
    #[test]
    fn try_start_degrades_to_none_when_shim_is_unavailable() {
        assert!(IbusEngine::try_start().is_none());
    }

    #[test]
    fn warn_unavailable_once_does_not_panic_when_called_repeatedly() {
        warn_unavailable_once();
        warn_unavailable_once();
    }
}
