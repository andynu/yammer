//! The `extern "C"` boundary. Kept deliberately tiny — primitive types only —
//! since this is a real dynamic-library edge: the caller (`yammer-ibus`)
//! `dlopen`s this `.so` at runtime via `libloading` rather than linking it in
//! at build time. See CLAUDE.local.md for why that distinction matters here.
//!
//! Logging crosses this boundary through `log_cb` rather than the `tracing`
//! crate — see the comment in Cargo.toml.

use std::ffi::{CString, c_char};
use std::sync::OnceLock;

use crate::bus::{self, Registration};
use crate::engine::{self, FocusTracker};

type LogCb = extern "C" fn(*const c_char);

struct Shim {
    runtime: tokio::runtime::Runtime,
    focus: FocusTracker,
    log_cb: LogCb,
    // Kept only to hold the connection open; never read again.
    _registration: Registration,
}

static SHIM: OnceLock<Shim> = OnceLock::new();

fn log(cb: LogCb, msg: &str) {
    if let Ok(c) = CString::new(msg) {
        cb(c.as_ptr());
    }
}

pub const OK: i32 = 0;
pub const ERR_ALREADY_STARTED: i32 = -1;
pub const ERR_CONNECT_FAILED: i32 = -2;
pub const ERR_NOT_STARTED: i32 = -3;
pub const ERR_NOTHING_CAPTURED: i32 = -4;
pub const ERR_COMMIT_FAILED: i32 = -5;
pub const ERR_INVALID_UTF8: i32 = -6;

/// Start the engine: connect to `ibus-daemon` and register the factory.
/// Safe to call at most once per process — a second call returns
/// `ERR_ALREADY_STARTED` without touching the existing connection.
///
/// # Safety
/// `log_cb` must be a valid function pointer for the lifetime of the process.
#[unsafe(no_mangle)]
pub extern "C" fn yammer_ibus_shim_start(log_cb: LogCb) -> i32 {
    if SHIM.get().is_some() {
        return ERR_ALREADY_STARTED;
    }

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            log(log_cb, &format!("yammer-ibus-shim: building runtime failed: {e}"));
            return ERR_CONNECT_FAILED;
        }
    };

    let focus = FocusTracker::new();
    let registration = runtime.block_on(bus::connect_and_register(focus.clone()));

    let registration = match registration {
        Ok(r) => r,
        Err(e) => {
            log(log_cb, &format!("yammer-ibus-shim: {e}"));
            return ERR_CONNECT_FAILED;
        }
    };

    let shim = Shim {
        runtime,
        focus,
        log_cb,
        _registration: registration,
    };
    // We just checked SHIM.get().is_some() above and returned early if so;
    // nothing else in this process can race us for the set (start is only
    // ever called once, from yammer-ibus's try_start()).
    let _ = SHIM.set(shim);
    log(log_cb, "yammer-ibus-shim: connected and registered");
    OK
}

/// Snapshot whatever currently has focus, for use by a later `commit_text`
/// call. Call this once per dictation session, at invocation time.
#[unsafe(no_mangle)]
pub extern "C" fn yammer_ibus_shim_capture_target() -> i32 {
    let Some(shim) = SHIM.get() else {
        return ERR_NOT_STARTED;
    };
    if shim.focus.capture() {
        OK
    } else {
        ERR_NOTHING_CAPTURED
    }
}

/// Number of `focus_in` events seen so far, or 0 before `start`. Callers
/// compare two readings to tell whether focus has arrived in between.
#[unsafe(no_mangle)]
pub extern "C" fn yammer_ibus_shim_focus_generation() -> u64 {
    SHIM.get().map_or(0, |shim| shim.focus.focus_generation())
}

/// Commit `text` (must be valid UTF-8, `text_len` bytes at `text_ptr`) into
/// the most recently captured target.
///
/// # Safety
/// `text_ptr` must point to `text_len` valid, readable bytes for the
/// duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn yammer_ibus_shim_commit_text(text_ptr: *const u8, text_len: usize) -> i32 {
    let Some(shim) = SHIM.get() else {
        return ERR_NOT_STARTED;
    };

    let bytes = unsafe { std::slice::from_raw_parts(text_ptr, text_len) };
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => return ERR_INVALID_UTF8,
    };

    match shim.runtime.block_on(engine::commit_text(&shim.focus, text)) {
        Ok(()) => OK,
        Err(e) => {
            log(shim.log_cb, &format!("yammer-ibus-shim: commit_text failed: {e}"));
            ERR_COMMIT_FAILED
        }
    }
}
