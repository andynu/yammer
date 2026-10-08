//! uinput virtual keyboard for sending the paste shortcut.
//!
//! On GNOME Wayland, xdotool's XTEST events go through the RemoteDesktop
//! portal and queue behind its approval dialog. A uinput device is seen by
//! the compositor as real hardware, so it needs no portal.
//!
//! The device is created once and kept open: the compositor ignores a new
//! device for a second or more after it appears, so a device created per
//! paste drops the keystroke.

use std::sync::Mutex;
use std::time::Duration;

use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, EventType, InputEvent, KeyCode};
use tracing::{debug, info};

static KEYBOARD: Mutex<Option<VirtualDevice>> = Mutex::new(None);

/// Time for the compositor to attach a freshly created device.
const ATTACH_DELAY: Duration = Duration::from_millis(2500);

fn create() -> std::io::Result<VirtualDevice> {
    let mut keys = AttributeSet::<KeyCode>::new();
    keys.insert(KeyCode::KEY_LEFTSHIFT);
    keys.insert(KeyCode::KEY_INSERT);
    VirtualDevice::builder()?
        .name("yammer-virtual-keyboard")
        .with_keys(&keys)?
        .build()
}

/// Create the virtual keyboard if it doesn't exist yet. Call at startup so
/// the device is attached before the first paste.
pub fn init() -> std::io::Result<()> {
    let mut guard = KEYBOARD.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(create()?);
        info!("Created uinput virtual keyboard");
    }
    Ok(())
}

/// Press and release Shift+Insert on the virtual keyboard.
pub fn shift_insert() -> std::io::Result<()> {
    let mut guard = KEYBOARD.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Some(create()?);
        debug!("Created uinput virtual keyboard on first paste");
        std::thread::sleep(ATTACH_DELAY);
    }
    let device = guard.as_mut().expect("keyboard initialized above");

    let key = |code: KeyCode, value| InputEvent::new(EventType::KEY.0, code.0, value);
    device.emit(&[key(KeyCode::KEY_LEFTSHIFT, 1)])?;
    device.emit(&[key(KeyCode::KEY_INSERT, 1)])?;
    device.emit(&[key(KeyCode::KEY_INSERT, 0)])?;
    device.emit(&[key(KeyCode::KEY_LEFTSHIFT, 0)])?;
    Ok(())
}
