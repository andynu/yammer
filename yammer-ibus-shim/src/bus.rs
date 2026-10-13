//! Connects to the private IBus bus and registers Yammer's engine factory.

use librush::ibus::{IBus, get_ibus_addr};

use crate::engine::{FocusTracker, YammerEngine, YammerFactory};

pub const ENGINE_NAME: &str = "yammer";
pub const WELL_KNOWN_NAME: &str = "org.freedesktop.IBus.Yammer";

/// Holds the live IBus connection + registration alive for as long as the
/// engine should keep running. Dropping it tears the connection down.
pub struct Registration {
    _ibus: IBus<YammerEngine, YammerFactory>,
}

pub async fn connect_and_register(focus: FocusTracker) -> Result<Registration, String> {
    let addr = get_ibus_addr().map_err(|e| format!("locating ibus-daemon: {e}"))?;
    let factory = YammerFactory::new(focus.clone());
    let ibus = IBus::new(addr, factory, WELL_KNOWN_NAME.to_string())
        .await
        .map_err(|e| format!("connecting to ibus-daemon: {e}"))?;
    focus.set_connection(ibus.conn());
    Ok(Registration { _ibus: ibus })
}
