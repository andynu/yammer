# IBus Input Source Setup

Since plans/002-ibus-input-engine.md, Yammer's dictated text is committed through a real
IBus input-method engine instead of simulated keystrokes — this is what makes it work
properly under Wayland. Super+H (see `GNOME_SETUP.md`) is unchanged: it's a
compositor-level shortcut, outside IBus entirely. This setup only affects how the dictated
text gets delivered once it's ready.

## Why this is a one-time setup step

IBus only sends focus/key events to an engine that is the **currently selected input
source** for a given text field — the same mechanism as switching keyboard layouts. For
Yammer to receive focus no matter which app or window you're dictating into, "Yammer
Dictation" needs to be your sole or default input source, not one you switch to manually
before each dictation.

If it isn't selected (or `ibus-daemon` isn't running, or the engine failed to start),
Yammer falls back to the legacy xdotool/xclip output path automatically — dictation still
works, just without the Wayland-native delivery. Nothing breaks either way; this setup
step only improves reliability.

## Setup

1. Open **Settings** → **Keyboard** → **Input Sources**.
2. Click **+** → search for **Yammer Dictation** → **Add**.
3. Drag it to the top of the list (or remove other sources) so it's the default/sole
   source, rather than one you'd need to switch to.

If "Yammer Dictation" doesn't appear in the list, the `.deb` package didn't install the
component descriptor — check `/usr/share/ibus/component/yammer.xml` exists, then
```bash
ibus restart
```

## Caveat: non-`us` keyboard layouts

The engine declares `<layout>us</layout>` in its component descriptor (it's a pure
passthrough — normal typing is never intercepted, only dictated text is committed — but
IBus still wants *a* declared base layout). If you rely on dead keys or compose sequences
from a different layout, check those still work with Yammer selected before making it
your sole source. This hasn't been tested against non-`us` layouts.

## Troubleshooting

### Dictation works but text never appears (and no xdotool fallback either)
Check whether `ibus-daemon` is running at all:
```bash
pgrep -a ibus-daemon
```
GNOME only starts it automatically once a non-default input source has been added once.

### Text appears via the old typing/clipboard behavior, not instantly
This means the IBus path isn't active this session (engine unavailable, or "Yammer
Dictation" isn't the selected source) and Yammer silently fell back. Check the app's logs:
```bash
journalctl --user -f | grep -i yammer
```
A line like `yammer-ibus: IBus engine unavailable this session` or
`ibus commit_text failed, falling back` identifies which.

### Only one app/window gets correct dictation when two are open
librush (the library backing the engine) only supports one live engine instance per
process as of 0.2.3 — see plans/002-ibus-input-engine.md's Known Risks. If you dictate
into one window, switch to another that also has Yammer selected, then dictate again
without the first window regaining focus in between, the second dictation may land in the
wrong place. Not fixed upstream yet.
