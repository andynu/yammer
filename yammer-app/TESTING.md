# Yammer UI Testing Guide

This guide walks you through testing each component of the Tauri UI in isolation.

## Prerequisites

- X11 session (not Wayland)
- Node.js and npm installed
- Rust toolchain installed

Verify X11:
```bash
echo $XDG_SESSION_TYPE
# Should output: x11
```

## 1. Start the Tauri App in Dev Mode

```bash
cd yammer-app
npm run tauri dev
```

This will:
- Start the Vite dev server (frontend)
- Launch the Tauri window (should appear as a small floating overlay)

You should see a small transparent window with rounded corners, a status indicator showing "Ready", and a placeholder message.

## 2. Test the Basic Window (Phase 5.1)

Once the window appears, verify:

- **Transparency**: Can you see through the window background? (only the dark overlay should be visible)
- **Rounded corners**: The window should have smooth 16px rounded corners
- **No decorations**: No title bar or window borders
- **Always on top**: Try opening another app - the Yammer window should stay on top
- **Draggable**: Click and drag anywhere on the dark area - window should move

**Known Issue**: If dragging doesn't work, all child elements need `data-tauri-drag-region` attribute (fixed in latest version).

## 3. Test the Waveform Visualization (Phase 5.2)

The waveform auto-simulates every second. Watch the canvas area:

- You should see **40 vertical bars** animating with a sine wave pattern
- Bars should have **orange/red gradient colors** based on amplitude
- Between updates, bars should **slowly decay** (fade down)

### Manual Waveform Test

1. Open dev tools: Right-click window → "Inspect Element"
2. In the console, type:
```javascript
window.testWaveform()
```
3. You should see bars animate immediately

## 4. Test Status Indicators & States (Phase 5.3)

In the dev tools console, run the state cycle test:

```javascript
window.testStates()
```

This will cycle through all states every 2 seconds:

1. **Listening** (green dot, pulsing) - "Listening..."
2. **Processing** (orange dot, fast pulse) - "Processing..." + partial text (dimmed, italic)
3. **Correcting** (blue dot, medium pulse) - "Correcting..." + final text
4. **Done** (solid green) - "Done" + final text (full brightness)
5. **Idle** (gray dot) - "Ready" + placeholder

Watch for:
- Status indicator color changes
- Pulsing animations
- Status text updates
- Transcript text appearing/disappearing
- Opacity/style changes on text

## 5. Test Individual States Manually

You can set any state manually in the console:

### Test listening state:
```javascript
window.setYammerState({ status: 'listening', transcript: '' })
```

### Test with partial transcript:
```javascript
window.setYammerState({
  status: 'processing',
  transcript: 'This is partial text...',
  isPartial: true
})
```

### Test final result:
```javascript
window.setYammerState({
  status: 'done',
  transcript: 'This is the final corrected text.',
  isPartial: false
})
```

### Test error state:
```javascript
window.setYammerState({ status: 'error', transcript: '' })
```

### Reset to idle:
```javascript
window.setYammerState({ status: 'idle', transcript: '' })
```

## 6. Test Window Behavior Across Apps

1. Open a text editor (like gedit or VSCode)
2. The Yammer window should stay on top
3. Try moving it around - should work smoothly
4. Focus different apps - window remains visible and draggable

## 7. Verify Backend Compilation

Check that the Rust backend compiles:

```bash
cd yammer-app/src-tauri
cargo check
```

Should complete without errors.

## Common Issues & Fixes

### Window doesn't appear transparent
- Verify X11 (not Wayland): `echo $XDG_SESSION_TYPE`
- Check compositor is running (usually automatic on GNOME)

### Waveform not animating
- Check browser console for errors (Right-click → Inspect)
- Try `window.testWaveform()` manually
- Auto-simulation should trigger every 1 second

### Can't drag window
- Make sure you're clicking on the dark overlay area
- Verify all elements have `data-tauri-drag-region` attribute
- If issues persist, see issue yam-f8a

### Window not staying on top
- Check `alwaysOnTop: true` in `src-tauri/tauri.conf.json`
- Some window managers may override this setting

## State Reference

| State | Indicator | Text | Transcript Style |
|-------|-----------|------|------------------|
| idle | Gray, small | "Ready" | Placeholder (gray, italic) |
| listening | Green, pulsing slow | "Listening..." | Empty or previous |
| processing | Orange, pulsing fast | "Processing..." | Partial (dimmed, italic) |
| correcting | Blue, pulsing medium | "Correcting..." | Normal brightness |
| done | Green, solid, large | "Done" | Full brightness |
| error | Red, solid | "Error" | Previous or empty |

## Next Steps: Audio Integration (Future)

Once the audio pipeline is wired up, the waveform will receive real audio samples instead of simulated data, and states will transition automatically:

- VAD detection → `listening`
- STT processing → `processing`
- LLM correction → `correcting`
- Output ready → `done`

For now, everything can be tested with the simulation and manual state changes.

## Development Tips

### Hot Reload
The Vite dev server supports hot reload. Changes to:
- HTML, CSS, JS → Instant reload
- Rust code → Requires full restart of `npm run tauri dev`

### Console Access
Always keep dev tools open during testing:
- Right-click window → "Inspect Element"
- Or press F12 (if dev tools is enabled in dev mode)

### Building for Production
```bash
npm run tauri build
```

The built application will be in `src-tauri/target/release/`.
