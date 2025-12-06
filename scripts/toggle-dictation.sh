#!/bin/bash
# Wrapper script for GNOME shortcut debugging

LOG="/tmp/yammer-toggle.log"
APP="/home/andy/projects/yammer/target/release/yammer-app"

echo "=== $(date) ===" >> "$LOG"
echo "Called with args: $@" >> "$LOG"
echo "USER: $USER" >> "$LOG"
echo "DISPLAY: $DISPLAY" >> "$LOG"
echo "WAYLAND_DISPLAY: $WAYLAND_DISPLAY" >> "$LOG"
echo "XDG_SESSION_TYPE: $XDG_SESSION_TYPE" >> "$LOG"

# Check if app is running
if pgrep -f "yammer-app" > /dev/null; then
    echo "App already running (PID: $(pgrep -f yammer-app))" >> "$LOG"
else
    echo "App not running" >> "$LOG"
fi

# Check if binary exists
if [ -x "$APP" ]; then
    echo "Binary exists and is executable" >> "$LOG"
else
    echo "ERROR: Binary not found or not executable: $APP" >> "$LOG"
    exit 1
fi

# Run the app with toggle
echo "Executing: $APP --toggle" >> "$LOG"
RUST_LOG=info "$APP" --toggle >> "$LOG" 2>&1 &
APP_PID=$!
echo "Spawned with PID: $APP_PID" >> "$LOG"

# Give it a moment and check if it's still running
sleep 1
if ps -p $APP_PID > /dev/null 2>&1; then
    echo "Process still running after 1s" >> "$LOG"
else
    echo "Process exited within 1s" >> "$LOG"
fi

echo "" >> "$LOG"
