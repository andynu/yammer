#!/bin/bash
# Setup GNOME keyboard shortcut for Yammer dictation
# Binds Super+H to toggle yammer dictation

set -e

# Find the yammer binary (prefer release build in this repo)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
YAMMER_BIN="$REPO_ROOT/target/release/yammer"

if [[ ! -x "$YAMMER_BIN" ]]; then
    echo "Error: yammer binary not found at $YAMMER_BIN"
    echo "Build it first: cargo build --release"
    exit 1
fi

echo "Using yammer binary: $YAMMER_BIN"

# Get existing custom keybindings
EXISTING=$(gsettings get org.gnome.settings-daemon.plugins.media-keys custom-keybindings 2>/dev/null || echo "[]")

# Check if yammer binding already exists
if echo "$EXISTING" | grep -q "yammer"; then
    echo "Yammer shortcut already configured, updating..."
else
    echo "Adding new yammer shortcut..."
    # Add yammer to the list
    if [[ "$EXISTING" == "@as []" || "$EXISTING" == "[]" ]]; then
        NEW_LIST="['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/yammer/']"
    else
        # Remove trailing ] and add our entry
        NEW_LIST="${EXISTING%]*}, '/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/yammer/']"
    fi
    gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "$NEW_LIST"
fi

# Configure the yammer shortcut
BINDING_PATH="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/yammer/"

gsettings set $BINDING_PATH name 'Yammer Dictation'
gsettings set $BINDING_PATH command "$YAMMER_BIN gui --toggle"
gsettings set $BINDING_PATH binding '<Super>h'

echo ""
echo "GNOME shortcut configured:"
echo "  Name:    Yammer Dictation"
echo "  Key:     Super+H"
echo "  Command: $YAMMER_BIN gui --toggle"
echo ""
echo "The shortcut should work immediately. If not, log out and back in."
