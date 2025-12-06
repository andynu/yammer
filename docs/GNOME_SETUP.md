# GNOME Keyboard Shortcut Setup

Yammer can be triggered with a global keyboard shortcut. On GNOME, use **Super+H** (the "Windows" key + H) to toggle dictation.

## Automatic Setup

Run the setup script from the repo:

```bash
./scripts/setup-gnome-shortcut.sh
```

This configures Super+H to run `yammer gui --toggle`, which:
- Shows the window if hidden and starts dictation
- Toggles dictation if the window is already visible
- Uses the singleton pattern (won't launch duplicate instances)

## Manual Setup via GNOME Settings

1. Open **Settings** → **Keyboard** → **Keyboard Shortcuts**
2. Scroll to **Custom Shortcuts** and click **+**
3. Fill in:
   - **Name**: `Yammer Dictation`
   - **Command**: `/path/to/yammer/target/release/yammer gui --toggle`
   - **Shortcut**: Press Super+H
4. Click **Add**

## Manual Setup via Command Line

```bash
# Add yammer to custom keybindings list
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings \
  "['/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/yammer/']"

# Configure the shortcut
BINDING="org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/yammer/"

gsettings set $BINDING name 'Yammer Dictation'
gsettings set $BINDING command '/home/you/projects/yammer/target/release/yammer gui --toggle'
gsettings set $BINDING binding '<Super>h'
```

## Removing the Shortcut

```bash
# Remove from the keybindings list
gsettings set org.gnome.settings-daemon.plugins.media-keys custom-keybindings "[]"

# Or reset to defaults
gsettings reset org.gnome.settings-daemon.plugins.media-keys custom-keybindings
```

## Troubleshooting

### Shortcut doesn't work immediately
Log out and back in, or restart GNOME Shell (Alt+F2, type `r`, press Enter on X11).

### Conflict with another shortcut
Super+H might be bound to something else. Check:
```bash
gsettings get org.gnome.desktop.wm.keybindings minimize
```
If it shows `['<Super>h']`, disable it:
```bash
gsettings set org.gnome.desktop.wm.keybindings minimize "[]"
```

### Window doesn't appear
Check if yammer is running:
```bash
pgrep -a yammer
```
Check logs:
```bash
journalctl --user -f | grep yammer
```
