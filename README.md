# Freehands Cursor

Standalone Python app that listens for a wake word and triggers Cursor voice mode hotkey.

## What It Does

- Continuously listens to microphone audio with `alphacep/vosk-model-small-ru` ASR model
- Detects wake word (default: `джарвис`)
- Focuses Cursor window
- On wake word, presses Cursor Focus Agent (`composer.focusComposer`) hotkey first, then presses the configured voice hotkey (default: `<ctrl>+<shift>+<space>`)
- Detects submit keyword (default: `газ`) to trigger Cursor Focus Agent (`composer.focusComposer`) hotkey (default: `<ctrl>+<shift>+y`) and press `Enter`

## Install / Run

Before running `freehands-cursor`, you must set a keybind for Focus Agent (`composer.focusComposer`): open Command Palette (`Ctrl+Shift+P`), type `Focus Agent`, click the gear icon, and add keybind `Ctrl+Shift+Y`.

```bash
# install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# run from package
uvx freehands-cursor
```

If installed into an environment, both commands are available:

- `freehands-cursor`
- `jarvis`

## CLI Options

```bash
freehands-cursor # standard run
freehands-cursor --toggle-voice-mode-hotkey "<ctrl>+<shift>+<space>" --wake-word "джарвис"
freehands-cursor --focus-agent-hotkey "<ctrl>+<shift>+u" --submit-keyword "газ"
freehands-cursor --model-name "alphacep/vosk-model-small-ru"
```

Available flags:

- `--toggle-voice-mode-hotkey`: Hotkey to trigger Cursor voice mode (default: `<ctrl>+<shift>+<space>`)
- `--wake-word`: Wake word (default: `джарвис`)
- `--focus-agent-hotkey`: Hotkey for Cursor Focus Agent (`composer.focusComposer`) (default: `<ctrl>+<shift>+u`)
- `--submit-keyword`: Word that triggers Cursor Focus Agent then `Enter` (default: `газ`)
- `--model-name`: ONNX ASR model name (default runtime fallback: `alphacep/vosk-model-small-ru`)
- `--dry-run-focus`: only focus Cursor window, then exit
- `--dry-run-hotkeys`: only press the configured hotkey once, then exit

## Linux / Desktop Notes

- On X11/Windows/macOS, window focusing uses `pywinctl` first (with X11 fallback to `wmctrl`)
- On Wayland, keyboard backend defaults to [`ydotool`](https://github.com/ReimuNotMoe/ydotool)
- On GNOME Wayland, focusing uses the [`activate-window-by-title`](https://github.com/lucaswerkmeister/activate-window-by-title) GNOME extension DBus API (`de.lucaswerkmeister.ActivateWindowByTitle`)

For Wayland, install both:

- [`ydotool`](https://github.com/ReimuNotMoe/ydotool)
- [`activate-window-by-title`](https://github.com/lucaswerkmeister/activate-window-by-title) (GNOME Wayland focus integration)

If hotkeys fail on Wayland, ensure `ydotoold` daemon is running.

## Troubleshooting

- **No microphone input**: verify OS microphone permissions and selected input device
- **Cursor window not focused**: ensure Cursor is running and window title contains `Cursor`
- **Hotkey not pressed**: check `ydotool` (Wayland) or desktop input permissions
- **Model load failed**: check internet on first run and model name validity
- **Logs**: inspect `~/.freehands_cursor/freehands_cursor.log`

## Development

```bash
uv pip install -e .
uv run freehands-cursor
```

## License

MIT
