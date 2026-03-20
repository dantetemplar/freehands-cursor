# Freehands Cursor

Standalone Python application for voice control of Cursor IDE.

## Features

- 🎤 Continuous microphone listening in Russian
- 🔔 Voice Mode activation in Cursor via wake word "Джарвис"
- ⌨️ Automatic double "press" of push-to-talk (Cmd+Space / Ctrl+Shift+Space) to focus on Cursor chat, combination is set via CLI
- 📝 Audio accumulation after wake word and recognition as a single chunk
- ✍️ Auto-typing of recognized text in full
- 🪟 Automatic focus on Cursor IDE window
- 🛑 Submit word "Погнали" completes accumulation and starts recognition
- ⏱️ Automatic recognition after 3-second pause
- ⌨️ Pressing Enter also starts recognition
- 🔔 System notifications about status

## Requirements

The application uses [ONNX ASR](https://github.com/alphacep/onnx-asr) for offline speech recognition with VAD (Voice Activity Detection). Models are automatically downloaded on first use via Hugging Face Hub.

### Recognition Model

By default, the model `alphacep/vosk-model-ru` is used. The model is downloaded automatically on first run and cached locally.

You can specify a different model via the `--model-name` parameter:

```bash
uvx freehands-cursor --model-name "alphacep/vosk-model-ru"
```

## Installation

### Via uvx (recommended for users)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run application (will automatically install dependencies)
uvx freehands-cursor

# With custom hotkey
uvx freehands-cursor --hotkey ctrl+shift+space

# With custom model
uvx freehands-cursor --model-name "alphacep/vosk-model-ru"
```

**Note:** On first run, `uvx` will automatically install all dependencies. The recognition model will be downloaded automatically on first use.

## Usage

### Running the Application

After installation, you can run the application via the command `jarvis` (or `uvx freehands-cursor`):

```bash
# Standard run
jarvis

# With custom hotkey
jarvis --hotkey ctrl+shift+space
```

The application starts and immediately begins listening to the microphone.

### Working with the Application

1. Start the application
2. Say "Джарвис" — the Cursor window gets focus, the application double-clicks the voice input hotkey and activates the chat field
3. Dictate your request — audio accumulates in the buffer without recognition
4. Complete the input in one of the ways:
   - Say "Погнали" — recognition of all accumulated audio starts
   - Press Enter — recognition of all accumulated audio starts
   - Wait 3 seconds of pause — recognition starts automatically
5. During recognition, "..." is shown, then all recognized text is inserted in one piece
6. When using submit word "Погнали", the application automatically presses Enter to send the message

**Features:**
- Audio accumulates without recognition until input completion
- Recognition is performed in one piece for all accumulated audio
- If you start typing/clicking manually, auto-typing is disabled, but accumulation continues

## Configuration

### Recognition Model

By default, the application uses the model `alphacep/vosk-model-ru`. You can specify a different model via the `--model-name` parameter:

```bash
# Use another model
uvx freehands-cursor --model-name "alphacep/vosk-model-ru"
```

Models are automatically downloaded via Hugging Face Hub and cached locally.

### Custom Hotkey

By default, the following combinations are used:
- **Windows/Linux**: Ctrl+Shift+Space
- **macOS**: Cmd+Space

The combination is specified in the format `HotKey.parse`, i.e., special keys must be wrapped in angle brackets:

```bash
jarvis --hotkey "<ctrl>+<shift>+space"
jarvis --hotkey "<alt>+<space>"
jarvis --hotkey "<shift>+f1"
```

### Additional Startup Parameters

- `--wake-word` — activation word (default: "джарвис")
- `--submit-word` — word to complete input and start recognition (default: "погнали")
- `--save-to-buffer/--no-save-to-buffer` — enable/disable automatic copying of the last message to clipboard (default: enabled)
- `--model-name` — ONNX ASR model name from Hugging Face Hub (default: "alphacep/vosk-model-ru")

## Troubleshooting

### Microphone Not Working

- Check microphone access permissions in OS settings
- Ensure microphone is connected and working

### Cursor Not Found

- Ensure Cursor IDE is running
- Check that the process name contains "cursor"

### Keys Not Being Pressed

- Ensure the application has keyboard emulation rights
- On Linux, `ydotool` is required (installed separately)
- Ensure `ydotool` is running: `ydotool daemon` (may require sudo)
- Try a different key combination: `jarvis --hotkey alt+space`

### Model Not Loading

- Check internet connection (model is downloaded via Hugging Face Hub on first run)
- Ensure the specified model name exists in Hugging Face Hub
- Check application logs in `~/.freehands_cursor/freehands_cursor.log`
- Model is cached locally, subsequent runs don't require internet

### Installation Errors on Linux

If compilation errors occur during installation, install system dependencies:

**Fedora/RHEL/CentOS:**
```bash
sudo dnf install python3-devel portaudio-devel ydotool
```

**Debian/Ubuntu:**
```bash
sudo apt-get install python3-dev portaudio19-dev ydotool
```

**Arch Linux:**
```bash
sudo pacman -S python portaudio ydotool
```

**openSUSE:**
```bash
sudo zypper install python3-devel portaudio-devel ydotool
```

**Note:** After installing `ydotool`, you may need to start the daemon:
```bash
sudo ydotool daemon
```

## Logs

Logs are saved to `~/.freehands_cursor/freehands_cursor.log`

---

## For Developers

### Development Installation

```bash
# Clone repository
git clone <repository-url>
cd freehands-cursor

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package in editable mode (for development)
uv pip install -e .[dev]

# Run application
uv run jarvis
```

### Dependencies

- `onnx-asr` - Offline speech recognition with VAD
- `sounddevice>=0.4.6` - Audio capture
- `psutil>=5.9.0` - Process management
- `ydotool` - Keyboard automation on Linux (system dependency)
- `numpy>=1.24.0` - Numerical computations

### Technologies

The application uses:
- **[ONNX ASR](https://github.com/alphacep/onnx-asr)** - ONNX-based offline speech recognition with VAD support
- Models are loaded via Hugging Face Hub and cached locally
- Works completely offline after first load
- Low recognition latency due to batch processing

### Model Configuration

To change the recognition model, use the `--model-name` parameter:

```bash
uvx freehands-cursor --model-name "alphacep/vosk-model-ru"
```

Or in code:

```python
app = VoiceWakeupApp(model_path="alphacep/vosk-model-ru")
```

### Project Structure

```
freehands-cursor/
├── freehands_cursor/
│   └── __init__.py          # Main module (all components in one file)
├── pyproject.toml
└── README.md
```

### Logic

1. **IDLE mode**: Continuous listening and wake word detection
2. **ACCUMULATING mode**: After wake word, audio accumulation starts in buffer without recognition
3. **Recognition triggers**:
   - Pronouncing submit word
   - User pressing Enter
   - 3-second pause
4. **Recognition**: Entire accumulated buffer is processed in one piece via model
5. **Text insertion**: "..." is shown during recognition, then entire text is inserted

## License

MIT
