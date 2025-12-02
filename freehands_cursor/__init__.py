"""
Voice Wakeup Cursor - hands-free управление Voice Mode
"""

import argparse
import json
import logging
import os
import platform
import queue
import subprocess
import sys
import threading
import time
from enum import StrEnum
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
from pynput.keyboard import Controller, GlobalHotKeys, HotKey, Key
from vosk import KaldiRecognizer, Model, SetLogLevel

# ===== CONFIG =====
APP_HOME_DIR = Path.home() / ".freehands_cursor"
APP_HOME_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_HOME_DIR / "freehands_cursor.log"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 4096

# Disable Vosk logs
SetLogLevel(-1)

# ==================


# ===== LOGGING =====
class CleanFormatter(logging.Formatter):
    """Formatter that shortens module names for readability."""

    def format(self, record):
        if record.name.startswith("freehands_cursor."):
            record.name = record.name.replace("freehands_cursor.", "")
        elif record.name == "vosk":
            record.name = "vosk"
        return super().format(record)


file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(CleanFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)

logging.getLogger("vosk").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ===== GLOBALS =====
class AppState(StrEnum):
    """Состояния приложения."""

    IDLE = "idle"
    LISTENING = "listening"
    PUSHED = "pushed"
    RELEASED = "released"


# Platform checks
IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"

# Check if Windows-specific libraries are available
WINDOWS_AVAILABLE = False
if IS_WINDOWS:
    try:
        import importlib.util

        if importlib.util.find_spec("win32gui") and importlib.util.find_spec("win32con"):
            WINDOWS_AVAILABLE = True
    except ImportError:
        logger.warning("win32gui/win32con not available")


# ===== AUDIO PROCESSING =====
class AudioProcessor:
    """Захват аудио с микрофона."""

    def __init__(self, sample_rate=DEFAULT_SAMPLE_RATE, chunk_size=DEFAULT_CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.stream = None

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_recording:
            audio_data = indata[:, 0].astype(np.float32)
            try:
                self.audio_queue.put(audio_data.copy(), block=False)
            except queue.Full:
                logger.warning("Audio queue is full, dropping frames")

    def start(self):
        if self.is_recording:
            return
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()
        self.is_recording = True
        logger.info(f"Audio recording started (sample_rate={self.sample_rate})")

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Audio recording stopped")

    def get_audio_chunk(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# ===== SPEECH DETECTION =====
def download_vosk_model(model_url="https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"):
    """Download and extract Vosk model."""
    import tempfile
    import urllib.request
    import zipfile

    logger.info("Downloading Vosk model from %s...", model_url)

    # Download to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        tmp_path = tmp_file.name

    try:
        urllib.request.urlretrieve(model_url, tmp_path)
        logger.info("Model downloaded successfully")

        # Extract to ~/.freehands_cursor/
        extract_dir = APP_HOME_DIR
        logger.info(f"Extracting model to {extract_dir}...")

        with zipfile.ZipFile(tmp_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        logger.info("Model extracted successfully")

        # Find the extracted directory
        for item in extract_dir.iterdir():
            if item.is_dir() and "vosk-model" in item.name.lower():
                return str(item)

        raise FileNotFoundError("Extracted model directory not found")

    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class SpeechDetector:
    """Распознавание речи с использованием Vosk."""

    def __init__(self, model_path: str, sample_rate=DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.model = None
        self.recognizer = None
        self._load_model()
        logger.info(f"SpeechDetector initialized with model: {self.model_path}")

    def _load_model(self):
        logger.info(f"Loading Vosk model from {self.model_path}...")
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        logger.info("Vosk model loaded successfully")

    def process_audio_frame(self, audio_chunk):
        """Process audio frame and return recognized text if available."""
        if self.recognizer is None:
            return None

        # Convert float32 to int16 PCM
        mono = audio_chunk
        if audio_chunk.dtype != np.float32:
            mono = audio_chunk.astype(np.float32)
        mono = np.clip(mono, -1.0, 1.0)
        pcm16 = (mono * 32767).astype("int16").tobytes()

        # Feed audio continuously to Vosk
        if self.recognizer.AcceptWaveform(pcm16):
            # Final result (end of utterance detected by Vosk)
            result_json = self.recognizer.Result()
            try:
                result = json.loads(result_json)
                text = result.get("text", "")
                if text:
                    return text.strip().lower()
            except Exception:
                pass
        else:
            # Partial result (ongoing recognition)
            partial_json = self.recognizer.PartialResult()
            try:
                partial = json.loads(partial_json)
                partial_text = partial.get("partial", "")
                if partial_text:
                    logger.debug(f"[PARTIAL] {partial_text}")
            except Exception:
                pass

        return None


# ===== KEY SIMULATION =====
class KeySimulator:
    """Имитация нажатий клавиш push-to-talk."""

    DEFAULT_COMBOS = {
        "Darwin": [Key.cmd, Key.space],
        "Windows": [Key.ctrl, Key.shift, Key.space],
        "Linux": [Key.ctrl, Key.shift, Key.space],
    }

    KEY_MAP = {
        "cmd": Key.cmd,
        "command": Key.cmd,
        "win": Key.cmd,
        "super": Key.cmd,
        "meta": Key.cmd,
        "control": Key.ctrl,
        "ctrl": Key.ctrl,
        "shift": Key.shift,
        "alt": Key.alt,
        "option": Key.alt,
        "spacebar": Key.space,
        "space": Key.space,
    }

    DISPLAY_MAP = {
        Key.cmd: "Cmd",
        Key.ctrl: "Ctrl",
        Key.shift: "Shift",
        Key.alt: "Alt",
        Key.space: "Space",
    }

    def __init__(self, combo_select_chat_input: str = "<ctrl>+<insert>"):
        self.system = platform.system()
        self.keyboard = Controller()
        self.combo_select_chat_input = HotKey.parse(combo_select_chat_input)
        self.is_pressed = False
        logger.info("Hotkey set to %s", self.get_key_combination())

    def _resolve_combo(self, custom_combo):
        if custom_combo:
            parsed = self._parse_combo(custom_combo)
            if parsed:
                return parsed
            logger.warning(f"Failed to parse combo '{custom_combo}', using default")
        return self.DEFAULT_COMBOS.get(self.system, [Key.ctrl, Key.shift, Key.space])

    def _parse_combo(self, combo):
        parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
        if not parts:
            return []
        result = []
        for part in parts:
            key = self.KEY_MAP.get(part)
            if key:
                result.append(key)
            else:
                logger.warning(f"Unknown key: {part}")
        return result if result else None

    def press_select_chat_input(self):
        if self.is_pressed:
            return
        try:
            for key in self.key_sequence:
                self.keyboard.press(key)
                time.sleep(0.01)  # Small delay between key presses
            self.is_pressed = True
            logger.info("Push-to-talk pressed")
        except Exception as e:
            logger.error(f"Error pressing push-to-talk: {e}")
            self.is_pressed = False

    def release_push_to_talk(self):
        if not self.is_pressed:
            return
        try:
            for key in reversed(self.key_sequence):
                self.keyboard.release(key)
                time.sleep(0.01)  # Small delay between key releases
            self.is_pressed = False
            logger.info("Push-to-talk released")
        except Exception as e:
            logger.error(f"Error releasing push-to-talk: {e}")

    def get_key_combination(self):
        parts = [self.DISPLAY_MAP.get(key, str(key)) for key in self.key_sequence]
        return "+".join(parts)


# ===== WINDOW MANAGEMENT =====
class WindowManager:
    """Поиск и управление окном Cursor IDE."""

    def __init__(self):
        self.system = platform.system()
        self.cursor_window = None

    def find_cursor_window(self):
        if self.system == "Windows":
            return self._find_cursor_windows()
        elif self.system == "Darwin":
            return self._find_cursor_macos()
        else:
            return self._find_cursor_linux()

    def _find_cursor_windows(self):
        if not WINDOWS_AVAILABLE:
            return self._find_cursor_by_process()

        import win32gui as wgui

        def enum_handler(hwnd, ctx):
            if wgui.IsWindowVisible(hwnd):
                window_text = wgui.GetWindowText(hwnd) or ""
                class_name = wgui.GetClassName(hwnd) or ""
                if "cursor" in window_text.lower() or "cursor" in class_name.lower():
                    self.cursor_window = hwnd
                    return False
            return True

        try:
            wgui.EnumWindows(enum_handler, None)
            if self.cursor_window:
                logger.info(f"Found Cursor window: {self.cursor_window}")
                return True
        except Exception as e:
            logger.error(f"Error finding Cursor window on Windows: {e}")
        return False

    def _find_cursor_macos(self):
        try:
            script = """
            tell application "System Events"
                set cursorApp to first application process whose name contains "Cursor"
                return name of cursorApp
            end tell
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=False)
            if result.returncode == 0 and result.stdout.strip():
                logger.info("Found Cursor application on macOS")
                return True
        except Exception as e:
            logger.error(f"Error finding Cursor on macOS: {e}")
        return self._find_cursor_by_process()

    def _find_cursor_linux(self):
        # Try different xdotool search patterns
        search_patterns = [
            ["xdotool", "search", "--name", "Cursor"],
            ["xdotool", "search", "--class", "cursor"],
            ["xdotool", "search", "--classname", "Cursor"],
        ]

        for pattern in search_patterns:
            try:
                result = subprocess.run(pattern, capture_output=True, text=True, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    window_ids = result.stdout.strip().split("\n")
                    if window_ids:
                        self.cursor_window = window_ids[0]
                        logger.info(f"Found Cursor window on Linux: {self.cursor_window}")
                        return True
            except Exception as e:
                logger.debug(f"xdotool pattern {pattern} failed: {e}")

        # If xdotool fails, try wmctrl
        try:
            result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if "cursor" in line.lower():
                        window_id = line.split()[0]
                        self.cursor_window = window_id
                        logger.info(f"Found Cursor window via wmctrl: {window_id}")
                        return True
        except Exception as e:
            logger.debug(f"wmctrl not available or error: {e}")

        return self._find_cursor_by_process()

    def _find_cursor_by_process(self):
        try:
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    name = proc.info["name"] or ""
                    exe = proc.info["exe"] or ""
                    if "cursor" in name.lower() or "cursor" in exe.lower():
                        logger.info(f"Found Cursor process: {proc.info['pid']}")
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error finding Cursor process: {e}")
        return False

    def focus_cursor_window(self):
        if not self.find_cursor_window():
            logger.warning("Cursor window not found")
            return False

        if self.system == "Windows":
            return self._focus_windows()
        elif self.system == "Darwin":
            return self._focus_macos()
        else:
            return self._focus_linux()

    def _focus_windows(self):
        if not WINDOWS_AVAILABLE or not self.cursor_window:
            return False
        try:
            import win32con as wcon
            import win32gui as wgui

            wgui.ShowWindow(int(self.cursor_window), wcon.SW_RESTORE)
            wgui.SetForegroundWindow(int(self.cursor_window))
            logger.info("Focused Cursor window on Windows")
            return True
        except Exception as e:
            logger.error(f"Error focusing window on Windows: {e}")
            return False

    def _focus_macos(self):
        try:
            script = """
            tell application "Cursor"
                activate
            end tell
            """
            result = subprocess.run(["osascript", "-e", script], capture_output=True, check=False)
            if result.returncode == 0:
                logger.info("Focused Cursor window on macOS")
                return True
        except Exception as e:
            logger.error(f"Error focusing window on macOS: {e}")
        return False

    def _focus_linux(self):
        if not self.cursor_window:
            logger.warning("No window ID available for focusing")
            return False

        # Try multiple approaches to ensure window gets focus
        success = False

        # Method 1: xdotool with sync flag (waits for operation to complete)
        try:
            result = subprocess.run(
                ["xdotool", "windowactivate", "--sync", str(self.cursor_window)],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0:
                logger.info("Focused Cursor window on Linux using xdotool")
                success = True
        except Exception as e:
            logger.debug(f"xdotool not available: {e}")

        # Method 2: wmctrl as fallback/additional attempt
        if not success:
            try:
                # First raise the window
                subprocess.run(
                    ["wmctrl", "-ia", str(self.cursor_window)], capture_output=True, text=True, check=False, timeout=2
                )
                # Then activate it
                subprocess.run(
                    ["wmctrl", "-iR", str(self.cursor_window)], capture_output=True, text=True, check=False, timeout=2
                )
                logger.info("Focused Cursor window on Linux using wmctrl")
                success = True
            except Exception as e:
                logger.debug(f"wmctrl not available: {e}")

        # Method 3: Try xdotool focus as last resort
        if not success:
            try:
                subprocess.run(
                    ["xdotool", "windowfocus", str(self.cursor_window)],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                logger.info("Focused Cursor window on Linux using xdotool focus")
                success = True
            except Exception as e:
                logger.debug(f"xdotool focus failed: {e}")

        if not success:
            logger.warning("Failed to focus Cursor window with available tools")

        return success


# ===== NOTIFICATION =====
def show_notification(title, message):
    """Показать системное уведомление."""
    try:
        if IS_MACOS:
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                capture_output=True,
                check=False,
            )
        elif IS_WINDOWS:
            try:
                from win10toast import ToastNotifier

                toaster = ToastNotifier()
                toaster.show_toast(title, message, duration=3)
            except ImportError:
                pass
        else:  # Linux
            try:
                subprocess.run(["notify-send", title, message], capture_output=True, check=False)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Failed to show notification: {e}")


# ===== MAIN APPLICATION =====
class VoiceWakeupApp:
    """Главный класс приложения."""

    def __init__(self, hotkey=None, wake_word="джарвис", release_word="погнали"):
        self.state = AppState.IDLE
        self.is_running = False
        self.thread = None

        self.wake_word = wake_word.lower()
        self.release_word = release_word.lower()

        self.audio_processor = AudioProcessor()
        self.speech_detector = None
        self.window_manager = WindowManager()
        self.key_simulator = KeySimulator(custom_combo=hotkey)

        logger.info(f"Voice Wakeup Cursor initialized (wake: '{self.wake_word}', release: '{self.release_word}')")

    def initialize_speech_detector(self, model_path=None):
        """Initialize speech detector with model path resolution and download."""
        # Resolve model path
        if model_path is None:
            model_path = os.getenv("VOSK_MODEL_PATH")
            if model_path is None:
                # Try multiple locations
                candidates = [
                    Path.cwd() / "models" / "vosk-model-small-ru-0.22",  # Project directory
                    APP_HOME_DIR / "vosk-model-small-ru-0.22",  # User home
                    APP_HOME_DIR / "vosk-model-ru",  # Legacy name
                ]

                for candidate in candidates:
                    if candidate.exists():
                        model_path = str(candidate)
                        break

                if model_path is None:
                    model_path = str(APP_HOME_DIR / "vosk-model-small-ru-0.22")

        # Download model if it doesn't exist
        if not os.path.exists(model_path):
            logger.warning(f"Vosk model not found at {model_path}")
            logger.info("Attempting to download model automatically...")

            try:
                model_path = download_vosk_model()
                logger.info(f"Model downloaded to {model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                logger.error(
                    f"Please manually download a Russian model from https://alphacephei.com/vosk/models "
                    f"and extract it to {model_path}"
                )
                raise FileNotFoundError("Vosk model not found and auto-download failed")

        # Create SpeechDetector with resolved path
        self.speech_detector = SpeechDetector(model_path=model_path)
        logger.info("Speech detector initialized")

    def check_wake_word(self, text):
        """Check if text contains wake word."""
        if not text:
            return False
        text_lower = text.lower().strip()
        return self.wake_word in text_lower or text_lower == self.wake_word

    def check_release_word(self, text):
        """Check if text contains release word."""
        if not text:
            return False
        text_lower = text.lower().strip()
        return self.release_word in text_lower or text_lower == self.release_word

    def start(self):
        if self.is_running:
            logger.warning("Application already running")
            return

        try:
            if self.speech_detector is None:
                self.initialize_speech_detector()

            self.audio_processor.start()

            self.is_running = True
            self.thread = threading.Thread(target=self._main_loop, daemon=True)
            self.thread.start()

            self.state = AppState.IDLE
            logger.info("Application started")

            show_notification("Приложение запущено", f"Ожидание wake word: '{self.wake_word}'")
            logger.info("Используемая горячая клавиша: %s", self.key_simulator.get_key_combination())

        except Exception:
            logger.error("Failed to start application", exc_info=True)
            raise

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self.state == AppState.PUSHED:
            self.key_simulator.release_push_to_talk()

        self.audio_processor.stop()

        self.state = AppState.IDLE
        logger.info("Application stopped")

        show_notification("Приложение остановлено", "")

    def quit(self):
        logger.info("Quitting application...")
        self.stop()
        sys.exit(0)

    def _main_loop(self):
        logger.info("Main loop started")

        while self.is_running:
            try:
                audio_chunk = self.audio_processor.get_audio_chunk(timeout=0.5)

                if audio_chunk is None:
                    continue

                # Process audio frame directly with Vosk (like test_copy.py)
                if self.speech_detector:
                    text = self.speech_detector.process_audio_frame(audio_chunk)
                    if text:
                        logger.info(f"[UTTERANCE] {text}")
                        self._process_speech(text)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Main loop stopped")

    def _process_speech(self, text):
        if not text:
            return

        if self.state == AppState.IDLE:
            if self.check_wake_word(text):
                logger.info(f"Wake word detected: {text}")
                self._activate_voice_mode()

        elif self.state == AppState.PUSHED:
            if self.check_release_word(text):
                logger.info(f"Release word detected: {text}")
                self._deactivate_voice_mode()

    def _activate_voice_mode(self):
        try:
            logger.info("Attempting to focus Cursor window...")
            if not self.window_manager.focus_cursor_window():
                logger.warning("Failed to focus Cursor window")
                show_notification("Ошибка", "Не удалось найти окно Cursor IDE")
                return

            # Give window manager more time to actually bring window to foreground and settle
            logger.debug("Waiting for window to come into focus and settle...")
            time.sleep(0.8)

            logger.info("Pressing push-to-talk hotkey...")
            self.key_simulator.press_select_chat_input()
            self.state = AppState.PUSHED

            logger.info("Voice mode activated")
            show_notification("Голосовой режим активирован", f"Скажите '{self.release_word}' для завершения")

        except Exception as e:
            logger.error(f"Error activating voice mode: {e}")
            self.state = AppState.IDLE

    def _deactivate_voice_mode(self):
        try:
            self.key_simulator.release_push_to_talk()
            self.state = AppState.IDLE

            logger.info("Voice mode deactivated")
            show_notification("Голосовой режим деактивирован", f"Ожидание wake word: '{self.wake_word}'")

        except Exception as e:
            logger.error(f"Error deactivating voice mode: {e}")
            self.state = AppState.IDLE

    def _process_state(self):
        pass

    def run(self):
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.quit()


# ===== MAIN ENTRY POINT =====
def parse_args():
    parser = argparse.ArgumentParser(description="Voice Wakeup Cursor - hands-free управление Voice Mode")
    parser.add_argument(
        "--hotkey",
        help="Комбинация для push-to-talk, например 'ctrl+shift+space'",
    )
    parser.add_argument(
        "--wake-word",
        default="джарвис",
        help="Слово для активации Cursor IDE (по умолчанию: 'джарвис')",
    )
    parser.add_argument(
        "--release-word",
        default="погнали",
        help="Слово для отправки промпта (по умолчанию: 'погнали')",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = VoiceWakeupApp(hotkey=args.hotkey, wake_word=args.wake_word, release_word=args.release_word)

    app.start()
    app.run()


if __name__ == "__main__":
    main()
