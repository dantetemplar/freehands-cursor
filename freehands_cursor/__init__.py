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
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path

import numpy as np
import psutil
import sounddevice as sd
from pynput import mouse
from pynput.keyboard import Controller, HotKey, Key, Listener as KeyboardListener
from vosk import KaldiRecognizer, Model, SetLogLevel

try:
    import pyperclip
except ImportError:  # pragma: no cover - optional dependency
    pyperclip = None

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
    CAPTURING = "capturing"


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
    """Имитация нажатий горячих клавиш."""

    DEFAULT_COMBOS = {
        "Darwin": "<cmd>+<space>",
        "Windows": "<ctrl>+<shift>+<space>",
        "Linux": "<ctrl>+<shift>+<space>",
    }

    DISPLAY_MAP = {
        Key.cmd: "Cmd",
        Key.ctrl: "Ctrl",
        Key.shift: "Shift",
        Key.alt: "Alt",
        Key.space: "Space",
    }

    KEY_ALIASES = {
        "command": "cmd",
        "win": "cmd",
        "super": "cmd",
        "meta": "cmd",
        "control": "ctrl",
        "option": "alt",
        "spacebar": "space",
    }

    def __init__(self, combo_text: str | None = None):
        self.system = platform.system()
        self.keyboard = Controller()
        self.combo_text = combo_text or self.DEFAULT_COMBOS.get(self.system, "<ctrl>+<shift>+<space>")
        self.key_sequence = self._parse_combo_text(self.combo_text)
        logger.info("Комбинация горячих клавиш: %s", self.get_key_display())

    def _parse_combo_text(self, combo_text: str):
        normalized = self._normalize_combo(combo_text)
        try:
            sequence = list(HotKey.parse(normalized))
        except ValueError:
            fallback = self.DEFAULT_COMBOS.get(self.system, "<ctrl>+<shift>+<space>")
            logger.warning("Не удалось разобрать комбинацию '%s', используем %s", combo_text, fallback)
            normalized = self._normalize_combo(fallback)
            sequence = list(HotKey.parse(normalized))
        self.combo_text = normalized
        return sequence

    def _normalize_combo(self, combo_text: str):
        tokens = [token.strip() for token in combo_text.replace(" ", "").split("+") if token.strip()]
        normalized_parts = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower.startswith("<") and token_lower.endswith(">"):
                normalized_parts.append(token_lower)
                continue
            alias = self.KEY_ALIASES.get(token_lower, token_lower)
            if len(alias) == 1:
                normalized_parts.append(alias)
            else:
                normalized_parts.append(f"<{alias}>")
        return "+".join(normalized_parts)

    def tap(self, times=1, delay_between=0.2):
        for index in range(times):
            self._press_sequence()
            self._release_sequence()
            if index < times - 1:
                time.sleep(delay_between)

    def _press_sequence(self):
        for key in self.key_sequence:
            self.keyboard.press(key)
            time.sleep(0.01)

    def _release_sequence(self):
        for key in reversed(self.key_sequence):
            self.keyboard.release(key)
            time.sleep(0.01)

    def get_key_display(self):
        parts = []
        for key in self.key_sequence:
            if key in self.DISPLAY_MAP:
                parts.append(self.DISPLAY_MAP[key])
            else:
                char = getattr(key, "char", None)
                if char:
                    parts.append(char.upper())
                else:
                    parts.append(str(key))
        return "+".join(parts)


# ===== CLIPBOARD & USER ACTIVITY =====
class ClipboardManager:
    """Сохранение последнего сообщения в буфер."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        if self.enabled and pyperclip is None:
            logger.warning("pyperclip недоступен, сохранение в буфер отключено")
            self.enabled = False

    def save(self, text: str):
        if not self.enabled:
            return
        sanitized = (text or "").strip()
        if not sanitized:
            return
        try:
            pyperclip.copy(sanitized)
            logger.info("Сообщение сохранено в буфер обмена")
        except Exception as err:  # pragma: no cover - системный буфер может отсутствовать
            logger.warning("Не удалось сохранить сообщение в буфер: %s", err)


class UserActivityMonitor:
    """Отслеживание пользовательского ввода для остановки автоматической печати."""

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.keyboard_listener = None
        self.mouse_listener = None
        self.active = False
        self._interrupted = False
        self._enter_pressed = False
        self._ignore_depth = 0
        self._lock = threading.Lock()

    def start(self):
        if not self.enabled:
            return
        if self.keyboard_listener is None:
            self.keyboard_listener = KeyboardListener(on_press=self._on_key_press)
            self.keyboard_listener.start()
        if self.mouse_listener is None:
            self.mouse_listener = mouse.Listener(on_click=self._on_mouse_click)
            self.mouse_listener.start()

    def shutdown(self):
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None
        if self.mouse_listener:
            self.mouse_listener.stop()
            self.mouse_listener = None

    def begin_session(self):
        if not self.enabled:
            return
        with self._lock:
            self.active = True
            self._interrupted = False
            self._enter_pressed = False

    def end_session(self):
        if not self.enabled:
            return
        with self._lock:
            self.active = False

    @property
    def was_interrupted(self):
        if not self.enabled:
            return False
        with self._lock:
            return self._interrupted

    def consume_enter_pressed(self):
        if not self.enabled:
            return False
        with self._lock:
            if self._enter_pressed:
                self._enter_pressed = False
                return True
        return False

    @contextmanager
    def simulated_input(self):
        if not self.enabled:
            yield
            return
        self._increment_ignore_depth()
        try:
            yield
        finally:
            self._decrement_ignore_depth()

    def _on_key_press(self, key):
        if not self.enabled:
            return
        with self._lock:
            if not self.active or self._ignore_depth > 0:
                return
            self._interrupted = True
            if key == Key.enter:
                self._enter_pressed = True

    def _on_mouse_click(self, _x, _y, _button, pressed):
        if not pressed:
            return
        if not self.enabled:
            return
        with self._lock:
            if not self.active or self._ignore_depth > 0:
                return
            self._interrupted = True

    def _increment_ignore_depth(self):
        with self._lock:
            self._ignore_depth += 1

    def _decrement_ignore_depth(self):
        with self._lock:
            self._ignore_depth = max(0, self._ignore_depth - 1)


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

    def __init__(
        self,
        hotkey=None,
        wake_word="джарвис",
        submit_word="погнали",
        save_to_buffer=True,
        guard_user_input=True,
        cursor_voice_input=False,
    ):
        self.state = AppState.IDLE
        self.is_running = False
        self.thread = None

        self.wake_word = wake_word.lower().strip()
        self.submit_word = submit_word.lower().strip()
        self.cursor_voice_input = bool(cursor_voice_input)
        self.clipboard = ClipboardManager(enabled=save_to_buffer)
        self.user_monitor = UserActivityMonitor(enabled=guard_user_input)

        self.audio_processor = AudioProcessor()
        self.speech_detector = None
        self.window_manager = WindowManager()
        self.key_simulator = KeySimulator(combo_text=hotkey)
        self.keyboard = Controller()
        self.current_message_tokens = []
        self._typing_enabled = True

        logger.info(
            "Voice Wakeup Cursor initialized (wake='%s', submit='%s', cursor_voice=%s)",
            self.wake_word,
            self.submit_word,
            self.cursor_voice_input,
        )

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

    @staticmethod
    def _normalize_word(text):
        return text.strip().strip(".,!?\"'").lower()

    def check_wake_word(self, text):
        """Check if text contains wake word."""
        if not text:
            return False
        normalized_tokens = [self._normalize_word(part) for part in text.split()]
        return self.wake_word in normalized_tokens

    def check_submit_word(self, text):
        """Check if text contains submit word."""
        if not text:
            return False
        normalized_tokens = [self._normalize_word(part) for part in text.split()]
        return self.submit_word in normalized_tokens

    def start(self):
        if self.is_running:
            logger.warning("Application already running")
            return

        try:
            if self.speech_detector is None:
                self.initialize_speech_detector()

            self.audio_processor.start()
            self.user_monitor.start()

            self.is_running = True
            self.thread = threading.Thread(target=self._main_loop, daemon=True)
            self.thread.start()

            self.state = AppState.IDLE
            logger.info("Application started")

            show_notification("Приложение запущено", f"Ожидание wake word: '{self.wake_word}'")
            logger.info("Используемая горячая клавиша: %s", self.key_simulator.get_key_display())

        except Exception:
            logger.error("Failed to start application", exc_info=True)
            raise

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        self.user_monitor.end_session()
        self.audio_processor.stop()
        self.user_monitor.shutdown()

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

        tokens = [token.strip() for token in text.split() if token.strip()]
        if not tokens:
            return

        if self.state == AppState.IDLE:
            if self.check_wake_word(text):
                logger.info("Wake word detected: %s", text)
                if self._start_session():
                    start_index = self._find_word_index(tokens, self.wake_word)
                    self._consume_tokens(tokens[start_index:])
        elif self.state == AppState.CAPTURING:
            self._consume_tokens(tokens)

    def run(self):
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.quit()

    def _start_session(self):
        if not self._prepare_chat_input():
            logger.warning("Не удалось подготовить окно Cursor")
            return False
        self.current_message_tokens = []
        self._typing_enabled = not self.cursor_voice_input
        self.state = AppState.CAPTURING
        self.user_monitor.begin_session()
        show_notification("Голосовой режим активирован", f"Скажите '{self.submit_word}' для завершения")
        logger.info("Начали новую голосовую сессию")
        return True

    def _prepare_chat_input(self):
        try:
            logger.info("Фокусируем окно Cursor...")
            if not self.window_manager.focus_cursor_window():
                show_notification("Ошибка", "Не удалось найти окно Cursor IDE")
                return False

            time.sleep(0.4)
            if self.cursor_voice_input:
                logger.info("Запускаем Cursor Voice Input")
                self.key_simulator.tap(times=1)
            else:
                logger.info("Двойное нажатие горячей клавиши для фокуса чата")
                self.key_simulator.tap(times=1)
                time.sleep(0.25)
                self.key_simulator.tap(times=1)
            time.sleep(0.1)
            return True
        except Exception as err:
            logger.error("Не удалось нажать горячую клавишу: %s", err)
            return False

    def _consume_tokens(self, tokens):
        if not tokens:
            return
        if self._handle_user_enter():
            return

        if self._typing_enabled and self.user_monitor.was_interrupted:
            self._typing_enabled = False
            logger.info("Пользователь вмешался, автоматическая печать отключена")

        for token in tokens:
            is_submit_word = self.check_submit_word(token)
            self._append_word(token, is_submit_word=is_submit_word)

            if is_submit_word:
                logger.info("Submit word detected: %s", token)
                self._finalize_session(reason="submit-word")
                break

            if self._handle_user_enter():
                break

    def _handle_user_enter(self):
        if self.user_monitor.consume_enter_pressed():
            logger.info("Пользователь нажал Enter, завершаем ввод")
            self._finalize_session(reason="user-enter")
            return True
        return False

    def _append_word(self, token, is_submit_word=False):
        cleaned = token.strip()
        if not cleaned:
            return
        self.current_message_tokens.append(cleaned)

        if self._should_type():
            suffix = "" if is_submit_word else " "
            self._type_text(cleaned + suffix)

    def _should_type(self):
        return self._typing_enabled and not self.cursor_voice_input

    def _type_text(self, text):
        if not text:
            return
        with self.user_monitor.simulated_input():
            for char in text:
                self.keyboard.press(char)
                self.keyboard.release(char)
                time.sleep(0.01)

    def _press_enter(self):
        with self.user_monitor.simulated_input():
            self.keyboard.press(Key.enter)
            self.keyboard.release(Key.enter)
            time.sleep(0.05)

    def _finalize_session(self, reason):
        message = " ".join(self.current_message_tokens).strip()
        if message:
            logger.info("Фраза завершена (%s): %s", reason, message)
            self.clipboard.save(message)
        else:
            logger.info("Сессия завершена без текста (%s)", reason)

        if self.cursor_voice_input:
            try:
                logger.info("Отключаем Cursor Voice Input")
                self.key_simulator.tap(times=1)
            except Exception as err:
                logger.warning("Не удалось отключить Cursor Voice Input: %s", err)

        if self._should_press_enter(reason):
            self._press_enter()

        self.user_monitor.end_session()
        self.current_message_tokens = []
        self.state = AppState.IDLE
        self._typing_enabled = True
        show_notification("Голосовой режим деактивирован", f"Ожидание wake word: '{self.wake_word}'")

    def _should_press_enter(self, reason):
        if reason == "user-enter":
            return False
        if self.user_monitor.was_interrupted:
            return False
        if self.cursor_voice_input:
            return True
        return self._typing_enabled

    def _find_word_index(self, tokens, target):
        for idx, token in enumerate(tokens):
            if self._normalize_word(token) == target:
                return idx
        return 0


# ===== MAIN ENTRY POINT =====
def parse_args():
    parser = argparse.ArgumentParser(description="Voice Wakeup Cursor - hands-free управление Voice Mode")
    parser.add_argument(
        "--hotkey",
        help="Комбинация для push-to-talk в формате HotKey.parse, например '<ctrl>+<shift>+space'",
    )
    parser.add_argument(
        "--wake-word",
        default="джарвис",
        help="Слово для активации Cursor IDE (по умолчанию: 'джарвис')",
    )
    parser.add_argument(
        "--submit-word",
        default="погнали",
        help="Слово для отправки промпта (по умолчанию: 'погнали')",
    )
    parser.add_argument(
        "--save-to-buffer",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Сохранять последнее сообщение в буфер обмена (по умолчанию: включено)",
    )
    parser.add_argument(
        "--user-interrupt-guard",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Останавливать автопечать при действиях пользователя (по умолчанию: включено)",
    )
    parser.add_argument(
        "--cursor-voice-input",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Использовать встроенный Cursor Voice Input вместо печати (по умолчанию: выключено)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    app = VoiceWakeupApp(
        hotkey=args.hotkey,
        wake_word=args.wake_word,
        submit_word=args.submit_word,
        save_to_buffer=args.save_to_buffer,
        guard_user_input=args.user_interrupt_guard,
        cursor_voice_input=args.cursor_voice_input,
    )

    app.start()
    app.run()


if __name__ == "__main__":
    main()
