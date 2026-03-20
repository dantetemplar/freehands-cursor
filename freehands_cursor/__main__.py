"""
Voice Wakeup Cursor - hands-free Voice Mode control
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

import numpy as np
import numpy.typing as npt

from freehands_cursor.keyboard_operations import KeyboardBackend, press_hotkey

# ===== CONFIG =====
APP_HOME_DIR = Path.home() / ".freehands_cursor"
APP_HOME_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_HOME_DIR / "freehands_cursor.log"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 4096


# Describes the current display/server environment for platform-specific hotkeys/focusing.
class Environment(StrEnum):
    X11 = "x11"
    GNOME_WAYLAND = "gnome+wayland"
    UNKNOWN_WAYLAND = "unknown-wayland"
    WINDOWS = "windows"
    MACOS = "macos"


def _detect_environment() -> Environment:
    if sys.platform.startswith("win"):
        return Environment.WINDOWS
    if sys.platform == "darwin":
        return Environment.MACOS

    xdg_session_type = os.environ.get("XDG_SESSION_TYPE", "").lower()
    wayland_display = os.environ.get("WAYLAND_DISPLAY")
    x_display = os.environ.get("DISPLAY")

    xdg_current_desktop = os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    desktop_session = os.environ.get("DESKTOP_SESSION", "").upper()

    is_gnome = ("GNOME" in xdg_current_desktop) or ("GNOME" in desktop_session)

    # Prefer explicit session type if available.
    if xdg_session_type == "x11":
        return Environment.X11
    if xdg_session_type == "wayland":
        return Environment.GNOME_WAYLAND if is_gnome else Environment.UNKNOWN_WAYLAND

    # Fallback to environment variables.
    if wayland_display:
        return Environment.GNOME_WAYLAND if is_gnome else Environment.UNKNOWN_WAYLAND
    if x_display:
        return Environment.X11

    # Last resort: treat as an unknown non-X11 graphical environment.
    return Environment.UNKNOWN_WAYLAND


ENVIRONMENT: Environment = _detect_environment()

# GNOME Wayland: extension sort-order needs to be set only once per script run.
# Keep as mutable container to avoid `global` assignment inside functions.
_GNOME_WAYLAND_SORT_ORDER_SET_ONCE = {"value": False}

# ONNX ASR model configuration
DEFAULT_MODEL_NAME = "alphacep/vosk-model-small-ru"

if TYPE_CHECKING:
    # Only needed for static typing; imported lazily at runtime when voice mode starts.
    from onnx_asr.utils import SampleRates


# ===== LOGGING =====
class CleanFormatter(logging.Formatter):
    """Formatter that shortens module names for readability."""

    def format(self, record):
        if record.name.startswith("freehands_cursor."):
            record.name = record.name.replace("freehands_cursor.", "")
        return super().format(record)


file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(CleanFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"))

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler],
)

logger = logging.getLogger("freehands_cursor")


class AppState(StrEnum):
    """Application states."""

    IDLE = "idle"
    CAPTURING = "capturing"


# ===== SPEECH DETECTION =====
class SpeechDetector:
    """Speech recognition with audio capture using ONNX ASR."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        sample_rate: SampleRates = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        # Import heavy/optional runtime dependency only when voice mode is used.
        import onnx_asr
        import sounddevice as sd

        self.sample_rate: SampleRates = sample_rate
        self.chunk_size = chunk_size
        self._sd = sd

        logger.info(f"Loading ONNX ASR model '{model_name}' from Hugging Face...")
        try:
            # onnx-asr automatically downloads models from Hugging Face
            vad = onnx_asr.load_vad("silero")
            self.model = onnx_asr.load_model(model_name, quantization="int8")
            self.model_with_vad = self.model.with_vad(vad)
            logger.info("ONNX ASR model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX ASR model: {e}")
            raise

        # Audio capture
        self.audio_queue: queue.Queue[npt.NDArray[np.float32]] = queue.Queue()
        self.is_recording: bool = False
        self.stream = None

        # Wake word detection buffer (sliding window)
        self.wake_buffer: list[npt.NDArray[np.float32]] = []
        self.wake_buffer_duration: float = 2.0  # Buffer 2 seconds of audio before processing
        self.wake_buffer_samples: int = int(self.sample_rate * self.wake_buffer_duration)
        self.wake_buffer_overlap_duration: float = 0.5  # Keep 0.5 seconds overlap between windows
        self.wake_buffer_overlap_samples: int = int(self.sample_rate * self.wake_buffer_overlap_duration)

        # Session buffer (accumulates audio only when we explicitly capture a command).
        # Keep it disabled by default so wake-word listening doesn't grow memory.
        self.session_buffer: list[npt.NDArray[np.float32]] | None = None

    # ===== AUDIO CAPTURE =====
    def _audio_callback(self, indata, frames, time, status):
        """Audio capture callback."""
        if status:
            logger.warning(f"Audio status: {status}")
        if self.is_recording:
            audio_data: npt.NDArray[np.float32] = indata[:, 0].astype(np.float32)
            try:
                self.audio_queue.put(audio_data.copy(), block=False)
            except queue.Full:
                logger.warning("Audio queue is full, dropping frames")

    def start(self):
        """Start audio capture from microphone."""
        if self.is_recording:
            return
        self.stream = self._sd.InputStream(
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
        """Stop audio capture from microphone."""
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

    # ===== BUFFER MANAGEMENT =====
    def session_started(self):
        """Start a new session, clearing session buffer."""
        self.session_buffer = []
        # Clear wake buffer to avoid interference
        self.wake_buffer = []

    def session_finished(self):
        """Finish the session, clearing the session buffer."""
        self.session_buffer = None
        self.wake_buffer = []

    # ===== SPEECH RECOGNITION =====
    def transcribe_wake_buffer(self, timeout: float = 0.5) -> str | None:
        """Process audio frame for wake word detection and return recognized text if available."""

        try:
            audio_chunk: np.ndarray = self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

        # Ensure audio is float32 and mono
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Clip to valid range
        audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        # Add to wake buffer
        self.wake_buffer.append(audio_chunk)

        # Also add to session buffer if we're in a session
        if self.session_buffer is not None:
            self.session_buffer.append(audio_chunk.copy())

        # Check if we have enough audio to process
        total_samples = sum(len(chunk) for chunk in self.wake_buffer)
        if total_samples < self.wake_buffer_samples:
            return None

        # Concatenate buffer and process
        try:
            waveform: npt.NDArray[np.float32] = np.concatenate(self.wake_buffer)

            # Process with onnx-asr (sample_rate must be one of the supported literal values)
            # Type checker requires literal, but runtime accepts int if it's a valid value
            text = self.model.recognize(waveform, sample_rate=self.sample_rate, language="ru")

            # Keep overlap samples for next window to avoid losing words at boundaries
            if len(waveform) > self.wake_buffer_overlap_samples:
                # Keep last overlap_samples from the waveform
                overlap_audio = waveform[-self.wake_buffer_overlap_samples :]
                self.wake_buffer = [overlap_audio]
            else:
                # If waveform is too short, keep everything
                self.wake_buffer = [waveform]

            if text and text.strip():
                return text.strip().lower()
        except Exception as e:
            logger.debug(f"Error recognizing audio: {e}")
            # Clear buffer on error, but keep some overlap if possible
            if len(self.wake_buffer) > 0:
                last_chunk = self.wake_buffer[-1]
                if len(last_chunk) > self.wake_buffer_overlap_samples:
                    self.wake_buffer = [last_chunk[-self.wake_buffer_overlap_samples :]]
                else:
                    self.wake_buffer = [last_chunk]
            else:
                self.wake_buffer = []

            return None

    def transcribe_session_buffer(self) -> str | None:
        """Transcribe the session buffer and return recognized text if available."""
        if self.session_buffer is None or not self.session_buffer:
            return None

        try:
            # Concatenate all buffered audio
            waveform: npt.NDArray[np.float32] = np.concatenate(self.session_buffer)

            # Need at least 1 second of audio for meaningful transcription
            min_samples = int(self.sample_rate * 1.0)
            if len(waveform) < min_samples:
                return None

            # Process with onnx-asr with VAD
            results = self.model_with_vad.recognize(waveform, sample_rate=self.sample_rate, language="ru")
            logger.info(f"VAD results: {results}")
            text_parts = []
            for result in results:
                text_parts.append(result.text)
            text = " ".join(text_parts)
            return text.strip()
        except Exception as e:
            logger.debug(f"Error recognizing audio buffer: {e}")

        return None


# ===== WINDOW FOCUS =====
def focus_window_flow(name: str) -> bool:
    """Find and focus window. Returns True if successful, False otherwise."""
    # Try to focus window using pywinctl for Windows and X11
    if ENVIRONMENT == Environment.WINDOWS or ENVIRONMENT == Environment.X11 or ENVIRONMENT == Environment.MACOS:
        try:
            import pywinctl as pwc

            # Find windows containing specific text in title
            windows = pwc.getWindowsWithTitle(name, condition=pwc.Re.CONTAINS, flags=re.IGNORECASE)
            logger.info(f"Found {len(windows)} windows with title {name} using pywinctl")

            # Activate the first matching window
            if windows:
                windows[0].activate(True)
                logger.info(f"Activated {name} window using pywinctl")
                return True
        except Exception as e:
            logger.warning(f"pywinctl not available or error: {e}", exc_info=True)

    # Try to focus window using wmctrl for X11
    if ENVIRONMENT == Environment.X11:

        def focus_window_with_wmctrl(window_id: str) -> bool:
            try:
                # Activate window
                subprocess.run(
                    ["wmctrl", "-ia", str(window_id)], capture_output=False, text=True, check=True, timeout=2
                )
                # Raise window
                subprocess.run(
                    ["wmctrl", "-iR", str(window_id)], capture_output=False, text=True, check=True, timeout=2
                )
                logger.info(f"Focused {name} window using wmctrl")
                return True
            except Exception as e:
                logger.warning(f"wmctrl not available or error: {e}", exc_info=True)
                return False

        try:
            result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if name.lower() in line.lower():
                        window_id = line.split()[0]
                        logger.info(f"Found {name} window via wmctrl: {window_id}")
                        return focus_window_with_wmctrl(window_id)

        except Exception as e:
            logger.warning(f"wmctrl not available or error: {e}", exc_info=True)

    # Try to focus window via Gnome Extensions
    if ENVIRONMENT == Environment.GNOME_WAYLAND:
        # Uses the GNOME Shell extension described in README:
        # - activateBySubstring: focuses window by substring in title
        # - setSortOrder(highest_user_time): always prefer the most recently used match
        try:
            if not _GNOME_WAYLAND_SORT_ORDER_SET_ONCE["value"]:
                res = subprocess.run(
                    [
                        "gdbus",
                        "call",
                        "--session",
                        "--dest",
                        "org.gnome.Shell",
                        "--object-path",
                        "/de/lucaswerkmeister/ActivateWindowByTitle",
                        "--method",
                        "de.lucaswerkmeister.ActivateWindowByTitle.setSortOrder",
                        "highest_user_time",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=3,
                )
                if res.returncode == 0:
                    logger.debug("GNOME extension sort order set to highest_user_time")
                else:
                    logger.debug(
                        "GNOME extension setSortOrder failed (rc=%s): %s",
                        res.returncode,
                        (res.stderr or res.stdout or "").strip(),
                    )
                _GNOME_WAYLAND_SORT_ORDER_SET_ONCE["value"] = True

            res = subprocess.run(
                [
                    "gdbus",
                    "call",
                    "--session",
                    "--dest",
                    "org.gnome.Shell",
                    "--object-path",
                    "/de/lucaswerkmeister/ActivateWindowByTitle",
                    "--method",
                    "de.lucaswerkmeister.ActivateWindowByTitle.activateBySubstring",
                    name,
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res.returncode != 0:
                logger.debug(
                    "GNOME extension activateBySubstring failed (rc=%s): %s",
                    res.returncode,
                    (res.stderr or res.stdout or "").strip(),
                )
            else:
                out = (res.stdout or "") + (res.stderr or "")
                out_lower = out.lower()
                if "false" in out_lower:
                    success = False
                elif "true" in out_lower:
                    success = True
                else:
                    # If extension didn't return an explicit boolean, consider it a failure.
                    # This avoids false-positive "success" when GNOME extension can't find the window.
                    success = False
                    logger.debug(
                        "GNOME extension activateBySubstring returned no boolean (rc=%s). Output: %r",
                        res.returncode,
                        out.strip(),
                    )

                if success:
                    logger.info("Focused %s window via GNOME extension", name)
                    return True

        except FileNotFoundError:
            logger.warning("gdbus not found; GNOME focus via extension is unavailable")
        except Exception as e:
            logger.warning("Failed to focus window via GNOME extension: %s", e, exc_info=True)

    return False


def normalize_word(text: str) -> str:
    """Normalize word by stripping punctuation and converting to lowercase."""
    return text.strip().strip(".,!?\"'").lower()


# ===== NOTIFICATION =====
def show_notification(title, message):
    """Show system notification via notify-send."""
    try:
        subprocess.run(["notify-send", title, message], capture_output=True, check=False)
    except Exception as e:
        logger.debug(f"Failed to show notification: {e}")


# ===== MAIN APPLICATION =====
class VoiceWakeupApp:
    """Main application class."""

    def __init__(
        self,
        toggle_voice_mode_hotkey: str | None = None,
        focus_agent_hotkey: str | None = None,
        wake_word: str = "джарвис",
        submit_keyword: str = "газ",
        model_name: str | None = None,
        keyboard_backend: KeyboardBackend | None = None,
    ):
        """
        Args:
            hotkey: Cursor hotkey to activate voice mode (default: "Ctrl+Shift+Space").
                    Format should work with both backends, e.g. "<ctrl>+<shift>+<space>" or "Ctrl+Shift+Space".
        """
        self.state = AppState.IDLE
        self.is_running = False
        self.thread = None
        self._stop_event = threading.Event()

        self.wake_word = wake_word.lower().strip()
        self.submit_keyword = submit_keyword.lower().strip()
        self.model_name = model_name
        self.toggle_voice_mode_hotkey_str = toggle_voice_mode_hotkey or "Ctrl+Shift+Space"
        self.focus_agent_hotkey_str = focus_agent_hotkey or "<ctrl>+<shift>+u"
        self._last_trigger_time = 0.0
        # Prevent repeated triggers from a single wake-phrase.
        self._trigger_cooldown_seconds = 1.5

        if keyboard_backend is None:
            if ENVIRONMENT == Environment.GNOME_WAYLAND or ENVIRONMENT == Environment.UNKNOWN_WAYLAND:
                keyboard_backend = KeyboardBackend.YDOTOOL
            else:
                keyboard_backend = KeyboardBackend.PYNPUT

        self.keyboard_backend = keyboard_backend

        # Initialize speech detector
        model_name = model_name or DEFAULT_MODEL_NAME
        self.speech_detector = SpeechDetector(model_name=model_name, chunk_size=DEFAULT_CHUNK_SIZE)
        logger.info("Voice Wakeup Cursor initialized (wake='%s')", self.wake_word)
        logger.info("Cursor voice hotkey: %s", self.toggle_voice_mode_hotkey_str)
        logger.info("Submit keyword: %s", self.submit_keyword)
        logger.info("Cursor Focus Agent hotkey (composer.focusComposer): %s", self.focus_agent_hotkey_str)

    def start(self):
        if self.is_running:
            logger.warning("Application already running")
            return

        try:
            self.speech_detector.start()

            self.is_running = True
            self.thread = threading.Thread(target=self.main_loop, daemon=True)
            self.thread.start()

            self.state = AppState.IDLE
            logger.info("Application started")

            show_notification("Application started", f"Waiting for wake word: '{self.wake_word}'")
            logger.info("Pressing Cursor hotkey on wake word: %s", self.toggle_voice_mode_hotkey_str)

        except Exception:
            logger.error("Failed to start application", exc_info=True)
            raise

    def run(self):
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.quit()

    def stop(self):
        if not self.is_running:
            return

        self.is_running = False
        self._stop_event.set()
        self.speech_detector.stop()
        self.state = AppState.IDLE
        logger.info("Application stopped")
        show_notification("Application stopped", "")

    def quit(self):
        logger.info("Quitting application...")
        self.stop()
        sys.exit(0)

    def main_loop(self):
        logger.info("Main loop started")

        while self.is_running:
            try:
                if self.state == AppState.IDLE:
                    text = self.speech_detector.transcribe_wake_buffer(timeout=0.5)

                    if text and len(text) > 1:  # ignore single character utterances (mostly noise)
                        logger.info(f"[UTTERANCE] {text}")

                    if self.check_submit_keyword(text):
                        logger.info("Submit keyword detected: %s", text)
                        now = time.time()
                        if now - self._last_trigger_time >= self._trigger_cooldown_seconds:
                            if self.focus_chat_input_and_submit():
                                self._last_trigger_time = now
                            else:
                                logger.error("Failed to trigger Cursor Focus Agent + submit hotkeys")
                                show_notification("Submit failed", "Could not trigger Cursor Focus Agent and submit")
                    elif self.check_wake_word(text):
                        logger.info("Wake word detected: %s", text)
                        now = time.time()
                        if now - self._last_trigger_time >= self._trigger_cooldown_seconds:
                            if self.focus_cursor_window_and_chat_input():
                                self._last_trigger_time = now
                            else:
                                logger.error("Failed to trigger Cursor hotkey on wake word")
                                show_notification("Voice mode activation failed", "Could not focus Cursor chat input")

                elif self.state == AppState.CAPTURING:
                    # Legacy placeholder: CAPTURING is no longer used, but keep the
                    # branch to satisfy type checking and older state transitions.
                    self.state = AppState.IDLE
                else:
                    assert_never(self.state)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(0.1)

        logger.info("Main loop stopped")

    def check_wake_word(self, text):
        """Check if text contains wake word."""
        if not text:
            return False
        normalized_tokens = [normalize_word(part) for part in text.split()]
        return self.wake_word in normalized_tokens

    def check_submit_keyword(self, text):
        """Check if text contains submit keyword."""
        if not text:
            return False
        normalized_tokens = [normalize_word(part) for part in text.split()]
        return self.submit_keyword in normalized_tokens

    def focus_cursor_window_and_chat_input(self) -> bool:
        try:
            logger.info("Focusing Cursor window...")
            if not focus_window_flow(name="Cursor"):
                logger.error("Could not find Cursor IDE window")
                show_notification("Error", "Could not find Cursor IDE window")
                return False

            if not press_hotkey(self.focus_agent_hotkey_str, backend=self.keyboard_backend):
                logger.error("Could not press Cursor Focus Agent hotkey before voice mode")
                show_notification("Error", "Could not press Cursor Focus Agent hotkey")
                return False

            # Give Cursor a moment to apply focus to the composer before voice mode.
            time.sleep(0.05)
            if not press_hotkey(self.toggle_voice_mode_hotkey_str, backend=self.keyboard_backend):
                logger.error("Could not press Cursor hotkey")
                show_notification("Error", "Could not press Cursor hotkey")
                return False
        except Exception as err:
            logger.error("Failed to press Cursor Focus Agent + voice mode hotkeys: %s", err)
            return False

        return True

    def focus_chat_input_and_submit(self) -> bool:
        try:
            logger.info("Focusing chat input and submitting...")
            if not press_hotkey(self.focus_agent_hotkey_str, backend=self.keyboard_backend):
                logger.error("Could not press Cursor Focus Agent hotkey")
                show_notification("Error", "Could not press Cursor Focus Agent hotkey")
                return False

            # Small pause helps ensure composer input is focused before submit.
            time.sleep(0.05)
            if not press_hotkey("<enter>", backend=self.keyboard_backend):
                logger.error("Could not press Enter for submit")
                show_notification("Error", "Could not press Enter for submit")
                return False
        except Exception as err:
            logger.error("Failed to press Cursor Focus Agent + submit hotkeys: %s", err)
            return False

        return True


# ===== MAIN ENTRY POINT =====
def main():
    parser = argparse.ArgumentParser(description="Voice Wakeup Cursor - hands-free Voice Mode control")
    parser.add_argument(
        "--toggle-voice-mode-hotkey",
        default="<ctrl>+<shift>+<space>",
        help="Hotkey to trigger Cursor voice mode (default: <ctrl>+<shift>+<space>). Note that you should disable When property for composer.toggleVoiceDictation keybind if you want focus chat input from terminal. Format: '<ctrl>+<shift>+a'",
    )
    parser.add_argument(
        "--wake-word",
        default="джарвис",
        help="Word to activate Cursor IDE (default: 'джарвис')",
    )
    parser.add_argument(
        "--submit-keyword",
        default="газ",
        help="Word that will submit prompt (presses Enter) (default: 'газ')",
    )
    parser.add_argument(
        "--focus-agent-hotkey",
        default="<ctrl>+<shift>+y",
        help="Hotkey for Cursor Focus Agent (command: composer.focusComposer) (default: <ctrl>+<shift>+y)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"ONNX ASR model name from Hugging Face (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--dry-run-focus",
        action="store_true",
        help="Only focus the Cursor window and exit. No voice mode starts.",
    )
    parser.add_argument(
        "--dry-run-hotkeys",
        action="store_true",
        help="Only press Cursor voice-mode hotkey once and exit. No voice mode starts.",
    )
    args = parser.parse_args()

    if args.dry_run_focus:
        success = focus_window_flow("Cursor")
        logger.info("Dry-run focus completed (success=%s)", success)
        sys.exit(0 if success else 1)
    elif args.dry_run_hotkeys:
        if ENVIRONMENT == Environment.GNOME_WAYLAND or ENVIRONMENT == Environment.UNKNOWN_WAYLAND:
            keyboard_backend = KeyboardBackend.YDOTOOL
        else:
            keyboard_backend = KeyboardBackend.PYNPUT

        focus_agent_success = press_hotkey(args.focus_agent_hotkey, backend=keyboard_backend)
        toggel_voice_mode_success = press_hotkey(args.toggle_voice_mode_hotkey, backend=keyboard_backend)

        time.sleep(3)
        submit_success = press_hotkey("<enter>", backend=keyboard_backend)
        logger.info(
            "Dry-run voice hotkey completed (focus_agent_success=%s toggel_voice_mode_success=%s submit_success=%s)",
            focus_agent_success,
            toggel_voice_mode_success,
            submit_success,
        )
        sys.exit(0 if focus_agent_success and toggel_voice_mode_success and submit_success else 1)

    app = VoiceWakeupApp(
        toggle_voice_mode_hotkey=args.toggle_voice_mode_hotkey,
        focus_agent_hotkey=args.focus_agent_hotkey,
        wake_word=args.wake_word,
        submit_keyword=args.submit_keyword,
        model_name=args.model_name,
    )

    app.start()
    app.run()
