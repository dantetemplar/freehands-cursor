"""
Voice Wakeup Cursor - hands-free Voice Mode control
"""

import argparse
import logging
import queue
import subprocess
import sys
import threading
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt
import onnx_asr
import psutil
import sounddevice as sd
from onnx_asr.utils import SampleRates

from freehands_cursor.keyboard_operations import KeyboardBackend, copy_paste, press_hotkey, press_hotkey_twice

# ===== CONFIG =====
APP_HOME_DIR = Path.home() / ".freehands_cursor"
APP_HOME_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = APP_HOME_DIR / "freehands_cursor.log"

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_SIZE = 4096

# ONNX ASR model configuration
DEFAULT_MODEL_NAME = "alphacep/vosk-model-small-ru"


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
        self.sample_rate: SampleRates = sample_rate
        self.chunk_size = chunk_size

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
        self.stream: sd.InputStream | None = None

        # Wake word detection buffer (sliding window)
        self.wake_buffer: list[npt.NDArray[np.float32]] = []
        self.wake_buffer_duration: float = 2.0  # Buffer 2 seconds of audio before processing
        self.wake_buffer_samples: int = int(self.sample_rate * self.wake_buffer_duration)
        self.wake_buffer_overlap_duration: float = 0.5  # Keep 0.5 seconds overlap between windows
        self.wake_buffer_overlap_samples: int = int(self.sample_rate * self.wake_buffer_overlap_duration)

        # Session buffer (accumulates audio from wake word to submit word)
        self.session_buffer: list[npt.NDArray[np.float32]] = []

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
        self.session_buffer = []
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
        if not self.session_buffer:
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
    window = None

    # Try wmctrl first
    try:
        result = subprocess.run(["wmctrl", "-l"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if name.lower() in line.lower():
                    window_id = line.split()[0]
                    window = window_id
                    logger.info(f"Found {name} window via wmctrl: {window_id}")
                    break
    except Exception as e:
        logger.debug(f"wmctrl not available or error: {e}")

    # Fallback to process search
    if not window:
        try:
            for proc in psutil.process_iter(["pid", "name", "exe"]):
                try:
                    name = proc.info["name"] or ""
                    exe = proc.info["exe"] or ""
                    if name.lower() in name.lower() or name.lower() in exe.lower():
                        logger.info(f"Found Cursor process: {proc.info['pid']}")
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            logger.error(f"Error finding {name} process: {e}")

    if not window:
        logger.warning(f"{name} window not found")
        return False

    # Use wmctrl to focus window
    try:
        # First raise the window
        subprocess.run(["wmctrl", "-ia", str(window)], capture_output=True, text=True, check=False, timeout=2)
        # Then activate it
        subprocess.run(["wmctrl", "-iR", str(window)], capture_output=True, text=True, check=False, timeout=2)
        logger.info(f"Focused {name} window using wmctrl")
        return True
    except Exception as e:
        logger.error(f"Failed to focus {name} window: {e}")
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
        hotkey: str | None = None,
        wake_word: str = "джарвис",
        submit_word: str = "погнали",
        model_name: str | None = None,
        keyboard_backend: KeyboardBackend = KeyboardBackend.YDOTOOL,
    ):
        """
        Args:
            hotkey: Hotkey for enabling Voice Mode in Cursor and focusing on chat window.
                   Default: "Ctrl+Shift+Space"
        """
        self.state = AppState.IDLE
        self.is_running = False
        self.thread = None
        self._stop_event = threading.Event()

        self.wake_word = wake_word.lower().strip()
        self.submit_word = submit_word.lower().strip()
        self.model_name = model_name
        self.keyboard_backend = keyboard_backend

        # Initialize speech detector
        model_name = model_name or DEFAULT_MODEL_NAME
        self.speech_detector = SpeechDetector(model_name=model_name, chunk_size=DEFAULT_CHUNK_SIZE)

        # Hotkey for enabling Voice Mode in Cursor and focusing on chat window
        self.hotkey_str = hotkey or "Ctrl+Shift+Space"

        # Update timing
        self.last_update_time = 0  # Timestamp of last transcription update
        self.update_interval = 2.5  # Update every 2.5 seconds
        self.current_transcription = ""  # Current transcription text

        logger.info(
            "Voice Wakeup Cursor initialized (wake='%s', submit='%s')",
            self.wake_word,
            self.submit_word,
        )
        logger.info("Hotkey combination for Cursor Voice Mode: %s", self.hotkey_str)

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
            logger.info("Hotkey used for Cursor Voice Mode: %s", self.hotkey_str)

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
                # In IDLE state, check for wake word and start session if positive
                if self.state == AppState.IDLE:
                    text = self.speech_detector.transcribe_wake_buffer(timeout=0.5)

                    if text and len(text) > 1:  # ignore single character utterances (mostly noise)
                        logger.info(f"[UTTERANCE] {text}")

                    if self.check_wake_word(text):
                        logger.info("Wake word detected: %s", text)
                        self.start_session()

                # In CAPTURING state, check if we need periodic update because
                #  it will require transcription of the full buffer (demanding operation)
                elif self.state == AppState.CAPTURING:
                    current_time = time.time()
                    if current_time - self.last_update_time >= self.update_interval:
                        self.transcribe_session_fully()
                        self.last_update_time = time.time()

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

    def check_submit_word(self, text):
        """Check if text contains submit word."""
        if not text:
            return False
        normalized_tokens = [normalize_word(part) for part in text.split()]
        return self.submit_word in normalized_tokens

    def start_session(self):
        if not self.focus_cursor_window_and_chat_input():
            return False
        # Reset transcription state and start new session
        self.current_transcription = ""
        self.last_update_time = time.time()
        self.speech_detector.session_started()
        self.state = AppState.CAPTURING
        show_notification("Voice mode activated", f"Say '{self.submit_word}' to finish")
        logger.info("Started new voice session")
        return True

    def focus_cursor_window_and_chat_input(self) -> bool:
        try:
            logger.info("Focusing Cursor window...")
            if not focus_window_flow(name="cursor"):
                logger.error("Could not find Cursor IDE window")
                show_notification("Error", "Could not find Cursor IDE window")
                return False

            if not press_hotkey_twice(self.hotkey_str, backend=self.keyboard_backend):
                logger.error("Could not focus chat input")
                show_notification("Error", "Could not focus chat input")
                return False
        except Exception as err:
            logger.error("Failed to press hotkey: %s", err)
            return False

        return True

    def transcribe_session_fully(self):
        """Process the full audio buffer and update the input field."""
        try:
            # Transcribe session buffer with speech detector
            transcription = self.speech_detector.transcribe_session_buffer()

            if transcription:
                logger.info(f"[FULL TRANSCRIPTION] {transcription}")
                # Update chat input with new transcription
                self.update_chat_input(transcription)

                # Check for submit word in transcription and finish session if positive
                if self.check_submit_word(transcription):
                    logger.info("Submit word detected in transcription")
                    self.finish_session(reason="submit-word")
                    return

        except Exception as e:
            logger.error(f"Error processing full buffer: {e}", exc_info=True)

    def update_chat_input(self, text: str):
        """Update the input field by deleting previous text and typing new text."""
        if not text:
            return

        # Count characters to delete (current transcription length)
        delete_prev_chars = len(self.current_transcription)
        # Delete previous input and set new text
        copy_paste(delete_prev_chars, text, backend=self.keyboard_backend)
        self.current_transcription = text

    def finish_session(self, reason: Literal["submit-word"]):
        # Finish session
        self.speech_detector.session_finished()
        message = self.current_transcription.strip()
        if message:
            logger.info("Phrase completed (%s): %s", reason, message)
        else:
            logger.info("Session ended without text (%s)", reason)

        # Press Enter only on submit-word
        if reason == "submit-word":
            press_hotkey("<enter>", backend=self.keyboard_backend)

        # Reset state
        self.current_transcription = ""
        self.state = AppState.IDLE
        show_notification("Voice mode deactivated", f"Waiting for wake word: '{self.wake_word}'")


# ===== MAIN ENTRY POINT =====
def main():
    parser = argparse.ArgumentParser(description="Voice Wakeup Cursor - hands-free Voice Mode control")
    parser.add_argument(
        "--hotkey",
        help="Hotkey for enabling Voice Mode in Cursor and focusing on chat window. "
        "Format: 'Ctrl+Shift+Space' (example: 'Ctrl+Shift+Space')",
    )
    parser.add_argument(
        "--wake-word",
        default="джарвис",
        help="Word to activate Cursor IDE (default: 'джарвис')",
    )
    parser.add_argument(
        "--submit-word",
        default="погнали",
        help="Word to submit prompt (default: 'погнали')",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=f"ONNX ASR model name from Hugging Face (default: {DEFAULT_MODEL_NAME})",
    )
    args = parser.parse_args()
    app = VoiceWakeupApp(
        hotkey=args.hotkey,
        wake_word=args.wake_word,
        submit_word=args.submit_word,
        model_name=args.model_name,
    )

    app.start()
    app.run()
