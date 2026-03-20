import logging
import subprocess
import time
from enum import StrEnum
from typing import TYPE_CHECKING, assert_never

if TYPE_CHECKING:
    import pynput.keyboard

logger = logging.getLogger(__name__)


class KeyboardBackend(StrEnum):
    PYNPUT = "pynput"
    YDOTOOL = "ydotool"


# ===== PUBLIC API =====
def press_hotkey(hotkey_str: str, backend: KeyboardBackend) -> bool:
    """Press a key combination using the specified backend.
    Returns:
        True if successful, False otherwise
    """
    if backend == KeyboardBackend.PYNPUT:
        return _press_hotkey_pynput(hotkey_str)
    elif backend == KeyboardBackend.YDOTOOL:
        return _press_hotkey_ydotool(hotkey_str)
    else:
        assert_never(backend)


def simulate_typing(delete_prev_chars: int, text: str, backend: KeyboardBackend) -> bool:
    """Simulate typing with the specified backend."""
    if backend == KeyboardBackend.PYNPUT:
        return _simulate_typing_with_pynput(delete_prev_chars, text)
    elif backend == KeyboardBackend.YDOTOOL:
        return _simulate_typing_with_ydotool(delete_prev_chars, text)
    else:
        assert_never(backend)


def copy_paste(delete_prev_chars: int, text: str, backend: KeyboardBackend) -> bool:
    import pyperclip

    if delete_prev_chars:
        simulate_typing(delete_prev_chars=delete_prev_chars, text="", backend=backend)
    was_in_clipboard = pyperclip.paste()
    pyperclip.copy(text)
    time.sleep(0.05)
    success = press_hotkey("<ctrl>+V", backend=backend)
    time.sleep(0.05)
    pyperclip.copy(was_in_clipboard)
    return success


# ===== PYNPUT =====


def _pynput_keyboard_controller() -> "pynput.keyboard.Controller":
    """Return a singleton instance of the keyboard controller"""
    import pynput.keyboard

    if hasattr(_pynput_keyboard_controller, "_instance"):
        return _pynput_keyboard_controller._instance  # pyright: ignore[reportFunctionMemberAccess]
    else:
        _pynput_keyboard_controller._instance = pynput.keyboard.Controller()  # pyright: ignore[reportFunctionMemberAccess]
        return _pynput_keyboard_controller._instance  # pyright: ignore[reportFunctionMemberAccess]


def _press_hotkey_pynput(hotkey_str: str) -> bool:
    """Press a key combination using pynput. Format of hotkey_str: "<ctrl>+<shift>+<space>"
    Returns:
        True if successful, False otherwise
    """
    import pynput.keyboard

    _keyboard_controller = _pynput_keyboard_controller()
    keys = pynput.keyboard.HotKey.parse(hotkey_str)

    valid_keys = [k for k in keys if k]
    if not valid_keys:
        logger.warning("No valid keys provided for hotkey")
        return False

    try:
        # Press all keys
        for key in valid_keys:
            _keyboard_controller.press(key)

        # Release all keys (in reverse order)
        for key in reversed(valid_keys):
            _keyboard_controller.release(key)

        return True
    except Exception as e:
        logger.error(f"Failed to press hotkey '{hotkey_str}' with pynput: {e}", exc_info=True)
        return False


def _simulate_typing_with_pynput(delete_prev_chars: int, text: str) -> bool:
    """Simulate typing with pynput, optionally deleting previous characters."""
    from pynput.keyboard import Key

    _keyboard_controller = _pynput_keyboard_controller()

    if delete_prev_chars:
        # Press backspace multiple times
        for _ in range(delete_prev_chars):
            _keyboard_controller.press(Key.backspace)
            _keyboard_controller.release(Key.backspace)
            time.sleep(0.003)  # Small delay between backspaces

    # Type the text - pynput's type() method handles Unicode/Cyrillic properly
    if text:
        _keyboard_controller.type(text)

    return True


# ===== YDOTOOL =====


def _press_hotkey_ydotool(hotkey_str: str) -> bool:
    """Press a key combination using ydotool. Format of hotkey_str: "<ctrl>+<shift>+<space>"
    Returns:
        True if successful, False otherwise
    """

    def _parse_hotkey_string(hotkey_str: str):
        """Parse hotkey string (e.g., '<ctrl>+<shift>+<space>') into key names."""
        if not hotkey_str:
            return []
        parts = hotkey_str.split("+")
        return [part.strip("<> ").lower() for part in parts]

    def _key_name_to_ydotool_code(key_name: str):
        """Convert key name to ydotool keycode."""
        # Linux input event keycodes
        normalized = key_name.lower().strip()

        # Modifier keys (default to left variant)
        modifier_map = {
            "ctrl": 29,  # Left Ctrl
            "control": 29,
            "shift": 42,  # Left Shift
            "alt": 56,  # Left Alt
            "meta": 125,  # Left Meta/Super
            "super": 125,
            "cmd": 125,
            "command": 125,
        }

        # Special keys
        special_map = {
            "space": 57,
            "enter": 28,
            "return": 28,
            "backspace": 14,
            "delete": 111,
            "esc": 1,
            "escape": 1,
            "tab": 15,
            "up": 103,
            "down": 108,
            "left": 105,
            "right": 106,
            "home": 102,
            "end": 107,
            "page_up": 104,
            "page_down": 109,
            "f1": 59,
            "f2": 60,
            "f3": 61,
            "f4": 62,
            "f5": 63,
            "f6": 64,
            "f7": 65,
            "f8": 66,
            "f9": 67,
            "f10": 68,
            "f11": 87,
            "f12": 88,
        }

        # Letter keys (Linux input event keycodes - QWERTY layout order)
        letter_map = {
            "q": 16,
            "w": 17,
            "e": 18,
            "r": 19,
            "t": 20,
            "y": 21,
            "u": 22,
            "i": 23,
            "o": 24,
            "p": 25,
            "a": 30,
            "s": 31,
            "d": 32,
            "f": 33,
            "g": 34,
            "h": 35,
            "j": 36,
            "k": 37,
            "l": 38,
            "z": 44,
            "x": 45,
            "c": 46,
            "v": 47,
            "b": 48,
            "n": 49,
            "m": 50,
        }

        # Check modifiers first
        if normalized in modifier_map:
            return modifier_map[normalized]

        # Check special keys
        if normalized in special_map:
            return special_map[normalized]

        # Check letters
        if normalized in letter_map:
            return letter_map[normalized]

        # Handle numbers: 0=2, 1=3, ..., 9=11
        if len(normalized) == 1 and "0" <= normalized <= "9":
            return ord(normalized) - ord("0") + 2

        return None

    # Parse hotkey string
    key_names = _parse_hotkey_string(hotkey_str)

    if not key_names:
        logger.warning("No valid keys provided for hotkey")
        return False

    try:
        # Convert key names to ydotool keycodes
        keycodes = []
        for key_name in key_names:
            code = _key_name_to_ydotool_code(key_name)
            logger.debug(f"{key_name} -> {code}")

            if code is None:
                logger.warning(f"Unknown key for ydotool: {key_name}")
                return False

            keycodes.append(code)

        # Build ydotool command: press all keys, then release all keys (in reverse)
        key_sequence = []
        # Press all keys
        for code in keycodes:
            key_sequence.append(f"{code}:1")  # 1 = pressed
        # Release all keys in reverse order
        for code in reversed(keycodes):
            key_sequence.append(f"{code}:0")  # 0 = released

        # Execute ydotool command
        cmd = ["ydotool", "key", "--key-delay", "3", "--"] + key_sequence
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=2, check=False)

        if result.returncode != 0:
            logger.error(f"ydotool failed: {result.returncode}, {result.stdout}, {result.stderr}")
            return False

        return True
    except FileNotFoundError:
        logger.error("ydotool not found. Please install ydotool.")
        return False
    except Exception as e:
        logger.error(f"Failed to press hotkey '{hotkey_str}' with ydotool: {e}", exc_info=True)
        return False


def _run_command(cmd: list[str]) -> bool:
    try:
        subprocess.check_output(cmd)
    except FileNotFoundError as ex:
        logger.error(f"Command {cmd[0]!r} not found: {ex!s}")
        return False
    except subprocess.CalledProcessError as ex:
        logger.error(f"Command {cmd[0]!r} failed: {ex!s}, {ex.returncode}, {ex.output}, {ex.stderr}")
        return False
    return True


# Based on https://github.com/ideasman42/nerd-dictation/blob/main/nerd-dictation#L172
def _simulate_typing_with_ydotool(delete_prev_chars: int, text: str) -> bool:
    cmd = "ydotool"

    if delete_prev_chars:
        # ydotool's key subcommand works with int key IDs and key states. 14 is
        # the linux keycode for the backspace key, and :1 and :0 respectively
        # stand for "pressed" and "released."
        #
        # The key delay is lower than the typing setting because it applies to
        # each key state change (pressed, released).
        success = _run_command(
            [
                cmd,
                "key",
                "--key-delay",
                "3",
                "--",
                *(["14:1", "14:0"] * delete_prev_chars),
            ]
        )
        if not success:
            return False
    # The low delay value makes typing fast, making the output much snappier
    # than the slow default.
    if text:
        success = _run_command(
            [
                cmd,
                "type",
                "--next-delay",
                "5",
                "--",
                text,
            ]
        )
        if not success:
            return False
    return True
