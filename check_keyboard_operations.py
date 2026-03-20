import logging
import time

from freehands_cursor.keyboard_operations import copy_paste, press_hotkey

logging.basicConfig(level=logging.DEBUG)

N_in_carpalx = "F"
with open("dsatudsaud.txt", "w") as f:
    f.write("Hello, world!")

success = press_hotkey(f"<ctrl>+{N_in_carpalx}", backend="ydotool")
if not success:
    raise Exception("Failed to press hotkey")

time.sleep(0.3)
text = "Hello, world!"
success = copy_paste(0, text, backend="ydotool")
if not success:
    raise Exception("Failed to copy paste")
time.sleep(0.3)
success = copy_paste(len(text), "Привет, мир!", backend="ydotool")
if not success:
    raise Exception("Failed to copy paste")

# press_hotkey("<ctrl>+<shift>+<space>", backend="ydotool")
# time.sleep(0.3)
# press_hotkey("<ctrl>+<shift>+<space>", backend="ydotool")
