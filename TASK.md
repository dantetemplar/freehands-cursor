### Small refactoring

Use HotKey.parse to parse keybind combos, add note about format it README (<ctrl>+<shift>+a, etc.).
Rename "release word" to "submit word" everywhere.

### Rework chat focusing

Firstly, focus window, then run keybinds to focus chat input:
Invoke Ctrl+Shift+Space (Voice input keybing) once, wait a little and run it second time. It will enable and disable voice input mode so we focus current open chat. Note that keybind should be cliked, not dragged.

After it you able to type input until release word is said, when release word is said, press enter. Note that wake and release words also should be typed.

### Saving to buffer
Last message that you hear (between wake and submit words inclusive) should be saved to copy-paste buffer. Make this feature optional but true by default. (passed to the cli).

### User interruption
If user does something during the invocation (mouse cliks and keyboard input) - after wake word but before release word, stop typing and so on but continue listening to get full message. If user pressed Enter - that means we have released, so dont wait for release word. Make this behavior optional but enabled by default.

### Using Cursor Voice Input

As an alternative to our "typing" we may support native Cursor IDE Voice input - click once Ctrl+Shift+Space in Cursor and it will listen us for instructions. But I personally dont like as it works very slow, but it refine user prompt (rewriting in target language, fixing punctuation and so on). Also it does support Submit Keywoard ("submit" by default), but it does not support Russian submit keywoard (while input russian language works nice). So we will have option for using Cursor Voice Input that disabled by default. Add note about default "submit" submit keyword.
