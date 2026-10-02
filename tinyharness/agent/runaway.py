"""Spotting a reply that has stopped going anywhere while it still streams.

Small models sometimes get stuck: blank lines, or the same few words, over and over,
until they hit their output limit (which can take many minutes). Two signs, checked as
each piece arrives:

  repetition   the last TAIL characters are one short piece repeated (a blank line counts);
               sooner in its text and reasoning when the piece is a few characters long
               (a line of backticks), since a slow model takes minutes to repeat TAIL
  no progress  only whitespace for QUIET_SECONDS while the stream keeps going

A stream that sends nothing at all is a stall, which the client's timeout handles; a
provider's hidden reasoning (no text) counts as progress, since it can't be judged.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from tinyharness.llm.types import StreamChunk

TAIL = 4000  # characters: long enough that real output (code, tables) isn't one repeated piece
MAX_PERIOD = 400  # the longest repeated piece looked for
CHECK_EVERY = 500  # characters between repetition checks
QUIET_SECONDS = 60.0
# In text and reasoning (not tool arguments: a file can hold a long row of zeros), a piece
# of up to SHORT_PERIOD characters repeated over SHORT_TAIL characters is stuck already.
SHORT_PERIOD = 4
SHORT_TAIL = 600


class RunawayGuard:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_progress = clock()
        self._tail = ""
        self._unchecked = 0
        self._prose = ""  # the latest text and reasoning, for the quick check

    def check(self, chunk: StreamChunk) -> str | None:
        """Why this reply should be stopped, or None while it's fine."""
        visible = chunk.text + chunk.reasoning_text
        if chunk.tool_call_delta:
            function = chunk.tool_call_delta.get("function") or {}
            visible += (function.get("name") or "") + (function.get("arguments") or "")
        now = self._clock()
        if visible.strip() or (chunk.reasoning and not chunk.reasoning_text):
            self._last_progress = now
        elif now - self._last_progress > QUIET_SECONDS:
            return f"it sent nothing but blank output for over {QUIET_SECONDS:.0f} seconds"
        if prose := chunk.text + chunk.reasoning_text:
            self._prose = (self._prose + prose)[-SHORT_TAIL:]
            if len(self._prose) == SHORT_TAIL and _repeats(self._prose, SHORT_PERIOD):
                return "it kept repeating the same few characters"
        self._tail = (self._tail + visible)[-TAIL:]
        self._unchecked += len(visible)
        if self._unchecked >= CHECK_EVERY and len(self._tail) == TAIL:
            self._unchecked = 0
            if _repeats(self._tail):
                return "it kept repeating the same output"
        return None


def _repeats(text: str, longest: int = MAX_PERIOD) -> bool:
    """Whether `text` is one piece of at most `longest` characters, repeated."""
    return any(text[period:] == text[:-period] for period in range(1, longest + 1))
