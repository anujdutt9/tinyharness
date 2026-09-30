"""The input box's place on screen, checked in an emulated terminal (pyte).

Opt-in (`pytest --terminal`): these start a real process under a pseudo-terminal, so
they're slower than the rest, and they only matter when changing the screen layout.
"""

import os
import select
import signal
import sys
import time
from collections.abc import Callable

import pytest

pyte = pytest.importorskip("pyte")
pytestmark = [
    pytest.mark.terminal,
    pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal"),
]

ROWS, COLS = 24, 80
MARK = "printed while the box is up"

# The REPL's input box on its own: three lines of earlier output, then the box, then a
# line printed above it while it's up (as the agent's output is).
APP = f"""
import asyncio
from prompt_toolkit.patch_stdout import patch_stdout
from tinyharness.cli.repl import _CompactPrompt

async def main():
    for i in range(3):
        print(f"shell line {{i}}")
    prompt = _CompactPrompt(message="> ", multiline=True, bottom_toolbar=lambda: "status bar")
    with patch_stdout(raw=True):
        asyncio.create_task(prompt.prompt_async(show_frame=True))
        await asyncio.sleep(0.5)
        print({MARK!r})
        await asyncio.sleep(60)  # the test ends it

asyncio.run(main())
"""


# A picker (/model, /provider, ...) after some output: it shows where the input box was.
PICKER = """
import asyncio
from tinyharness.cli.widgets import pick

async def main():
    for i in range(3):
        print(f"shell line {i}")
    await pick("Choose a model", [("a", "Model A"), ("b", "Model B"), ("c", "Model C")], "a")

asyncio.run(main())
"""


def screen_of(
    answers_cursor: bool, done: Callable[[list[str]], bool], app: str = APP, timeout: float = 15
) -> list[str]:
    """Run `app` in a ROWS x COLS terminal until `done(screen)`; the screen's lines.
    `answers_cursor`: whether the terminal reports where its cursor is (most do)."""
    import fcntl
    import pty
    import struct
    import termios

    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    pid, fd = pty.fork()
    if pid == 0:  # the child: the app, in the pseudo-terminal
        os.execv(sys.executable, [sys.executable, "-c", app])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    if answers_cursor:
        screen.write_process_input = lambda data: os.write(fd, data.encode())
    lines: list[str] = []
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if select.select([fd], [], [], 0.05)[0]:
                try:
                    stream.feed(os.read(fd, 65536))
                except OSError:
                    break
            lines = [line.rstrip() for line in screen.display]
            if done(lines):
                return lines
        pytest.fail("the screen never got there:\n" + "\n".join(lines))
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)
    return lines


def box_is_up(lines: list[str]) -> bool:
    return MARK in lines and any(line.startswith("└") for line in lines)


def test_the_input_box_sits_at_the_bottom_and_output_goes_above_it():
    lines = screen_of(answers_cursor=True, done=lambda s: box_is_up(s) and s[-1] == "status bar")
    assert lines[:4] == ["shell line 0", "shell line 1", "shell line 2", MARK]  # nothing pushed away
    assert lines[-4].startswith("┌") and lines[-3].startswith("│>") and lines[-2].startswith("└")
    assert all(line == "" for line in lines[4:-4])  # the rows between are spare, not the box's


def test_a_terminal_that_cant_report_the_cursor_gets_the_box_under_the_output():
    lines = screen_of(answers_cursor=False, done=box_is_up)
    top = next(i for i, line in enumerate(lines) if line.startswith("┌"))
    assert lines.index(MARK) < top < ROWS - 4  # right under the output, as before


def test_a_picker_shows_where_the_input_box_was():
    lines = screen_of(answers_cursor=True, app=PICKER, done=lambda s: "Model C" in s[-1])
    assert lines[:3] == ["shell line 0", "shell line 1", "shell line 2"]
    assert "Choose a model" in lines[-4] and "Model A" in lines[-3] and "Model C" in lines[-1]
