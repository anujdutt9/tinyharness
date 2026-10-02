"""The input box's place on screen, checked in an emulated terminal (pyte): it follows
the output, directly under it, and never jumps or flashes (see `_CompactPrompt`).

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


# The REPL's rhythm: a line is sent, the box is erased, and a new prompt starts at once
# (so you can keep typing while the agent works).
REPROMPT = """
import asyncio
from prompt_toolkit.patch_stdout import patch_stdout
from tinyharness.cli.repl import _CompactPrompt

async def main():
    print("shell line 0")
    prompt = _CompactPrompt(message="> ", bottom_toolbar=lambda: "status bar", erase_when_done=True)
    with patch_stdout(raw=True):
        for _ in range(3):
            line = await prompt.prompt_async(show_frame=True)
            print(f"you said {line}")

asyncio.run(main())
"""


def frames_of(app: str, steps: list[tuple[Callable[[list[str]], bool], bytes]], timeout: float = 15):
    """Run `app` in a ROWS x COLS terminal that answers cursor-position requests; each
    time the screen satisfies the next step's check, type its bytes. Every screen seen
    along the way, in order: a flash that lasts one frame is in there."""
    import fcntl
    import pty
    import struct
    import termios

    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)
    pid, fd = pty.fork()
    if pid == 0:
        os.execv(sys.executable, [sys.executable, "-c", app])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    screen.write_process_input = lambda data: os.write(fd, data.encode())
    frames: list[list[str]] = []
    pending = list(steps)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and pending:
            if select.select([fd], [], [], 0.05)[0]:
                try:
                    stream.feed(os.read(fd, 65536))
                except OSError:
                    break
                frames.append([line.rstrip() for line in screen.display])
            if frames and pending[0][0](frames[-1]):
                os.write(fd, pending.pop(0)[1])
        if pending:
            pytest.fail("the screen never got there:\n" + "\n".join(frames[-1] if frames else []))
    finally:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(fd)
    return frames


def box_is_up(lines: list[str]) -> bool:
    return MARK in lines and any(line.startswith("└") for line in lines)


def box_top(lines: list[str]) -> int | None:
    return next((i for i, line in enumerate(lines) if line.startswith("┌")), None)


@pytest.mark.parametrize("answers_cursor", [True, False], ids=["cursor reports", "no cursor reports"])
def test_the_input_box_sits_right_under_the_output(answers_cursor):
    lines = screen_of(answers_cursor=answers_cursor, done=box_is_up)
    assert lines[:4] == ["shell line 0", "shell line 1", "shell line 2", MARK]
    assert box_top(lines) == 4 and lines[5].startswith("│>") and lines[6].startswith("└")


def test_a_picker_opens_right_under_the_output():
    lines = screen_of(answers_cursor=True, app=PICKER, done=lambda s: any("Model C" in line for line in s))
    assert lines[:3] == ["shell line 0", "shell line 1", "shell line 2"]
    assert "Choose a model" in lines[3] and "Model A" in lines[4]


def test_the_box_never_jumps_or_flashes_between_requests():
    """In every frame, the box is directly under the output: never lower, with a gap
    (as when it was pinned to the bottom and flashed while waiting to be placed)."""

    def under_output(lines: list[str]) -> bool:
        top = box_top(lines)
        return top is not None and lines[top + 3] == "status bar"

    frames = frames_of(
        REPROMPT,
        [
            (under_output, b"one\r"),
            (lambda s: "you said one" in s and under_output(s), b"two\r"),
            (lambda s: "you said two" in s and under_output(s), b""),
        ],
    )
    for lines in frames:
        top = box_top(lines)
        if top is None:
            continue
        screen = "\n".join(lines)
        assert top > 0 and lines[top - 1] != "", f"a gap above the box at row {top}:\n{screen}"


# The real REPL, idle, after a tool call whose output is longer than the screen.
EXPAND = """
import asyncio, os, tempfile
from pathlib import Path
from rich.console import Console
os.environ["OPENAI_API_KEY"] = "sk-test"
from tinyharness import Session, SessionOptions
from tinyharness.cli.render import Renderer
from tinyharness.cli.repl import Repl
from tinyharness.cli.theme import THEME

async def main():
    work = Path(tempfile.mkdtemp())
    session = Session(SessionOptions(config_path=work / "none.toml", working_dir=work, provider="openai"))
    console = Console(theme=THEME)
    renderer = Renderer(console)
    renderer.last_output = ("Bash(seq 40)", "\\n".join(f"row {i}" for i in range(1, 41)))
    print("earlier output")
    await Repl(session, console, renderer).loop()

asyncio.run(main())
"""


def test_ctrl_o_expands_the_last_output_and_collapses_it_again():
    def box_up(lines: list[str]) -> bool:
        return any(line.startswith("└") for line in lines) and "earlier output" in lines

    def expanded(lines: list[str]) -> bool:
        return any("ctrl+o to close" in line for line in lines)

    frames = frames_of(
        EXPAND,
        [
            (box_up, b"\x0f"),  # ctrl+o
            (expanded, b"\x0f"),
            (lambda s: box_up(s) and not expanded(s), b""),
        ],
        timeout=30,
    )
    opened = next(f for f in frames if expanded(f))
    screen = "\n".join(opened)
    assert "Bash(seq 40)" in screen and "row 1" in screen
    assert "more lines (/output prints it all)" in screen  # fitted to the 24-row screen
    assert len(opened) == ROWS
    closed = frames[-1]
    assert not any("row 1" in line for line in closed)  # nothing left behind in the scrollback
