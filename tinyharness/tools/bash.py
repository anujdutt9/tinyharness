"""Bash: run a shell command in the working directory, inside the OS sandbox.

Each call is a fresh process (no shell state carries over), started in its own
process group so a timeout or Ctrl-C kills everything it spawned. The environment
has credentials stripped, and output is capped (head and tail kept) before the
loop masks any remaining secrets.

With a sandbox, the command can write only to the project, scratch and temp dirs,
can't read credential files and has no network, so the gate can let it run without
asking in acceptEdits and auto modes. Without one, every command asks. The model can
ask to skip the sandbox (sandbox=false) for one command, which asks the user every time.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError

from tinyharness.process import kill_group
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.security import sanitized_env
from tinyharness.tools.base import Tool, ToolContext, ToolInput

HEAD_BYTES = 15_000  # output kept from the start...
TAIL_BYTES = 15_000  # ...and from the end

# Commands that only read, so subagents running side by side needn't take turns on them
# (Tool.writes). Deliberately short: anything else counts as writing.
_READERS = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "cut", "tr", "nl", "stat",
     "du", "df", "pwd", "which", "echo", "printf", "basename", "dirname", "realpath", "jq",
     "column", "diff", "cmp", "tree"}
)  # fmt: skip
_GIT_READS = frozenset({"status", "log", "diff", "show", "ls-files", "rev-parse", "blame"})
_FIND_WRITES = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"})
_SQLITE_WRITES = (".output", ".once", ".save", ".backup", ".import", ".shell", ".system", ".log", ".excel")


class BashInput(ToolInput):
    command: str = Field(description="The shell command to run")
    timeout: int = Field(default=120, ge=1, le=600, description="Seconds before the command is killed")
    description: str = Field(default="", description="What the command does, in 5-10 words")
    sandbox: bool = Field(
        default=True,
        description="false runs the command outside the sandbox. Only when it needs what the "
        "sandbox blocks (the network, e.g. installing packages; writes outside the project). "
        "The user must approve every such command.",
    )


class BashTool(Tool[BashInput]):
    name = "Bash"
    Input = BashInput

    def __init__(self, sandbox: Sandbox | None, policy: SandboxPolicy, shell: str | None = None) -> None:
        self.sandbox = sandbox
        self.policy = policy
        self.shell = shell or default_shell()
        limits = (
            "It runs in a sandbox: it can write only inside the working directory, scratch and temp "
            "directories, can't read credential files, and has no network access"
            + (" (except localhost)" if policy.allow_localhost else "")
            + ". So package managers can't download (npm/npx, pip, brew), and other apps can't be "
            "reached: no `open`, `osascript` or browser apps like Google Chrome.app (they fail or "
            "hang until the timeout); don't try them in the sandbox. To look at a page, use the "
            "Screenshot tool. API keys aren't available in the sandbox (key variables are stripped "
            "and key files can't be read), so a program that calls an API fails as if no key were "
            "set. That doesn't mean the user has none: to run it for real, use sandbox=false. "
            "If a command genuinely needs the network or to write elsewhere (e.g. "
            "installing a package), set sandbox=false: the user approves each such command. Don't use it to "
            "work around a denial of something the user didn't ask for."
            if sandbox
            else "It runs without a sandbox, so the user approves every command."
        )
        self.description = (
            "Run a shell command in the working directory and return its combined stdout/stderr "
            "and exit code. Use it to run tests, builds, linters, git and other CLI tools. Each call "
            "starts fresh in the working directory (use `cd dir && ...`); nothing interactive. "
            "Prefer Read/Grep/Glob/Edit over cat/grep/find/sed for files. "
            f"{limits} To show the user a file or URL, use the Open tool, not `open`."
        )

    def describe(self, args: BashInput) -> str:
        command = args.command if len(args.command) <= 80 else args.command[:79] + "…"
        return f"Bash({'unsandboxed: ' if self._unsandboxed(args) else ''}{command})"

    def writes(self, args: BashInput) -> bool:
        return not only_reads(args.command)

    def contained(self, args: dict[str, Any]) -> bool:
        return self.sandbox is not None and not self._unsandboxed(args)

    def escapes(self, args: dict[str, Any]) -> str | None:
        if not self._unsandboxed(args):
            return None
        return "this command runs outside the sandbox: it can use the network and change any of your files"

    def _unsandboxed(self, args: BashInput | dict[str, Any]) -> bool:
        """A sandbox exists and the model asked to skip it. The gate sees raw arguments,
        so read them the way the tool will ("false" as a string counts)."""
        if self.sandbox is None:
            return False
        if isinstance(args, dict):
            try:
                args = BashInput.model_validate(args)
            except ValidationError:
                return False  # the loop rejects the call before it runs
        return not args.sandbox

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        return None if self.sandbox else "runs a shell command without a sandbox"

    async def run(self, args: BashInput, ctx: ToolContext) -> str:
        argv = [self.shell, "-c", args.command]
        sandboxed = self.sandbox is not None and args.sandbox
        if sandboxed:
            assert self.sandbox is not None
            argv = self.sandbox.wrap(argv, self.policy)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=ctx.workspace.root,
            env={**sanitized_env(), **(self.policy.env if sandboxed else {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,  # its own process group, so we can kill all of it
        )
        output = _Capture()
        timed_out = False
        try:
            assert proc.stdout is not None
            await asyncio.wait_for(output.drain(proc.stdout), args.timeout)
            await proc.wait()
        except TimeoutError:
            timed_out = True
            await kill_group(proc)
        except BaseException:  # Ctrl-C: don't leave the command running
            await kill_group(proc)
            raise
        return self._report(output.text(), proc.returncode, timed_out, args, sandboxed)

    def _report(self, text: str, code: int | None, timed_out: bool, args: BashInput, sandboxed: bool) -> str:
        text = text.rstrip() or "(no output)"
        if sandboxed and ".app/" in args.command and code:
            # A macOS app needs LaunchServices to start, which the sandbox blocks (it would
            # let `open` run things outside it); Chrome dies with "Abort trap: 6".
            text += (
                "\n(note: macOS apps, including browsers, can't run in the sandbox. To look "
                "at a page, use the Screenshot tool; otherwise run this command with sandbox=false)"
            )
        elif sandboxed and "Operation not permitted" in text:
            text += (
                "\n(note: the sandbox blocks writes outside the working directory, reads of "
                "credential files and network access; this may be why)"
            )
        # The outcome goes last: the CLI reads it from the final line (a failed test run
        # showed as passing when this note came after it).
        if timed_out:
            text += f"\n(timed out after {args.timeout}s; the command was killed)"
        elif code:
            text += f"\n(exit code {code})"
        return text


def only_reads(command: str) -> bool:
    """Whether a command certainly changes nothing: known readers, joined only by pipes,
    with no redirection, substitution or chaining. Unsure means no."""
    if any(c in command for c in "><;&`$(){}\n"):
        return False
    for segment in command.split("|"):
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            return False
        name, rest = words[0], words[1:]
        if name in _READERS:
            continue
        if name == "find" and not _FIND_WRITES.intersection(rest):
            continue
        if name == "git" and rest[:1] and rest[0] in _GIT_READS:
            continue
        if name == "sqlite3" and "-readonly" in rest and not any(w in segment for w in _SQLITE_WRITES):
            continue
        return False
    return True


def default_shell() -> str:
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


class _Capture:
    """Keeps the first HEAD_BYTES and last TAIL_BYTES of a stream, whatever its size."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(65536):
            room = HEAD_BYTES - len(self.head)
            self.head += chunk[:room]
            self.tail += chunk[room:] if room > 0 else chunk
            if len(self.tail) > TAIL_BYTES:
                self.dropped += len(self.tail) - TAIL_BYTES
                del self.tail[:-TAIL_BYTES]

    def text(self) -> str:
        head = self.head.decode("utf-8", errors="replace")
        tail = self.tail.decode("utf-8", errors="replace")
        if self.dropped:
            return f"{head}\n… ({self.dropped:,} bytes of output omitted) …\n{tail}"
        return head + tail
