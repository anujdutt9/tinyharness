"""The agent's memory: short notes that later sessions start with (the `memory` extension).

Two plain Markdown files, one note per line, which you can also edit by hand:

    ~/.tinyharness/memory.md                     about you, for every project
    ~/.tinyharness/projects/<project>/memory.md  about this project

Both live outside the project (never committed) and outside what sandboxed commands can
write, so only the Memory tool changes them, and the permission gate sees every change.
A session reads them once, at its start (see extensions/memory.py): saving a note
changes the next session's system prompt, never this one's, so the cached prefix holds.
The frozen snapshot and dedicated tool keep memory updates bounded and explicit. A generic
Write call would ask every time because ~/.tinyharness is protected.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from tinyharness.agent.prompts import repo_root
from tinyharness.paths import home_dir, state_dir
from tinyharness.security import find_api_key
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

Scope = Literal["user", "project"]

# Every note is in every request of every later session, so the character budget is small.
# A full memory makes the model merge or drop notes.
LIMITS: dict[Scope, int] = {"user": 1500, "project": 3000}
TITLES: dict[Scope, str] = {"user": "user memory", "project": "project memory"}


def memory_path(scope: Scope, working_dir: Path) -> Path:
    # Keyed on the repository's root, so starting in a subfolder finds the same notes.
    if scope == "user":
        return home_dir() / "memory.md"
    return state_dir(repo_root(working_dir)) / "memory.md"


def read_notes(path: Path) -> list[str]:
    """The notes in a memory file: each non-empty line, without its list marker."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    notes = (line.strip().removeprefix("- ").removeprefix("* ").strip() for line in text.splitlines())
    return [n for n in notes if n]


def write_notes(path: Path, notes: list[str]) -> None:
    """Replace the file in one step, so a session reading it never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".memory-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("".join(f"- {n}\n" for n in notes))
        os.chmod(tmp, 0o600)  # notes about you are yours alone, like auth.json
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def size(notes: list[str]) -> int:
    return sum(len(n) for n in notes)


def render(notes: list[str]) -> str:
    return "\n".join(f"- {n}" for n in notes) if notes else "(empty)"


class MemoryInput(ToolInput):
    action: Literal["add", "replace", "remove"]
    scope: Scope = Field(
        description=(
            "Required destination: 'user' for durable facts about the user across projects; "
            "'project' for durable context that only applies to this project"
        ),
    )
    note: str = Field(default="", description="The note to add, or the new text for 'replace'")
    old: str = Field(
        default="",
        description="For 'replace' and 'remove': a part of the existing note that only it contains",
    )


@dataclass
class _Change:
    notes: list[str]
    done: str


class MemoryTool(Tool[MemoryInput]):
    name = "Memory"
    description = (
        "Add, replace or remove a note in your memory. Memory is loaded into the system prompt "
        "of every later session (not this one), so a note is something a future session needs "
        "to know. Choose the scope explicitly:\n"
        "- 'user': durable facts about the user across projects, such as their name, role, "
        "expertise or communication preferences\n"
        "- 'project': durable decisions or external context specific to this project that "
        "cannot be learned from its files or Git history\n\n"
        "Save a note when:\n"
        "- the user asks you to remember something (or to forget it: remove the note)\n"
        "- the user corrects how you work, or confirms an approach that wasn't obvious: save "
        "the rule and the reason\n"
        "- you learn something about the project that later sessions need and can't get from "
        "the code, git history or AGENTS.md: a decision and why, a deadline, where something "
        "outside the repository lives\n\n"
        "Don't save what the code or Git already says, progress on the current task, anything "
        "only true for now, or secrets. Explicit standing instructions for every project belong "
        "in ~/.tinyharness/AGENTS.md; a rule the project team should share belongs in its "
        "AGENTS.md. Suggest the appropriate file instead.\n\n"
        "A note is one fact, in one short line that makes sense on its own (two facts are two "
        "calls, each in its own scope); write dates as dates "
        "(2026-10-01, not 'yesterday'). Before adding, look at your memory in the system "
        "prompt: if a note there covers the same thing or is out of date, replace it instead."
    )
    Input = MemoryInput
    persists = True

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir

    def describe(self, args: MemoryInput) -> str:
        return f"Memory({args.action}, {args.scope})"

    async def run(self, args: MemoryInput, ctx: ToolContext) -> str:
        path = memory_path(args.scope, self.working_dir)
        # Read the file now, not the session's snapshot: another session or the user may
        # have changed it since this one started.
        change = self._apply(args, read_notes(path), ctx.config.credential_values())
        limit = LIMITS[args.scope]
        if size(change.notes) > limit:
            raise ToolError(
                f"The {TITLES[args.scope]} is full: this would make it {size(change.notes)} of "
                f"{limit} characters. Make room first: remove notes that no longer matter, or "
                "replace several related ones with one shorter note; then try again. Its notes:\n"
                + render(read_notes(path))
            )
        write_notes(path, change.notes)
        return (
            f"{change.done} ({len(change.notes)} notes, {size(change.notes)} of {limit} "
            "characters). Sessions started from now on will see it."
        )

    def _apply(self, args: MemoryInput, notes: list[str], secrets: dict[str, str] | None = None) -> _Change:
        title = TITLES[args.scope]
        note = " ".join(args.note.split())  # one line: the file has one note per line
        if args.action != "remove":
            if not note:
                raise ToolError(f"Give the note's text in `note` to {args.action} it.")
            if (key := find_api_key(note, secrets)) is not None:
                raise ToolError(
                    f"The note contains what looks like an API key ({key}). Memory is plain text "
                    "loaded into every later session: never save credentials. Save where the key "
                    "is kept instead (e.g. 'the API key is in .env as OPENAI_API_KEY')."
                )
        if args.action == "add":
            if note in notes:
                return _Change(notes, f"The {title} already has this note")
            return _Change([*notes, note], f"Added to the {title}")
        index = self._find(args.old, notes, title)
        if args.action == "remove":
            return _Change(notes[:index] + notes[index + 1 :], f"Removed from the {title}")
        return _Change([*notes[:index], note, *notes[index + 1 :]], f"Replaced in the {title}")

    @staticmethod
    def _find(old: str, notes: list[str], title: str) -> int:
        old = " ".join(old.split())
        if not old:
            raise ToolError("Say which note in `old`: a part of its text that only it contains.")
        matches = [i for i, n in enumerate(notes) if old in n]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ToolError(
                f"No note in the {title} contains {old!r}. Its notes now (they may have changed "
                f"since this session started):\n{render(notes)}"
            )
        shown = "\n".join(f"- {notes[i]}" for i in matches)
        raise ToolError(f"{len(matches)} notes contain {old!r}; give more of the one you mean:\n{shown}")
