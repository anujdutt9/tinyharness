"""The plan the agent writes in plan mode, as a Markdown file you can read and edit.

The plan is plain Markdown the agent writes, not a schema it has to fill in: any model
that can write a file can write one (seen live: Qwen 4B rarely got a
structured plan call right). With `ready`, the agent hands it to you for review and its
turn ends. Leaving plan mode approves the exact bytes on disk (their hash), and from then
on the build is held to that revision (see docs/plans-design.md).

It lives with what tinyharness keeps for the project, not in the project:

    ~/.tinyharness/projects/<project>/plans/<id>.md

The agent follows the plan when it builds, so it's instructions. Outside the project, a
cloned repository can't ship one and a sandboxed command can't rewrite it; only the Plan
tool and you change it.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from tinyharness.agent.prompts import repo_root
from tinyharness.paths import pretty_path, state_dir
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


class PlanInput(ToolInput):
    plan: str = Field(description="The whole plan in Markdown, starting with a '# Title' line")
    ready: bool = Field(
        default=True,
        description="True (the default) when the plan is complete: the user reviews it now and "
        "your turn ends. False to save a draft and keep working on it.",
    )


def plans_dir(working_dir: Path) -> Path:
    # Keyed on the repository's root, like memory: starting in a subfolder finds the same plans.
    return state_dir(repo_root(working_dir)) / "plans"


def revision(text: str) -> str:
    """The plan's revision: a hash of the whole file, byte for byte."""
    return hashlib.sha256(text.encode()).hexdigest()


def title_of(text: str) -> str:
    """The plan's name: its first heading, else its first line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading = next((line.lstrip("#").strip() for line in lines if line.startswith("#")), "")
    return (heading or (lines[0] if lines else "") or "Plan")[:80]


def _write(path: Path, text: str) -> None:
    """Replace the file in one step, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".plan-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-") or "plan"


@dataclass
class PlanFile:
    """This session's plan: where it is and what the Plan tool last wrote there."""

    id: str
    path: Path
    written: str  # revision of the text the tool wrote; another one on disk means a hand edit
    observed_edit: str = ""  # edit already shown to the model; never replaces `written`
    title: str = ""
    approved: str = ""  # exact on-disk revision approved by leaving plan mode

    def text(self) -> str | None:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError:
            return None

    def edited(self) -> bool:
        text = self.text()
        return text is None or revision(text) != self.written


class PlanTool(Tool[PlanInput]):
    name = "Plan"
    # A good plan records the context, decisions, concrete steps and explicit boundaries
    # that the builder would otherwise have to rediscover or guess.
    description = (
        "Save your plan for the user to review: the whole plan in Markdown, starting with a "
        "'# Title' line. Call it in plan mode once the questions that would change the plan "
        "are answered; the plan goes here, not in your reply. Keep it short and specific:\n"
        "- the goal and context, in a line or two\n"
        "- decisions made with the user; record a default you chose without asking as a "
        "decision starting 'assumed:'\n"
        "- the steps in order, riskiest first, each a concrete change with the files it touches "
        "and how to check it works (a command where there is one); split a step that touches "
        "more than about five files, and never write a step that only says to investigate\n"
        "- what's out of scope\n"
        "Prefer reusing what exists to new files or abstractions, and add detail only where the "
        "builder would otherwise have to guess. Each call replaces the whole plan: when revising, "
        "keep what the feedback doesn't touch. With ready (the default) the user reviews it now "
        "and your turn ends; use ready=false only to save a draft you're still working on."
    )
    Input = PlanInput
    # It writes tinyharness's private planning state. That is deliberately available in
    # plan mode, but it is a real write and must not run in parallel with another Plan call.
    available_in_plan_mode = True
    needs_permission = False
    taken_from_json = True  # the user reviews it first; seen live, Gemma 3 wrote plans as JSON

    def __init__(self, working_dir: Path, planning: Callable[[], bool] = lambda: True) -> None:
        self.working_dir = working_dir
        self.planning = planning
        self.current: PlanFile | None = None

    def offered(self) -> bool:
        # Only in plan mode: every tool's definition is sent with every request.
        return self.planning()

    def describe(self, args: PlanInput) -> str:
        return f"Plan({title_of(args.plan)})"

    async def run(self, args: PlanInput, ctx: ToolContext) -> str:
        text = args.plan.strip()
        if not text:
            raise ToolError("Give the whole plan in `plan`, as Markdown starting with a '# Title' line.")
        text += "\n"
        if self.current is not None and self.current.edited():
            # Saving now would throw away what the user changed by hand. The error shows the
            # edited file once, so the next call is the revision that includes it.
            edited = self.current.text()
            edit_revision = revision(edited) if edited is not None else "<missing>"
            if edit_revision != self.current.observed_edit:
                self.current.observed_edit = edit_revision
                state = "deleted" if edited is None else "edited"
                body = "" if edited is None else " The file now:\n\n" + edited
                raise ToolError(
                    f"The user {state} the plan since you last wrote it. Work that change into "
                    f"your revision and call Plan again with the whole plan.{body}"
                )
        title = title_of(text)
        if self.current is None:
            self.current = self._new_file(title)
        _write(self.current.path, text)
        self.current.written = revision(text)
        self.current.observed_edit = ""
        self.current.title = title
        self.current.approved = ""  # a new draft supersedes the approved revision
        ctx.plan_summary = ""
        where = f"Saved the plan to {pretty_path(self.current.path)} (revision {self.current.written[:8]})."
        if not args.ready:
            return f"{where} It's a draft: call Plan again with ready=true when it's complete."
        ctx.hand_back = "plan_ready"
        return f"{where} The user is reviewing it now, so your turn ends here; don't repeat it."

    def approve(self) -> str:
        """Approve the exact file on disk. Called only for a user mode transition.

        Leaving plan mode is the user's choice and never fails: a plan file they emptied or
        deleted means they're building without a plan, so the plan ends (returns "")."""
        if self.current is None:
            return ""
        text = self.current.text()
        if text is None or not text.strip():
            self.current = None
            return ""
        approved = revision(text)
        self.current.written = approved  # a hand edit is now the accepted baseline
        self.current.observed_edit = ""
        self.current.title = title_of(text)
        self.current.approved = approved
        return approved

    def build_error(self) -> str:
        """Why a mutating build call cannot run, or empty when its plan is current."""
        if self.current is None:
            return ""
        if not self.current.approved:
            return (
                f"draft plan {self.current.id!r} is not approved; ask the user to review it "
                "and switch out of plan mode"
            )
        text = self.current.text()
        if text is None:
            return "the approved plan file is missing; return to plan mode and write it again"
        if revision(text) != self.current.approved:
            return (
                f"plan {self.current.id!r} changed since approval; return to plan mode, explain "
                "the change, and ask the user to approve this revision"
            )
        return ""

    def snapshot(self) -> dict[str, Any]:
        """Small trusted record sufficient to restore this session's plan state."""
        if self.current is None:
            return {}
        return {
            "id": self.current.id,
            "path": str(self.current.path),
            "written": self.current.written,
            "title": self.current.title,
            "approved": self.current.approved,
        }

    def restore(self, saved: dict[str, Any]) -> None:
        """Restore a plan record from this session's private append-only log."""
        try:
            path = Path(str(saved["path"])).expanduser().resolve()
            folder = plans_dir(self.working_dir).resolve()
            if not path.is_relative_to(folder):
                return
            self.current = PlanFile(
                id=str(saved["id"]),
                path=path,
                written=str(saved.get("written", "")),
                title=str(saved.get("title", "")),
                approved=str(saved.get("approved", "")),
            )
        except (KeyError, TypeError, ValueError, OSError):
            return

    def _new_file(self, title: str) -> PlanFile:
        folder, base = plans_dir(self.working_dir), _slug(title)
        folder.mkdir(parents=True, exist_ok=True)
        plan_id, n = base, 1
        while True:
            path = folder / f"{plan_id}.md"
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:  # another process/session claimed it: leave it alone
                n += 1
                plan_id = f"{base}-{n}"
                continue
            os.close(fd)
            return PlanFile(plan_id, path, written="")
