"""What the agent knew about the world, and what has changed since.

A resumed conversation is only as good as the facts in it. While a session runs, each
tool result that touched a file records the file's fingerprint; each turn records git
HEAD and a hash of the AGENT.md instructions. On resume, `check()` compares them with
the world as it is now, and the harness (not the model) acts on the difference:

  * results of reading a file that has changed since are replaced by a stub saying
    so, so the model can't keep relying on stale content;
  * the model is told, once, what changed: files, git HEAD or branch, instructions;
  * a tool call with no recorded result is flagged as maybe run, maybe not.

Models are poor at noticing on their own that something they remember is out of date,
so this is not left to them.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinyharness.agent.history import Message
    from tinyharness.tools.base import Tool, ToolContext

STALE_MARK = "[stale:"
UNFINISHED = (
    "(no result recorded: the session ended before this finished. It may or may not have "
    "run; check its effects before running it again)"
)


def fingerprint(path: Path) -> dict[str, Any] | None:
    """Content hash and size of a file, or None if it isn't a readable file."""
    try:
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return {"sha": digest.hexdigest()[:20], "size": path.stat().st_size}
    except OSError:
        return None


def file_meta(tool: Tool[Any], args: Any, ctx: ToolContext) -> dict[str, Any]:
    """What to remember about the file a tool call touched, for a later `check()`.
    `content` marks results that show the file (Read), as opposed to changing it (Edit)."""
    target = tool.target_path(args)
    path = ctx.workspace.resolve(target) if target else None
    if path is None or not path.is_file():
        return {}
    fp = fingerprint(path)
    return {"path": str(path), **fp, "content": tool.read_only} if fp else {}


def stub_superseded(messages: list[Message], meta: dict[str, Any], root: Path) -> int:
    """After a call touched a file, replace earlier copies of that file's content in the
    history with a stub, so the model can't copy from a version that's gone: an Edit or
    Write changed it, or a newer Read shows it changed. Unchanged copies stay (stubbing them
    would only cost the prompt cache). (Models editing from an old copy
    kept failing: the file had changed under the text they were trying to replace.)
    How many results were stubbed."""
    path = meta.get("path")
    if not path:
        return 0
    stubbed = 0
    for m in messages:
        if (
            m.role == "tool"
            and m.meta is not meta
            and m.meta.get("path") == path
            and m.meta.get("content")
            and m.meta.get("sha") != meta.get("sha")  # an identical copy can't mislead
            and not (m.content or "").startswith(STALE_MARK)
        ):
            why = "read again below" if meta.get("content") else "changed since this was read; Read it again"
            m.content = f"{STALE_MARK} {_short(path, root)}: {why}]"
            m.images = []
            stubbed += 1
    return stubbed


def git_state(root: Path) -> dict[str, str] | None:
    """HEAD commit and branch, or None outside a git repository."""
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return None
    return {"head": head, "branch": _git(root, "branch", "--show-current") or ""}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:20]


@dataclass
class ResumeReport:
    """What changed while the session was away; `note()` is what the model is told."""

    messages: int = 0
    changed: list[str] = field(default_factory=list)  # files read or edited that differ now
    missing: list[str] = field(default_factory=list)  # ...or are gone
    stale_results: int = 0  # file contents in the history replaced by a stub
    git_before: dict[str, str] | None = None
    git_now: dict[str, str] | None = None
    commits_since: int | None = None  # when the old HEAD is an ancestor of the new one
    instructions_changed: bool = False
    unfinished: int = 0  # tool calls with no recorded result
    stale_paths: set[str] = field(default_factory=set)  # full paths of changed and missing files
    now: dict[str, dict[str, Any] | None] = field(default_factory=dict)  # their fingerprints now
    current_reads: set[str] = field(
        default_factory=set
    )  # files read, and unchanged since the agent last saw them

    @property
    def git_moved(self) -> bool:
        return self.git_before is not None and self.git_now is not None and self.git_before != self.git_now

    @property
    def anything_changed(self) -> bool:
        return bool(
            self.changed or self.missing or self.git_moved or self.instructions_changed or self.unfinished
        )

    def note(self) -> str:
        """For the model, prepended to the first prompt after resuming. Empty if nothing changed."""
        lines = []
        if self.changed:
            lines.append(
                f"- Changed since you last saw them: {', '.join(self.changed)}. Read them again before relying on them."
            )
        if self.missing:
            lines.append(f"- No longer exist: {', '.join(self.missing)}.")
        if self.git_moved:
            lines.append(f"- Git: {self._git_change()}.")
        if self.instructions_changed:
            lines.append(
                "- The project instructions (AGENT.md) changed; the current version is in the system prompt."
            )
        if self.unfinished:
            lines.append(
                f"- {self.unfinished} tool call(s) from before have no recorded result; they may or may not "
                "have run. Check their effects before repeating them."
            )
        if not lines:
            return ""
        return "This conversation was resumed. Since it was last active:\n" + "\n".join(lines)

    def summary(self) -> str:
        """For people: one line."""
        parts = [f"{self.messages} messages"]
        if n := len(self.changed) + len(self.missing):
            parts.append(f"{n} file{'s' if n != 1 else ''} changed since")
        if self.git_moved:
            parts.append(self._git_change())
        if self.instructions_changed:
            parts.append("AGENT.md changed")
        if self.unfinished:
            parts.append(f"{self.unfinished} unfinished tool call{'s' if self.unfinished != 1 else ''}")
        return " · ".join(parts)

    def _git_change(self) -> str:
        before, now = self.git_before or {}, self.git_now or {}
        if before.get("branch") != now.get("branch"):
            return f"branch {before.get('branch') or '(detached)'} → {now.get('branch') or '(detached)'}"
        if self.commits_since:
            return f"{self.commits_since} new commit{'s' if self.commits_since != 1 else ''} on {now.get('branch')}"
        return f"HEAD moved {before.get('head', '')[:7]} → {now.get('head', '')[:7]}"


def check(
    messages: list[Message],
    every_message: list[Message],
    *,
    root: Path,
    reported: dict[str, dict[str, Any] | None],
    git_before: dict[str, str] | None,
    instructions_before: str | None,
    instructions_now: str,
) -> ResumeReport:
    """Compare what the session recorded with the world now, and stub stale file contents
    in `messages` (the history about to be resumed) in place.

    `every_message` is every message the session ever logged, including ones compacted
    away since: the last fingerprint of each file in it is what the agent last saw.
    `reported` holds changes already reported on an earlier resume, so they aren't
    reported again (the agent still has to read those files again before editing them)."""
    seen_by_agent: dict[str, dict[str, Any]] = {}
    read: set[str] = set()
    for m in every_message:
        if path := m.meta.get("path"):
            seen_by_agent[path] = m.meta
            if m.meta.get("content"):
                read.add(path)
    report = ResumeReport(messages=len(messages))
    for path, meta in seen_by_agent.items():
        now = fingerprint(Path(path))
        if now is not None and now["sha"] == meta.get("sha"):
            if path in read:
                report.current_reads.add(path)
            continue
        last_reported = reported.get(path, meta)
        if (now or {}).get("sha") == (last_reported or {}).get("sha"):
            continue  # changed, but the model was told on an earlier resume
        (report.changed if now else report.missing).append(_short(path, root))
        report.stale_paths.add(path)
        report.now[path] = now
    for m in messages:
        path = m.meta.get("path")
        if (
            path in report.stale_paths
            and m.meta.get("content")
            and not (m.content or "").startswith(STALE_MARK)
        ):
            m.content = f"{STALE_MARK} {_short(path, root)} has changed since this was read; Read it again]"
            m.state = None
            report.stale_results += 1

    report.git_before, report.git_now = git_before, git_state(root)
    if report.git_moved and report.git_before and report.git_now:
        old_head, new_head = report.git_before["head"], report.git_now["head"]
        if _git(root, "merge-base", "--is-ancestor", old_head, new_head) is not None:
            count = _git(root, "rev-list", "--count", f"{old_head}..{new_head}")
            report.commits_since = int(count) if count and count.isdigit() else None
    report.instructions_changed = instructions_before is not None and instructions_before != instructions_now
    return report


def _short(path: str, root: Path) -> str:
    p = Path(path)
    return str(p.relative_to(root)) if p.is_relative_to(root) else path


def _git(root: Path, *args: str) -> str | None:
    """Output of a git command, or None if it fails or exits non-zero."""
    try:
        proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()
