"""Saving a conversation losslessly, and loading it back to resume it.

The session log is the same append-only JSONL file as the transcript. Next to the
audit events (tool_use, permission, ...) it holds records that rebuild the history
exactly:

    msg    a message, in full: content, tool calls, reasoning state, meta
    edit   compaction rewrote a message in place (new content; reasoning state dropped?)
    view   the message order changed other than by appending (a summary, /clear):
           the full list of message ids, in order
    todos  the todo list changed
    model  the session switched model ("provider:model")
    world  git HEAD and the AGENT.md hash, at the end of each turn
    files  file changes a resume found and reported, so the next resume doesn't again

Replaying them in order gives back the history as it was at the last record. The log
lives in ~/.tinyharness/projects/<project>/, outside the project (so it's never committed)
and outside what sandboxed commands can write, so what resume trusts can't be doctored
by the agent's own commands.

`SessionStore` is the storage interface: `FileSessionStore` (the default) keeps one
JSONL file per session there; a server could keep them in a database.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from tinyharness.agent.history import History, Message
from tinyharness.agent.transcript import Transcript
from tinyharness.paths import state_dir


class LogWriter(Protocol):
    def write(self, event: str, **payload: Any) -> None: ...
    def close(self) -> None: ...


@dataclass
class SessionInfo:
    id: str
    updated: datetime
    title: str  # the first prompt
    messages: int


class SessionStore(Protocol):
    """Where session logs live."""

    def writer(self, session_id: str) -> LogWriter:
        """Append to a session's log (creating it if needed)."""
        ...

    def load(self, session_id: str) -> list[dict[str, Any]]:
        """Every record of a session, in order. KeyError if there's no such session."""
        ...

    def sessions(self, limit: int = 20) -> list[SessionInfo]:
        """Sessions that can be resumed, most recently active first."""
        ...


class FileSessionStore:
    """One JSONL file per session in ~/.tinyharness/projects/<project>/sessions/."""

    def __init__(self, working_dir: Path) -> None:
        self.working_dir = Path(working_dir)
        self.directory = state_dir(self.working_dir) / "sessions"

    def writer(self, session_id: str) -> Transcript:
        return Transcript(self.working_dir, session_id)

    def load(self, session_id: str) -> list[dict[str, Any]]:
        path = self.directory / f"{session_id}.jsonl"
        if "/" in session_id or not path.is_file():
            raise KeyError(session_id)
        return list(_read_jsonl(path))

    def sessions(self, limit: int = 20) -> list[SessionInfo]:
        if not self.directory.is_dir():
            return []
        files = sorted(self.directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        found: list[SessionInfo] = []
        for path in files:
            info = _describe(path)
            if info is not None:
                found.append(info)
            if len(found) >= limit:
                break
        return found


@dataclass
class Replayed:
    """A session log, replayed."""

    history: History
    every_message: list[Message]  # everything ever logged, including messages compacted away
    todos: list[dict[str, str]] = field(default_factory=list)
    model: str | None = None
    world: dict[str, Any] = field(default_factory=dict)  # the last `world` record
    reported: dict[str, dict[str, Any] | None] = field(default_factory=dict)  # from `files` records


def replay(records: list[dict[str, Any]]) -> Replayed:
    messages: dict[str, Message] = {}
    order: list[str] = []
    every: list[Message] = []
    out = Replayed(History(), every)
    for r in records:
        match r.get("event"):
            case "msg":
                m = Message.from_record(r["message"])
                messages[m.id] = m
                order.append(m.id)
                every.append(m)
            case "edit" if r["id"] in messages:
                messages[r["id"]].content = r.get("content")
                if r.get("drop_state"):
                    messages[r["id"]].state = None
            case "view":
                order = [i for i in r["ids"] if i in messages]
            case "todos":
                out.todos = list(r.get("todos", []))
            case "model":
                out.model = r.get("spec")
            case "files":
                out.reported.update(r.get("files", {}))
            case "world":
                out.world = {k: v for k, v in r.items() if k not in ("ts", "event")}
    out.history.messages = [messages[i] for i in order]
    return out


class SessionLog:
    """Writes the records that `replay` reads. Call `sync()` whenever the history may have
    changed: it logs only the difference since the last call."""

    def __init__(self, writer: LogWriter) -> None:
        self.writer = writer
        self._seen: dict[str, tuple[int, bool]] = {}  # id -> (hash of content, has state)
        self._order: list[str] = []
        self._todos: list[dict[str, str]] | None = None

    def seed(self, history: History, todos: list[dict[str, str]]) -> None:
        """Mark a replayed history as already logged."""
        self._seen = {m.id: _fingerprint(m) for m in history.messages}
        self._order = [m.id for m in history.messages]
        self._todos = list(todos)

    def sync(self, history: History, todos: list[dict[str, str]]) -> None:
        added: list[str] = []
        for m in history.messages:
            fp, seen = _fingerprint(m), self._seen.get(m.id)
            if seen is None:
                self.writer.write("msg", message=m.to_record())
                added.append(m.id)
            elif seen != fp:
                self.writer.write("edit", id=m.id, content=m.content, drop_state=seen[1] and not fp[1])
            self._seen[m.id] = fp
        order = [m.id for m in history.messages]
        if order != self._order + added:
            self.writer.write("view", ids=order)
        self._order = order
        if todos != self._todos:
            self.writer.write("todos", todos=list(todos))
            self._todos = list(todos)


def _fingerprint(m: Message) -> tuple[int, bool]:
    return hash(m.content), m.state is not None


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by a crash


def _describe(path: Path) -> SessionInfo | None:
    """None for logs with nothing to resume: older logs, no prompt yet, or emptied by a
    /clear (logs from before /clear started a new conversation)."""
    title, count = "", 0
    records = list(_read_jsonl(path))
    if not replay(records).history.messages:
        return None
    for r in records:
        if r.get("event") == "msg":
            count += 1
            message = r.get("message", {})
            if not title and message.get("role") == "user":
                title = " ".join(str(message.get("content") or "").split())[:80]
    if not title:
        return None
    return SessionInfo(path.stem, datetime.fromtimestamp(path.stat().st_mtime), title, count)
