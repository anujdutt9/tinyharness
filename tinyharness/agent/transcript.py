from __future__ import annotations

import contextlib
import json
import os
import uuid
from collections.abc import Iterator
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from tinyharness.llm import identity
from tinyharness.paths import state_dir
from tinyharness.security import mask_secrets, redact_secrets
from tinyharness.version import __version__

TRACE_SCHEMA = 1
_TRACE: ContextVar[dict[str, Any] | None] = ContextVar("tinyharness_trace", default=None)


@contextlib.contextmanager
def trace_scope(**fields: Any) -> Iterator[None]:
    """Attach correlation fields to every record written in this async context."""
    token = _TRACE.set({**(_TRACE.get() or {}), **fields})
    try:
        yield
    finally:
        _TRACE.reset(token)


def update_trace(**fields: Any) -> None:
    """Update the current turn's trace fields until its enclosing scope ends."""
    _TRACE.set({**(_TRACE.get() or {}), **fields})


def trace_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Transcript:
    """Append-only JSONL log of a session: one event per line.

    Main sessions go to ~/.tinyharness/projects/<project>/sessions/, subagent runs to .../sidechains/.
    Only you can read them (the directory is 0700, the files 0600), and credentials are
    masked before anything is written: the values of configured keys, and anything
    shaped like an API key (a key pasted into the chat, one in a URL or command).
    """

    def __init__(self, working_dir: Path, session_id: str, kind: str = "sessions") -> None:
        directory = state_dir(working_dir) / kind
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _chmod(directory, 0o700)  # made before logs were private: close it now
        self.path = directory / f"{session_id}.jsonl"
        self.session_id = session_id
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._file = os.fdopen(fd, "a", encoding="utf-8")
        _chmod(self.path, 0o600)

    def write(self, event: str, **payload: Any) -> None:
        who = identity.CURRENT.get()
        context = _TRACE.get() or {}
        record = {
            "schema": TRACE_SCHEMA,
            "ts": datetime.now(UTC).isoformat(),
            "event": event,
            "session": who.session if who is not None else self.session_id,
            "turn": None,
            "step": None,
            "agent": who.agent if who is not None else None,
            "model_call": None,
            "routing_decision": None,
            "tool_call": None,
            "harness_version": __version__,
            **context,
            **payload,
        }
        self._file.write(json.dumps(_redact(record), default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> Transcript:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def _redact(value: Any, *, patterns: bool = True) -> Any:
    """Mask credentials in every string of a log record. Provider reasoning (`state`) and
    images get only exact-value masking: signed and encrypted blobs, and base64 image
    data, are long random text that a key-shaped pattern could match by chance, and
    altering one would break the conversation (or the picture) when it's resumed."""
    if isinstance(value, str):
        return redact_secrets(value) if patterns else mask_secrets(value)
    if isinstance(value, dict):
        # Provider reasoning and image data: exact values only (see the docstring).
        return {k: _redact(v, patterns=patterns and k not in ("state", "images")) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_redact(v, patterns=patterns) for v in value]
    return value


def _chmod(path: Path, mode: int) -> None:
    with contextlib.suppress(OSError):  # not ours to change (a shared directory): leave it
        path.chmod(mode)
