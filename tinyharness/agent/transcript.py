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


class SessionBusyError(RuntimeError):
    """Another tinyharness process currently owns this session log."""


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

    def __init__(
        self,
        working_dir: Path,
        session_id: str,
        kind: str = "sessions",
        secrets: dict[str, str] | None = None,
    ) -> None:
        directory = state_dir(working_dir) / kind
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _chmod(directory, 0o700)  # made before logs were private: close it now
        self.path = directory / f"{session_id}.jsonl"
        self._lock_path = directory / f".{session_id}.lock"
        self.session_id = session_id
        self.secrets = dict(secrets or {})
        lock_fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        self._lock = os.fdopen(lock_fd, "r+b", buffering=0)
        _chmod(self._lock_path, 0o600)
        try:
            _lock(self._lock)
        except OSError as e:
            self._lock.close()
            raise SessionBusyError(
                "That conversation is open in another tinyharness window. "
                "Switch to that window or close it before resuming here."
            ) from e
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            self._file = os.fdopen(fd, "a", encoding="utf-8")
        except Exception:
            _unlock(self._lock)
            self._lock.close()
            raise
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
        self._file.write(json.dumps(_redact(record, extra=self.secrets), default=str) + "\n")
        self._file.flush()

    def set_secrets(self, secrets: dict[str, str]) -> None:
        """Replace the exact credential values masked from future records."""
        self.secrets = dict(secrets)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if not self._lock.closed:
            _unlock(self._lock)
            self._lock.close()

    def __enter__(self) -> Transcript:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()


def _redact(value: Any, *, patterns: bool = True, extra: dict[str, str] | None = None) -> Any:
    """Mask credentials in every string of a log record. Provider reasoning (`state`) and
    images get only exact-value masking: signed and encrypted blobs, and base64 image
    data, are long random text that a key-shaped pattern could match by chance, and
    altering one would break the conversation (or the picture) when it's resumed."""
    if isinstance(value, str):
        return redact_secrets(value, extra) if patterns else mask_secrets(value, extra)
    if isinstance(value, dict):
        # Provider reasoning and image data: exact values only (see the docstring).
        return {
            k: _redact(v, patterns=patterns and k not in ("state", "images"), extra=extra)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact(v, patterns=patterns, extra=extra) for v in value]
    return value


def _chmod(path: Path, mode: int) -> None:
    with contextlib.suppress(OSError):  # not ours to change (a shared directory): leave it
        path.chmod(mode)


def _lock(file: Any) -> None:
    """Take an advisory, non-blocking lock. The OS releases it if the process dies."""
    if os.name == "nt":
        import msvcrt

        if os.fstat(file.fileno()).st_size == 0:
            file.write(b"\0")
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(file: Any) -> None:
    with contextlib.suppress(OSError):
        if os.name == "nt":
            import msvcrt

            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def lock_is_held(path: Path) -> bool:
    """Whether another live Transcript holds `path`; used only for picker status."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    file = os.fdopen(fd, "r+b", buffering=0)
    try:
        _lock(file)
    except OSError:
        file.close()
        return True
    _unlock(file)
    file.close()
    return False
