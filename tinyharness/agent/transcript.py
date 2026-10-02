from __future__ import annotations

import contextlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from tinyharness.paths import state_dir
from tinyharness.security import mask_secrets, redact_secrets


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
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._file = os.fdopen(fd, "a", encoding="utf-8")
        _chmod(self.path, 0o600)

    def write(self, event: str, **payload: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **payload}
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
