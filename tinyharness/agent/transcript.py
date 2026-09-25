from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any

from tinyharness.paths import project_dir


class Transcript:
    """Append-only JSONL log of a session: one event per line.

    Main sessions go to <working-dir>/.tinyharness/sessions/, subagent runs to .tinyharness/sidechains/.
    """

    def __init__(self, working_dir: Path, session_id: str, kind: str = "sessions") -> None:
        directory = project_dir(working_dir) / kind
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{session_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, event: str, **payload: Any) -> None:
        record = {"ts": datetime.now(UTC).isoformat(), "event": event, **payload}
        self._file.write(json.dumps(record, default=str) + "\n")
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
