from __future__ import annotations

import atexit
import shutil
import time
from pathlib import Path

from tinyharness.paths import project_dir


class Scratch:
    """Per-session throwaway directory at <working-dir>/.tinyharness/tmp/<session-id>/.

    Removed when the session closes (or at interpreter exit). Directories left
    behind by crashed sessions are swept once they are older than `max_age_hours`.
    """

    def __init__(self, working_dir: Path, session_id: str, max_age_hours: int = 24) -> None:
        self.base = project_dir(working_dir) / "tmp"
        self.dir = self.base / session_id
        self.base.mkdir(parents=True, exist_ok=True)
        self._sweep_stale(max_age_hours)
        self.dir.mkdir(parents=True, exist_ok=True)
        atexit.register(self.cleanup)

    def _sweep_stale(self, max_age_hours: int) -> None:
        cutoff = time.time() - max_age_hours * 3600
        for entry in self.base.iterdir():
            try:
                if entry.is_dir() and entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue

    def contains(self, path: Path | str) -> bool:
        try:
            return Path(path).resolve().is_relative_to(self.dir.resolve())
        except OSError:
            return False

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)
