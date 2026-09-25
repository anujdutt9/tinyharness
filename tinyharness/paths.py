"""Well-known filesystem locations.

tinyharness keeps everything user-global under one home directory (``~/.tinyharness`` by
default, overridable with ``TINYHARNESS_HOME``) and everything project-local under
``<working-dir>/.tinyharness``. It never reads another tool's home (e.g. ``~/.claude``).
"""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    """User-global root for skills, agents, plugins and hooks."""
    override = os.environ.get("TINYHARNESS_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tinyharness"


def default_config_path() -> Path:
    return Path.home() / ".config" / "tinyharness" / "config.toml"


def project_dir(working_dir: Path) -> Path:
    """Per-project state: sessions, scratch space, project skills/hooks/agents."""
    return Path(working_dir) / ".tinyharness"


def search_dirs(working_dir: Path, name: str) -> list[Path]:
    """User-global then project-local directory for an extension type.

    Order matters: later entries take precedence when names collide.
    """
    return [home_dir() / name, project_dir(working_dir) / name]
