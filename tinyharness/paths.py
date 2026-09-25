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
    """~/.config/tinyharness/config.toml, or under $XDG_CONFIG_HOME when that is set."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / "tinyharness" / "config.toml"


def project_dir(working_dir: Path) -> Path:
    """Per-project state: sessions, scratch space, project skills/hooks/agents."""
    return Path(working_dir) / ".tinyharness"


def search_dirs(working_dir: Path, name: str) -> list[Path]:
    """User-global then project-local directory for an extension type.

    Order matters: later entries take precedence when names collide.
    """
    return [home_dir() / name, project_dir(working_dir) / name]


def pretty_path(path: Path | str, limit: int | None = None) -> str:
    """For display: ~/code/app, or …/last/parts when longer than `limit` characters."""
    path = Path(path)
    try:
        text = f"~/{path.relative_to(Path.home())}"
    except ValueError:
        text = str(path)
    if limit is None or len(text) <= limit:
        return text
    parts = path.parts
    for i in range(1, len(parts)):
        short = "…/" + "/".join(parts[i:])
        if len(short) <= limit:
            return short
    return "…" + text[-(limit - 1) :]
