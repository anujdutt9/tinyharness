"""Well-known filesystem locations.

tinyharness keeps everything user-global under one home directory (``~/.tinyharness`` by
default, overridable with ``TINYHARNESS_HOME``), including what it records per project
(``~/.tinyharness/projects/<project>-<id>/``). ``<working-dir>/.tinyharness`` is only for project
configuration you write yourself. It never reads another tool's home (e.g. ``~/.claude``).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import shutil
from pathlib import Path


def home_dir() -> Path:
    """Everything of yours, in one folder: config.toml, auth.json, skills/, agents/,
    plugins/, hooks.toml, and what tinyharness keeps (state, session logs, caches)."""
    override = os.environ.get("TINYHARNESS_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".tinyharness"


def default_config_path() -> Path:
    """~/.tinyharness/config.toml: everything of yours is in one folder (see `home_dir`)."""
    return home_dir() / "config.toml"


def auth_path() -> Path:
    """~/.tinyharness/auth.json: API keys, by provider (see config.py)."""
    return home_dir() / "auth.json"


def legacy_config_dir() -> Path:
    """Where earlier versions kept config.toml and the .env key file (~/.config/tinyharness,
    or under $XDG_CONFIG_HOME). Read only to move what's there (see config.py)."""
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base).expanduser() / "tinyharness"


def source_checkout() -> Path | None:
    """The repo root when running from a git checkout of tinyharness (development),
    None when installed from a package: site-packages has no pyproject.toml or .git."""
    root = Path(__file__).resolve().parent.parent
    return root if (root / "pyproject.toml").is_file() and (root / ".git").exists() else None


def project_dir(working_dir: Path) -> Path:
    """Project configuration you may commit and share: skills, agents, plugins, hooks, AGENTS.md.
    tinyharness never writes here itself; its own per-project files go in `state_dir`."""
    return Path(working_dir) / ".tinyharness"


def state_dir(working_dir: Path) -> Path:
    """What tinyharness keeps for a project (conversation logs, scratch space):
    ~/.tinyharness/projects/<project>-<id>/. Never inside the project, so it can't end up
    in a commit, an archive or a Docker build context."""
    root = str(Path(working_dir).expanduser().resolve())
    label = re.sub(r"[^A-Za-z0-9._-]", "-", Path(root).name).strip(".-_") or "project"
    # A deterministic id needs no registry to recover after a crash. Sixteen hex digits
    # keep collision risk negligible even for very large numbers of local checkouts.
    project_id = hashlib.sha256(root.encode()).hexdigest()[:16]
    return home_dir() / "projects" / f"{label[:64]}-{project_id}"


def _path_named_state_dir(working_dir: Path) -> Path:
    """Return the per-project directory name used before readable project identifiers."""
    root = str(Path(working_dir).expanduser().resolve())
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", root)
    if len(slug) > 120:
        slug = f"{slug[-100:]}-{hashlib.sha256(root.encode()).hexdigest()[:10]}"
    return home_dir() / "projects" / slug


def move_legacy_state(working_dir: Path) -> None:
    """Move older project state into the current private per-project directory."""
    new = state_dir(working_dir)
    path_named = _path_named_state_dir(working_dir)
    if path_named != new and path_named.is_dir() and not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            path_named.replace(new)

    # The oldest layout kept this state inside the repository.
    old = project_dir(working_dir)
    if not old.is_dir() or old.resolve() == home_dir().expanduser().resolve():
        return
    for kind in ("sessions", "sidechains"):
        src = old / kind
        if not src.is_dir():
            continue
        (new / kind).mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jsonl"):
            with contextlib.suppress(OSError):
                shutil.move(str(f), str(new / kind / f.name))
        with contextlib.suppress(OSError):
            src.rmdir()
    shutil.rmtree(old / "tmp", ignore_errors=True)
    with contextlib.suppress(OSError):
        old.rmdir()  # only if nothing else (your config) is in it


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
