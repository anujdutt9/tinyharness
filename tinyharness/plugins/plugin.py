"""Plugins bundle skills, hooks and subagents into one installable directory.

my-plugin/
  plugin.toml        name = "my-plugin", description = "..."
  skills/<name>/SKILL.md
  agents/<name>.md
  hooks.toml
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinyharness.hooks import Hook, load_hooks_file
from tinyharness.paths import search_dirs
from tinyharness.skills import Skill
from tinyharness.skills.skill import scan_skills
from tinyharness.subagents.specs import AgentSpec, scan_agents

log = logging.getLogger(__name__)


@dataclass
class Plugin:
    name: str
    root: Path
    description: str = ""
    skills: list[Skill] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)
    agents: list[AgentSpec] = field(default_factory=list)


def load_plugin(root: Path) -> Plugin | None:
    manifest = root / "plugin.toml"
    if not manifest.exists():
        return None
    try:
        with manifest.open("rb") as f:
            data = tomllib.load(f)
        return Plugin(
            name=str(data.get("name") or root.name),
            root=root,
            description=str(data.get("description") or ""),
            skills=scan_skills(plugin_path(root, data, "skills_dir", "skills")),
            hooks=load_hooks_file(plugin_path(root, data, "hooks_file", "hooks.toml")),
            agents=scan_agents(plugin_path(root, data, "agents_dir", "agents")),
        )
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError, ValueError, TypeError) as e:
        # One broken plugin shouldn't stop the session from starting.
        log.warning("skipping plugin %s: %s", root, e)
        return None


def plugin_path(root: Path, manifest: dict[str, Any], key: str, default: str) -> Path:
    """A path named in plugin.toml, kept inside the plugin. Folder trust fingerprints the
    plugin's directory (tinyharness/trust.py), so nothing it loads may live outside it."""
    value = manifest.get(key, default)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, not {value!r}")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{key} = {value!r} points outside the plugin")
    return path


def load_plugins(working_dir: Path, *, project: bool = True) -> list[Plugin]:
    """Plugins from ~/.tinyharness/plugins/* and, if `project` (a trusted folder; see
    tinyharness/trust.py), <working-dir>/.tinyharness/plugins/*."""
    plugins: list[Plugin] = []
    dirs = search_dirs(working_dir, "plugins")
    for directory in dirs if project else dirs[:1]:
        if directory.is_dir():
            plugins.extend(
                p for sub in sorted(directory.iterdir()) if sub.is_dir() and (p := load_plugin(sub))
            )
    return plugins
