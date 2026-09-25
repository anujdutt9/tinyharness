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
    except (OSError, tomllib.TOMLDecodeError) as e:
        log.warning("skipping plugin %s: %s", root, e)
        return None
    return Plugin(
        name=data.get("name") or root.name,
        root=root,
        description=data.get("description") or "",
        skills=scan_skills(root / data.get("skills_dir", "skills")),
        hooks=load_hooks_file(root / data.get("hooks_file", "hooks.toml")),
        agents=scan_agents(root / data.get("agents_dir", "agents")),
    )


def load_plugins(working_dir: Path) -> list[Plugin]:
    """Plugins from ~/.tinyharness/plugins/* and <working-dir>/.tinyharness/plugins/*."""
    plugins: list[Plugin] = []
    for directory in search_dirs(working_dir, "plugins"):
        if directory.is_dir():
            plugins.extend(
                p for sub in sorted(directory.iterdir()) if sub.is_dir() and (p := load_plugin(sub))
            )
    return plugins
