"""Subagent definitions.

Built-ins ship as Markdown in tinyharness/prompts/agents/. Users add their own in
~/.tinyharness/agents/ or <project>/.tinyharness/agents/, in the same format:

    ---
    name: reviewer
    description: Reviews a diff for bugs.
    tools: [Read, Grep, Glob]
    permission_mode: acceptEdits
    model: anthropic/claude-haiku-4-5
    ---
    You are a meticulous code reviewer...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tinyharness.frontmatter import read_frontmatter_file
from tinyharness.paths import search_dirs
from tinyharness.permissions.modes import PermissionMode
from tinyharness.prompts import AGENTS_DIR


@dataclass
class AgentSpec:
    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] = field(default_factory=lambda: ["Read", "Write", "Edit"])
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    model: str | None = None
    source: Path | None = None


def parse_agent_md(path: Path) -> AgentSpec | None:
    parsed = read_frontmatter_file(path)
    if parsed is None:
        return None
    meta, body = parsed
    return AgentSpec(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        system_prompt=body,
        tools=list(meta.get("tools") or ["Read", "Write", "Edit"]),
        permission_mode=PermissionMode.parse(meta.get("permission_mode") or "default"),
        model=meta.get("model"),
        source=path,
    )


def scan_agents(directory: Path) -> list[AgentSpec]:
    if not directory.is_dir():
        return []
    return [spec for p in sorted(directory.glob("*.md")) if (spec := parse_agent_md(p))]


def builtin_agents() -> dict[str, AgentSpec]:
    """The agents shipped in tinyharness/prompts/agents/."""
    return {spec.name: spec for spec in scan_agents(AGENTS_DIR)}


def load_agents(working_dir: Path) -> dict[str, AgentSpec]:
    """Built-in agents, overridden by user-global then project-local definitions."""
    agents = builtin_agents()
    for directory in search_dirs(working_dir, "agents"):
        agents.update({spec.name: spec for spec in scan_agents(directory)})
    return agents
