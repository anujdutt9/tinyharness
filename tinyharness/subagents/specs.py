"""Subagent definitions.

Built-ins ship as Markdown in tinyharness/prompts/agents/. Users add their own in
~/.tinyharness/agents/ or <project>/.tinyharness/agents/, in the same format:

    ---
    name: reviewer
    description: Reviews a diff for bugs.
    tools: [Read, Grep, Glob]          # leave out for every tool the session has
    permission_mode: default
    model: small                       # inherit | small | provider:model | model id
    max_steps: 60                      # model calls before it must wrap up (default 40)
    ---
    You are a meticulous code reviewer...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinyharness.frontmatter import read_frontmatter_file
from tinyharness.paths import search_dirs
from tinyharness.permissions.modes import PermissionMode
from tinyharness.prompts import AGENTS_DIR


@dataclass
class AgentSpec:
    """Define a subagent available to a session.

    Parameters
    ----------
    name : str
        Unique agent name.
    description : str
        Guidance the lead model uses when deciding whether to delegate.
    system_prompt : str, optional
        Instructions for the subagent.
    tools : list[str], optional
        Allowed tools. ``None`` uses every eligible session tool.
    permission_mode : PermissionMode, optional
        Upper bound on permissions for this agent.
    model : str, optional
        Model selection accepted by :meth:`Session.llm_for`.
    max_steps : int, optional
        Maximum model calls before the subagent must finish.
    source : pathlib.Path, optional
        Markdown definition that produced this specification.
    """

    name: str
    description: str
    system_prompt: str = ""
    tools: list[str] | None = None
    # A limit, not a mode: the agent runs in the session's mode, held to this if it's
    # stricter (None: no limit). An agent can't be more permissive than the session.
    permission_mode: PermissionMode | None = None
    model: str | None = None
    max_steps: int | None = None  # model calls before it must wrap up (None: SUBAGENT_MAX_STEPS)
    source: Path | None = None  # the Markdown file it came from, if any

    def __post_init__(self) -> None:
        if self.permission_mode is not None:
            self.permission_mode = PermissionMode.parse(self.permission_mode)  # accept "default"


def parse_agent_md(path: Path) -> AgentSpec | None:
    parsed = read_frontmatter_file(path)
    if parsed is None:
        return None
    meta, body = parsed
    return AgentSpec(
        name=str(meta.get("name") or path.stem),
        description=str(meta.get("description") or ""),
        system_prompt=body,
        tools=list(meta["tools"]) if meta.get("tools") is not None else None,
        permission_mode=PermissionMode.parse(meta["permission_mode"])
        if meta.get("permission_mode")
        else None,
        model=str(meta["model"]) if meta.get("model") else None,
        max_steps=int(meta["max_steps"]) if meta.get("max_steps") else None,
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
