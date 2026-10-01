"""The system prompt: prompts/system.md + user-provided AGENT.md context."""

from __future__ import annotations

import datetime
from pathlib import Path

from tinyharness.assets import ASSETS_DIR
from tinyharness.paths import home_dir, project_dir
from tinyharness.prompts import load_prompt


def agent_md_sources(working_dir: Path) -> list[Path]:
    """AGENT.md files, lowest precedence first: managed, user, project, local."""
    return [
        Path("/etc/tinyharness/AGENT.md"),
        home_dir() / "AGENT.md",
        working_dir / "AGENTS.md",  # the name most agents use (Codex, Cursor, Amp, Gemini CLI)
        working_dir / "AGENT.md",
        working_dir / "CLAUDE.md",  # so repos set up for Claude Code work out of the box
        project_dir(working_dir) / "AGENT.md",
        working_dir / "AGENT.local.md",
    ]


def load_agent_md(working_dir: Path) -> str:
    chunks: list[str] = []
    seen: set[Path] = set()
    for path in agent_md_sources(working_dir):
        try:
            resolved = path.resolve()
            if resolved in seen or not path.is_file():
                continue
            seen.add(resolved)
            chunks.append(
                f"# Context: {path}\n\n{path.read_text(encoding='utf-8', errors='replace').strip()}"
            )
        except OSError:
            continue
    return "\n\n---\n\n".join(chunks)


NO_INSTRUCTIONS = (
    "This project has no instructions file (AGENTS.md, AGENT.md or CLAUDE.md); don't look for one."
)

# Only when the TodoWrite tool is offered (the `todo_list` setting).
TODO_RULE = """# Planning
For work with several distinct steps, keep a short plan with TodoWrite; the user sees it.
Skip it for simple work. The tool's description says how to keep it useful."""


def build_system_prompt(
    working_dir: Path,
    scratch_dir: Path,
    *,
    model_name: str,
    model_id: str,
    provider: str,
    extra: str = "",
    todo_list: bool = False,
) -> str:
    parts = [
        load_prompt(
            "system",
            working_dir=working_dir,
            scratch_dir=scratch_dir,
            model_name=model_name,
            model_id=model_id,
            provider=provider,
            date=datetime.date.today().isoformat(),
            assets_dir=ASSETS_DIR,
        )
    ]
    if todo_list:
        parts.append(TODO_RULE)
    if context := load_agent_md(working_dir):
        parts.append(f"# User context (AGENT.md)\n\n{context}")
    else:  # say so, or the model goes looking for one
        parts.append(NO_INSTRUCTIONS)
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)
