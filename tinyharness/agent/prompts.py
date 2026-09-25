"""The system prompt: prompts/system.md + user-provided AGENT.md context."""

from __future__ import annotations

from pathlib import Path

from tinyharness.paths import home_dir, project_dir
from tinyharness.prompts import load_prompt


def agent_md_sources(working_dir: Path) -> list[Path]:
    """AGENT.md files, lowest precedence first: managed, user, project, local."""
    return [
        Path("/etc/tinyharness/AGENT.md"),
        home_dir() / "AGENT.md",
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


def build_system_prompt(
    working_dir: Path,
    scratch_dir: Path,
    *,
    model_name: str,
    model_id: str,
    provider: str,
    extra: str = "",
) -> str:
    parts = [
        load_prompt(
            "system",
            working_dir=working_dir,
            scratch_dir=scratch_dir,
            model_name=model_name,
            model_id=model_id,
            provider=provider,
        )
    ]
    if context := load_agent_md(working_dir):
        parts.append(f"# User context (AGENT.md)\n\n{context}")
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)
