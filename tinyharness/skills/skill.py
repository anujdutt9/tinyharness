"""Skills: reusable instructions packaged as `<name>/SKILL.md` (+ optional assets).

Only a skill's name and description are loaded at startup. The body enters the
model's context when the skill is invoked, and bundled reference files are read
lazily, step by step. This is "progressive disclosure".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tinyharness.frontmatter import read_frontmatter_file
from tinyharness.paths import search_dirs


@dataclass
class Skill:
    """Contain one installed skill and its metadata."""

    name: str
    description: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    model: str | None = None
    source: Path | None = None  # path to SKILL.md

    @property
    def directory(self) -> Path | None:
        return self.source.parent if self.source else None


def parse_skill_md(path: Path) -> Skill | None:
    parsed = read_frontmatter_file(path)
    if parsed is None:
        return None
    meta, body = parsed
    return Skill(
        name=str(meta.get("name") or path.parent.name),
        description=str(meta.get("description") or ""),
        body=body,
        allowed_tools=list(meta.get("allowed-tools") or meta.get("allowed_tools") or []),
        model=meta.get("model"),
        source=path,
    )


def scan_skills(root: Path) -> list[Skill]:
    """All `<root>/*/SKILL.md` files that parse."""
    if not root.is_dir():
        return []
    return [s for p in sorted(root.glob("*/SKILL.md")) if (s := parse_skill_md(p))]


def load_skills(working_dir: Path) -> list[Skill]:
    """Skills from ~/.tinyharness/skills and <working-dir>/.tinyharness/skills. The user-global
    copy wins on a name clash, so a project can't shadow a trusted skill."""
    seen: dict[str, Skill] = {}
    for directory in search_dirs(working_dir, "skills"):
        for skill in scan_skills(directory):
            seen.setdefault(skill.name, skill)
    return list(seen.values())


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills or []}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        return list(self._skills.values())

    def __len__(self) -> int:
        return len(self._skills)


def build_skill_prompt(skill: Skill, args: str = "") -> str:
    """The instruction sent to the model when a skill is invoked."""
    # Said outright: a model has read a loaded skill as the skill's help page, waited for it
    # to "run", and loaded it again. Loading does no work; the model does the work.
    parts = [
        f"# Skill: {skill.name}\n{skill.description}\n\n"
        "Skill loaded. Nothing has been done yet: these are instructions for you. Carry them "
        f"out now with your own tools.\n\n{skill.body}"
    ]
    if skill.directory is not None:
        parts.append(
            f"(Bundled files for this skill live in `{skill.directory}`. Read each one only "
            "when the current step needs it; do not pre-load them all.)"
        )
    if args:
        parts.append(f"User arguments: {args}")
    return "\n\n".join(parts)


BRIEF_CHARS = 240  # of each skill's description in the listing: enough to know when it applies


def skills_section(skills: list[Skill]) -> str:
    """The skills, as the model sees them in its system prompt: name, when to use it, and
    where its SKILL.md is. The model loads one by reading that file; no tool is needed."""
    listed = [s for s in skills if s.source is not None]
    if not listed:
        return ""
    lines = [
        "# Skills",
        "Expert instructions for kinds of task. When a task matches one, Read its SKILL.md and "
        "follow it; read the files next to it only when a step needs them.",
    ]
    lines += [f"- {s.name}: {_brief(s.description)} ({s.source})" for s in listed]
    return "\n".join(lines)


def _brief(description: str) -> str:
    text = " ".join(description.split())
    return text if len(text) <= BRIEF_CHARS else text[: BRIEF_CHARS - 1].rsplit(" ", 1)[0] + "…"
