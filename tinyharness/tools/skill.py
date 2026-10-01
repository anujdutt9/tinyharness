from __future__ import annotations

from pydantic import Field

from tinyharness.skills import SkillRegistry, build_skill_prompt
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


class SkillInput(ToolInput):
    name: str = Field(description="Skill name from the list above")
    args: str = Field(default="", description="Optional arguments for the skill")


class SkillTool(Tool[SkillInput]):
    """Lets the model load a skill's instructions on its own initiative.

    Its description lists each skill's name and the start of its description (a few
    hundred tokens, cached with the rest of the prompt); the full instructions are sent
    only when a skill is loaded.
    """

    name = "Skill"
    Input = SkillInput
    read_only = True

    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills
        listing = "\n".join(f"- {s.name}: {_brief(s.description)}" for s in skills.list()) or "(none)"
        self.description = (
            "Load a skill: expert instructions for a kind of task, which you then follow. Before "
            "specialized work (designing a page or interface, a slide deck, a document, a review), "
            "check this list and load the skill that matches; it beats improvising. Load it again "
            "in each subagent that needs it.\n\n"
            f"Available skills:\n{listing}"
        )

    def describe(self, args: SkillInput) -> str:
        return f"Skill({args.name})"

    async def run(self, args: SkillInput, ctx: ToolContext) -> str:
        skill = self.skills.get(args.name)
        if skill is None:
            raise ToolError(f"unknown skill: {args.name!r}")
        return build_skill_prompt(skill, args.args)


BRIEF_CHARS = 240  # of each skill's description, in the listing: enough to know when it applies


def _brief(description: str) -> str:
    text = " ".join(description.split())
    return text if len(text) <= BRIEF_CHARS else text[: BRIEF_CHARS - 1].rsplit(" ", 1)[0] + "…"
