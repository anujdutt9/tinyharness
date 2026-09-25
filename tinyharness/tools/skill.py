from __future__ import annotations

from pydantic import Field

from tinyharness.skills import SkillRegistry, build_skill_prompt
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


class SkillInput(ToolInput):
    name: str = Field(description="Skill name from the list above")
    args: str = Field(default="", description="Optional arguments for the skill")


class SkillTool(Tool[SkillInput]):
    """Lets the model load a skill's instructions on its own initiative.

    Off by default: listing every skill in the tool description costs tokens on
    every turn, while `/<skill>` from the REPL costs nothing until invoked.
    """

    name = "Skill"
    Input = SkillInput
    read_only = True

    def __init__(self, skills: SkillRegistry) -> None:
        self.skills = skills
        listing = "\n".join(f"- {s.name}: {s.description}" for s in skills.list()) or "(none)"
        self.description = (
            "Load a named skill. Returns the skill's instructions, which you should then follow.\n\n"
            f"Available skills:\n{listing}"
        )

    def describe(self, args: SkillInput) -> str:
        return f"Skill({args.name})"

    async def run(self, args: SkillInput, ctx: ToolContext) -> str:
        skill = self.skills.get(args.name)
        if skill is None:
            raise ToolError(f"unknown skill: {args.name!r}")
        return build_skill_prompt(skill, args.args)
