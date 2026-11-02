"""Asking the user a few questions with options, answered by picking one.

In plan mode the agent asks what would change the plan before writing it. Written into a
reply, the questions had to be answered by typing; with Ask, the user picks each answer
(the recommended one first, so Enter takes it) and the agent carries on in the same turn.
A model that doesn't call it still asks in its reply, as before.

Offered only when someone can answer: the CLI, or an SDK `question_callback`. Never in a
run with no one to ask (`-p`) or to subagents.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field

from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


class AskOption(BaseModel):
    """Describe one selectable answer to a question."""

    label: str = Field(description="The answer, in a few words")
    description: str = Field(default="", description="What choosing it means, in one short line")


class AskQuestion(BaseModel):
    """Describe one question and its selectable answers."""

    header: str = Field(description="A one- or two-word label for the question, e.g. 'Format'")
    question: str = Field(description="The question, in one sentence")
    options: list[AskOption] = Field(
        min_length=2,
        max_length=4,
        description="2 to 4 answers; your recommended one first",
    )


class AskInput(ToolInput):
    """Validate the questions submitted to the Ask tool."""

    questions: list[AskQuestion] = Field(min_length=1, max_length=3, description="1 to 3 questions")


# The user's answer to each question (an option's label, or what they typed), or None when
# they'd rather answer in chat.
QuestionCallback = Callable[[list[AskQuestion]], Awaitable[list[str] | None]]


class AskTool(Tool[AskInput]):
    """Ask an available user to choose answers during an agent turn."""

    name = "Ask"
    description = (
        "Ask the user up to three questions whose answers would change what you do (scope, "
        "data, tools, how it's checked), each with 2 to 4 options and your recommended one "
        "first. They pick an answer to each, or type their own, and you get the answers. "
        "Look and research first: ask user-owned preferences directly, but for a discoverable "
        "choice with several viable answers, offer the real candidates you found and their "
        "trade-offs. Each option must be one concrete answer, not a bundle of variants, when the "
        "plan needs one exact choice. Group related questions in one call; ask another round only "
        "if an answer or the research reveals a new high-impact choice."
    )
    Input = AskInput
    read_only = True  # it changes nothing: plan mode offers it
    taken_from_json = True  # it only asks; seen live, Gemma 3 wrote its calls as JSON
    available_in_plan_mode = True
    needs_permission = False

    def __init__(self, answer: QuestionCallback | None) -> None:
        self.answer = answer

    def offered(self) -> bool:
        """Return whether a callback is available to answer questions."""
        return self.answer is not None

    def runs_alongside(self, args: AskInput) -> bool:
        """Return ``False`` because concurrent questions share one input surface."""
        return False  # two at once would each want the keyboard (or the SDK's callback)

    def describe(self, args: AskInput) -> str:
        """Return a compact description of the question count."""
        n = len(args.questions)
        return f"Ask({n} question{'s' if n != 1 else ''})"

    async def run(self, args: AskInput, ctx: ToolContext) -> str:
        """Ask the questions and return the selected answers to the agent."""
        if self.answer is None:
            raise ToolError("No one is here to answer. Choose the recommended options and say so.")
        answers = await self.answer(args.questions)
        if answers is None:
            ctx.hand_back = "awaiting_answer"
            return "The user would rather answer in chat: your turn ends here; they'll reply."
        lines = [f"{q.header}: {a}" for q, a in zip(args.questions, answers, strict=True)]
        return "The user answered:\n" + "\n".join(lines)
