"""Observe a completed turn with a bounded post-turn evaluator.

An evaluator receives a copy of the trajectory after the agent finishes. Its return value
is written to the session trace, but it cannot change the conversation or the agent's result.

uv run python examples/09_evaluator.py
"""

import asyncio
import tempfile

from tinyharness import AssistantText, EvaluationContext, Session, SessionOptions


def answer_present(context: EvaluationContext) -> dict[str, object]:
    """Check that the completed turn contains a non-empty assistant answer."""
    answers = [message.content or "" for message in context.messages if message.role == "assistant"]
    answer = answers[-1] if answers else ""
    return {
        "passed": bool(answer.strip()),
        "assistant_characters": len(answer),
        "finish_reason": context.reason,
    }


async def main() -> None:
    observed: list[dict[str, object]] = []

    def record_result(context: EvaluationContext) -> dict[str, object]:
        result = answer_present(context)
        observed.append(result)
        return result

    options = SessionOptions(
        working_dir=tempfile.mkdtemp(prefix="tinyharness-"),
        evaluators=[record_result],
        evaluator_timeout=5.0,
    )
    async with Session(options) as session:
        async for event in session.run("In one sentence, why are regression tests useful?"):
            if isinstance(event, AssistantText):
                print(event.text, end="", flush=True)

    print(f"\n\nevaluation: {observed[0]}")


if __name__ == "__main__":
    asyncio.run(main())
