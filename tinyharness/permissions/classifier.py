"""LLM-backed safety classifier used by the `auto` permission mode."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from tinyharness.prompts import load_prompt
from tinyharness.security import SECRET_PATH_PATTERNS

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient

Verdict = Literal["allow", "ask", "deny"]

# Stricter than the read-time denylist: anything under these directories.
_SENSITIVE_DIRS = (
    re.compile(r"(^|/)\.ssh(/|$)"),
    re.compile(r"(^|/)\.aws(/|$)"),
    re.compile(r"(^|/)\.gnupg(/|$)"),
    re.compile(r"^/etc/"),
    re.compile(r"^/var/"),
)


@dataclass
class ClassifierDecision:
    decision: Verdict
    reason: str = ""


class AutoModeClassifier:
    def __init__(self, llm: LLMClient, model: str | None, timeout: float = 5.0) -> None:
        self.llm = llm
        self.model = model
        self.timeout = timeout

    def fast_filter(self, args: dict[str, Any]) -> ClassifierDecision | None:
        """Deny obviously sensitive paths without spending an LLM call."""
        path = str(args.get("path") or "")
        for pattern in (*SECRET_PATH_PATTERNS, *_SENSITIVE_DIRS):
            if pattern.search(path):
                return ClassifierDecision("deny", f"sensitive path matched {pattern.pattern}")
        return None

    async def classify(
        self, tool_name: str, args: dict[str, Any], history_tail: str = ""
    ) -> ClassifierDecision:
        if (early := self.fast_filter(args)) is not None:
            return early
        prompt = (
            f"Tool: {tool_name}\nArgs: {json.dumps(args)[:1000]}\n"
            f"Recent context (the user's messages and the agent's tool calls; tool output is "
            f"left out):\n{history_tail[-1500:]}\n"
        )
        try:
            text = await asyncio.wait_for(
                self.llm.one_shot(prompt, system=load_prompt("classifier"), model=self.model),
                timeout=self.timeout,
            )
        except Exception as e:  # timeouts, network errors, auth errors: fall back to asking
            return ClassifierDecision("ask", f"classifier unavailable: {type(e).__name__}")
        return parse_verdict(text)


def parse_verdict(text: str) -> ClassifierDecision:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return ClassifierDecision("ask", "classifier returned non-JSON")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return ClassifierDecision("ask", "classifier returned malformed JSON")
    decision = data.get("decision")
    if decision not in ("allow", "ask", "deny"):
        decision = "ask"
    return ClassifierDecision(decision, str(data.get("reason", ""))[:200])
