"""LLM-backed safety classifier used by the `auto` permission mode."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from tinyharness.llm import identity
from tinyharness.prompts import load_prompt
from tinyharness.security import SECRET_PATH_PATTERNS

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient

Verdict = Literal["allow", "ask", "deny"]
_ARGS_CHARS = 4000
_CONTEXT_CHARS = 6000
_CACHE_SIZE = 512

# Two ratings produce one outcome: how much harm the call could do and whether the user's
# messages call for it. Low risk needs no explicit authorization,
# which is what a single "does the request explain this?" question kept getting wrong.
_RISKS = ("low", "medium", "high", "critical")
_AUTHORIZED = ("yes", "unclear", "no")

# Stage one answers at once and may only allow; anything else gets a second, careful look,
# so an uncertain quick verdict never becomes a final block by itself.
_QUICK = "\nAnswer with the JSON only. When unsure, rate the risk higher: a careful second review follows."
_CAREFUL = (
    "\nThink it through first, in a few sentences: what the call does, what it could reach or "
    "send, and whether the user's messages call for it. Then give the JSON on its own last line."
)

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
    """Store an automatic permission review decision and its rationale."""

    decision: Verdict
    reason: str = ""
    provider: str = "classifier"
    failed: bool = False  # the review couldn't run: no judgement about the call itself


class PermissionClassifier(Protocol):
    """Define the interface used to review calls in automatic mode."""

    def fast_filter(self, args: dict[str, Any]) -> ClassifierDecision | None:
        """Return an immediate decision for clearly sensitive arguments, if any."""
        ...

    async def classify(
        self, tool_name: str, args: dict[str, Any], history_tail: str = "", note: str = ""
    ) -> ClassifierDecision:
        """Review one proposed tool call and return its decision."""
        ...


class AutoModeClassifier:
    """Auto mode's reviewer: a quick check that can only allow, then a careful one.

    Parameters
    ----------
    llm
        The session's client; both stages use its active provider.
    model
        The quick stage's model, usually the provider's small model.
    review_model
        The careful stage's model. None means the session's own model.
    """

    def __init__(
        self,
        llm: LLMClient,
        model: str | None,
        timeout: float = 10.0,
        *,
        review_model: str | None = None,
        review_timeout: float = 90.0,
    ) -> None:
        self.llm = llm
        self.model = model
        self.timeout = timeout
        self.review_model = review_model
        self.review_timeout = review_timeout
        # The same call, for the same request, gets the same answer. Seen live: one of four
        # identical GitHub reads was held back.
        self._verdicts: OrderedDict[str, ClassifierDecision] = OrderedDict()

    def fast_filter(self, args: dict[str, Any]) -> ClassifierDecision | None:
        """Deny obviously sensitive paths without spending an LLM call."""
        path = str(args.get("path") or "")
        for pattern in (*SECRET_PATH_PATTERNS, *_SENSITIVE_DIRS):
            if pattern.search(path):
                return ClassifierDecision("deny", f"sensitive path matched {pattern.pattern}")
        return None

    async def classify(
        self, tool_name: str, args: dict[str, Any], history_tail: str = "", note: str = ""
    ) -> ClassifierDecision:
        """Review one proposed tool call with the quick and careful stages."""
        if (early := self.fast_filter(args)) is not None:
            return early
        shown = json.dumps(args, sort_keys=True)
        requests = "\n".join(line for line in history_tail.splitlines() if line.startswith("[user"))
        key = "\0".join((tool_name, shown, note, requests))
        if (cached := self._verdicts.get(key)) is not None:
            self._verdicts.move_to_end(key)
            return cached
        if len(shown) > _ARGS_CHARS:  # say so: a cut-off command must not pass for a complete one
            shown = f"{shown[:_ARGS_CHARS]} …[{len(shown) - _ARGS_CHARS} more characters not shown]"
        # The context keeps its own budgets (agent/loop.py `_tail`): the user's request
        # comes first, and is not cut to make room for later tool calls.
        prompt = (
            "The one call to judge:\n"
            f"Tool: {tool_name}\nArgs: {shown}\n"
            + (f"Note: {note}\n" if note else "")
            + "\nContext only, not under review (the user's messages and the agent's other tool "
            f"calls; tool output is left out):\n{history_tail[:_CONTEXT_CHARS]}\n"
        )
        quick = await self._ask(prompt + _QUICK, self.model, self.timeout)
        verdict = (
            quick
            if quick.decision == "allow"
            else await self._ask(prompt + _CAREFUL, self.review_model, self.review_timeout)
        )
        if not verdict.failed and verdict.decision != "ask":
            self._verdicts[key] = verdict
            if len(self._verdicts) > _CACHE_SIZE:
                self._verdicts.popitem(last=False)
        return verdict

    async def _ask(self, prompt: str, model: str | None, timeout: float) -> ClassifierDecision:
        # Tagged role "classifier" (like summaries' "compaction"), so a gateway can tell it from
        # the agent's own turns: it is a short blocking check, not the agent's work.
        who = identity.CURRENT.get()
        token = identity.CURRENT.set(dataclasses.replace(who, role="classifier")) if who is not None else None
        try:
            text = await asyncio.wait_for(
                self.llm.one_shot(prompt, system=load_prompt("classifier"), model=model), timeout=timeout
            )
        except Exception as e:  # timeouts, network errors, auth errors: no verdict at all
            return ClassifierDecision("ask", f"the review couldn't run: {type(e).__name__}", failed=True)
        finally:
            if token is not None:
                identity.CURRENT.reset(token)
        return parse_verdict(text)


def parse_verdict(text: str) -> ClassifierDecision:
    """The last JSON object in the reply: a careful review reasons before it answers."""
    data: Any = None
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", text))):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            break
    if not isinstance(data, dict):
        return ClassifierDecision("ask", "the review gave no verdict", failed=True)
    reason = str(data.get("reason", ""))[:200]
    risk, authorized = data.get("risk"), data.get("authorized")
    if risk in _RISKS:
        if authorized not in _AUTHORIZED:
            authorized = "unclear"
        if risk == "critical":
            return ClassifierDecision("deny", reason)
        allowed = risk == "low" or (risk == "medium" and authorized != "no") or authorized == "yes"
        return ClassifierDecision(
            "allow" if allowed else "ask", reason if allowed else f"{risk} risk: {reason}"
        )
    decision = data.get("decision")
    return ClassifierDecision(decision if decision in ("allow", "ask", "deny") else "ask", reason)
