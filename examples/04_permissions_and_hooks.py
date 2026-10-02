"""Headless safety policy: a permission callback plus a PreToolUse hook.

The callback is asked whenever the gate would normally prompt a human; in `ask` mode
that's every edit and command, so the policy sees them all. (In the default mode, edits
inside the project and sandboxed commands run without asking.) The hook runs before
*every* tool call and can veto it, even in bypass mode.

A request with a `warning` is one the harness wants a person to see every time: a
command asking to run outside the sandbox, or a change to files that run code later
(.git/hooks, .tinyharness/). A policy with no person behind it should say no to those.

    uv run python examples/04_permissions_and_hooks.py
"""

import asyncio
import re
import tempfile
from typing import Any

from tinyharness import (
    AssistantText,
    Decision,
    Hook,
    PermissionDecision,
    PermissionRequest,
    SessionOptions,
    ToolUse,
    query,
)

SENSITIVE = re.compile(r"(secret|credential|token|password)", re.IGNORECASE)


async def policy(request: PermissionRequest) -> Decision:
    """Allow what's asked, except what needs a person, and writes to sensitive-looking paths."""
    if request.warning:
        return Decision(allow=False, reason=f"needs a person: {request.warning}", source="policy")
    if request.tool in {"Write", "Edit"} and SENSITIVE.search(request.path or ""):
        return Decision(allow=False, reason="path looks sensitive", source="policy")
    return Decision(allow=True, reason="policy", source="policy")


def no_python_writes(event: str, payload: dict[str, Any]) -> dict[str, str] | None:
    """Hooks receive the tool name and raw arguments; return a decision or None to abstain."""
    if payload["tool"] == "Write" and str(payload["args"].get("path", "")).endswith(".py"):
        return {"decision": "deny", "reason": "this project doesn't allow new Python files"}
    return None


async def main() -> None:
    options = SessionOptions(
        working_dir=tempfile.mkdtemp(prefix="tinyharness-"),
        permission_mode="ask",  # every edit and command goes to the policy
        permission_callback=policy,
        hooks=[Hook(event="PreToolUse", type="python", callable=no_python_writes)],
    )
    prompt = (
        "Create notes.md with three tips about Python, then create secrets.txt and tips.py, "
        "then install the requests package with pip."
    )
    async for event in query(prompt, options):
        match event:
            case AssistantText(text=text):
                print(text, end="", flush=True)
            case ToolUse(summary=summary):
                print(f"\n● {summary}")
            case PermissionDecision(allowed=allowed, source=source, reason=reason):
                print(f"  {'allowed' if allowed else 'DENIED'} by {source}: {reason}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
