"""Hooks: user code that runs at lifecycle events and can veto tool calls.

A hook is either a shell command (configured in hooks.toml) or a Python
callable (passed via the SDK). It receives the event payload as JSON on stdin
(or as arguments) and may return `{"decision": "allow"|"deny", "reason": ...}`.

    # ~/.tinyharness/hooks.toml or <project>/.tinyharness/hooks.toml
    [[hook]]
    event = "PreToolUse"
    matcher = { tool = "Write" }
    command = "./scripts/check-write.sh"
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import shlex
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tinyharness.paths import search_dirs
from tinyharness.security import sanitized_env

log = logging.getLogger(__name__)

HookEvent = Literal["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SessionStart"]
EVENTS: frozenset[str] = frozenset(HookEvent.__args__)  # type: ignore[attr-defined]

HookResult = dict[str, Any]
HookCallable = Callable[[str, dict[str, Any]], Any]


@dataclass
class Hook:
    event: str
    type: Literal["shell", "python"] = "shell"
    matcher: dict[str, Any] = field(default_factory=dict)
    command: str | None = None  # type="shell"
    callable: HookCallable | None = None  # type="python"
    name: str = ""

    def matches(self, payload: dict[str, Any]) -> bool:
        return "tool" not in self.matcher or payload.get("tool") == self.matcher["tool"]


class HookRunner:
    """Runs every hook registered for an event. Any "deny" wins immediately;
    otherwise the last non-empty result is returned."""

    def __init__(
        self, hooks: list[Hook] | None = None, cwd: Path | None = None, timeout: float = 10.0
    ) -> None:
        self.hooks = list(hooks or [])
        self.cwd = cwd
        self.timeout = timeout

    def add(self, hook: Hook) -> None:
        if hook.event not in EVENTS:
            raise ValueError(f"unknown hook event: {hook.event}")
        self.hooks.append(hook)

    async def fire(self, event: str, payload: dict[str, Any]) -> HookResult | None:
        merged: HookResult | None = None
        for hook in self.hooks:
            if hook.event != event or not hook.matches(payload):
                continue
            try:
                result = await self._invoke(hook, event, payload)
            except Exception as e:
                log.warning("hook %s failed: %s", hook.name or hook.command, e)
                result = {"decision": "ask", "reason": f"hook error: {type(e).__name__}: {e}"}
            if not result:
                continue
            if result.get("decision") == "deny":
                return result
            merged = result
        return merged

    async def _invoke(self, hook: Hook, event: str, payload: dict[str, Any]) -> HookResult | None:
        if hook.type == "python" and hook.callable is not None:
            res = hook.callable(event, payload)
            if inspect.isawaitable(res):
                res = await res
            if res is None:
                return None
            return res if isinstance(res, dict) else {"decision": str(res)}
        if hook.type == "shell" and hook.command:
            return await self._run_shell(hook.command, event, payload)
        return None

    async def _run_shell(self, cmd: str, event: str, payload: dict[str, Any]) -> HookResult | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=sanitized_env(),  # hook scripts never see API keys
            )
        except FileNotFoundError as e:
            return {"decision": "ask", "reason": f"hook command not found: {e}"}
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(json.dumps({"event": event, **payload}).encode()),
                timeout=self.timeout,
            )
        except TimeoutError:
            proc.kill()
            return {"decision": "ask", "reason": "hook timed out"}
        if proc.returncode != 0:
            return {"decision": "deny", "reason": f"hook exited {proc.returncode}"}
        out = stdout.decode(errors="replace").strip()
        if not out:
            return None
        try:
            parsed = json.loads(out)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None


def load_hooks_file(path: Path) -> list[Hook]:
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    hooks = []
    for entry in data.get("hook") or []:
        if entry.get("event") not in EVENTS:
            log.warning("skipping hook with unknown event %r in %s", entry.get("event"), path)
            continue
        hooks.append(
            Hook(
                event=entry["event"],
                type=entry.get("type", "shell"),
                matcher=entry.get("matcher") or {},
                command=entry.get("command"),
                name=entry.get("name", path.stem),
            )
        )
    return hooks


def load_hooks(working_dir: Path) -> list[Hook]:
    """Hooks from ~/.tinyharness/hooks.toml and <working-dir>/.tinyharness/hooks.toml."""
    return [h for d in search_dirs(working_dir, "hooks.toml") for h in load_hooks_file(d)]
