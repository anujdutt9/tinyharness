"""Hooks: user code that runs at lifecycle events and can veto tool calls.

A hook is either a shell command (configured in hooks.toml) or a Python
callable (passed via the SDK). It receives the event payload as JSON on stdin
(or as arguments) and may return `{"decision": "allow"|"deny", "reason": ...}`.
A shell command that exits non-zero denies, with its output as the reason.

What a result does depends on the event:
  PreToolUse   "deny" stops the call; the reason goes to the model
  PostToolUse  a "reason" is added to the tool's result, for the model to act on
  Stop         "block" (or "deny") keeps the agent going, with the reason as its next
               instruction: a check that must pass before the turn ends (twice per turn at most)

    # ~/.tinyharness/hooks.toml or <project>/.tinyharness/hooks.toml
    [[hook]]
    event = "PreToolUse"
    matcher = { tool = "Write" }
    command = "./scripts/check-write.sh"

    [[hook]]
    event = "Stop"
    command = "uv run pytest -q -x"
    timeout = 300
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
from tinyharness.process import kill_group
from tinyharness.security import redact_secrets, sanitized_env

log = logging.getLogger(__name__)

HookEvent = Literal["PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop", "SessionStart"]
EVENTS: frozenset[str] = frozenset(HookEvent.__args__)  # type: ignore[attr-defined]

HookResult = dict[str, Any]
HookCallable = Callable[[str, dict[str, Any]], Any]


@dataclass
class Hook:
    """Configure one shell or Python lifecycle hook."""

    event: str
    type: Literal["shell", "python"] = "shell"
    matcher: dict[str, Any] = field(default_factory=dict)
    command: str | None = None  # type="shell"
    callable: HookCallable | None = None  # type="python"
    name: str = ""
    timeout: float | None = None  # seconds; None: the runner's default

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
            return await self._run_shell(hook.command, event, payload, hook.timeout or self.timeout)
        return None

    async def _run_shell(
        self, cmd: str, event: str, payload: dict[str, Any], timeout: float
    ) -> HookResult | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.cwd,
                env=sanitized_env(),  # no credentials in the hook's environment
                start_new_session=True,  # its own process group, so a timeout stops all of it
            )
        except FileNotFoundError as e:
            return {"decision": "ask", "reason": f"hook command not found: {e}"}
        try:
            stdout, stderr = await asyncio.wait_for(
                # Hooks may forward their input (to an audit log, a chat): no credentials in it.
                proc.communicate(redact_secrets(json.dumps({"event": event, **payload})).encode()),
                timeout=timeout,
            )
        except TimeoutError:
            await kill_group(proc)
            return {"decision": "ask", "reason": f"hook timed out after {timeout:g}s"}
        if proc.returncode != 0:
            # What it printed is the useful part (a failing test, a lint error), the end of it most.
            said = (stdout + stderr).decode(errors="replace").strip()[-3000:]
            return {
                "decision": "deny",
                "reason": f"`{cmd}` exited {proc.returncode}" + (f":\n{said}" if said else ""),
            }
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
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as e:
        log.warning("skipping hooks in %s: %s", path, e)
        return []
    entries = data.get("hook") or []
    if not isinstance(entries, list):
        log.warning("skipping hooks in %s: `hook` must be a list of [[hook]] tables", path)
        return []
    hooks = []
    for entry in entries:
        if not isinstance(entry, dict):
            log.warning("skipping hook entry %r in %s: expected a [[hook]] table", entry, path)
            continue
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
                timeout=float(entry["timeout"]) if isinstance(entry.get("timeout"), int | float) else None,
            )
        )
    return hooks


def load_hooks(working_dir: Path, *, project: bool = True) -> list[Hook]:
    """Hooks from ~/.tinyharness/hooks.toml and, if `project` (a trusted folder; see
    tinyharness/trust.py), <working-dir>/.tinyharness/hooks.toml."""
    dirs = search_dirs(working_dir, "hooks.toml")
    return [h for d in (dirs if project else dirs[:1]) for h in load_hooks_file(d)]
