"""The permission gate: decides whether a tool call may run.

Layers are checked in order and the first one with an opinion wins:

    1. tool     bookkeeping tools (TodoWrite) never need permission
    2. path     the target path must resolve
    3. hooks    PreToolUse hooks can allow or deny
    4. rules    config rules and "don't ask again" answers (deny beats allow)
    5. mode     plan / bypass / reads-are-free / acceptEdits / auto (classifier)
    6. scratch  the agent's own scratch directory
    7. ask      the permission callback (interactive prompt or SDK policy)

With no callback configured, anything that reaches step 7 is denied.
"""

from __future__ import annotations

import glob
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tinyharness.permissions.modes import PermissionMode
from tinyharness.permissions.rules import PermissionRules, Rule

if TYPE_CHECKING:
    from tinyharness.hooks import HookRunner
    from tinyharness.permissions.classifier import AutoModeClassifier
    from tinyharness.tools.base import Tool
    from tinyharness.workspace import Scratch, Workspace


@dataclass
class PermissionRequest:
    """What the user (or an SDK policy) is asked to approve."""

    tool: str
    args: dict[str, Any]
    summary: str  # e.g. "Edit(src/app.py, -1/+4 lines)"
    path: str | None  # resolved target path, if the tool touches one
    remember_label: str  # what "don't ask again" would cover, in plain words


@dataclass
class Decision:
    allow: bool
    reason: str = ""
    source: str = ""  # tool | path | hook | rule | mode | classifier | prompt | policy ...
    remember: bool = False  # allow: apply the request's "don't ask again" scope
    interrupt: bool = False  # deny: skip the rest of this turn's tool calls
    feedback: str = ""  # with interrupt: what to do instead, sent as the user's next message


PermissionCallback = Callable[[PermissionRequest], Awaitable[Decision]]


@dataclass
class PermissionGate:
    workspace: Workspace
    mode: PermissionMode = PermissionMode.DEFAULT
    rules: PermissionRules = field(default_factory=PermissionRules)
    hooks: HookRunner | None = None
    classifier: AutoModeClassifier | None = None
    callback: PermissionCallback | None = None
    scratch: Scratch | None = None

    async def check(
        self,
        tool: Tool[Any],
        args: dict[str, Any],
        target_path: str | None,
        *,
        summary: str = "",
        history_tail: str = "",
    ) -> Decision:
        if not tool.needs_permission:
            return Decision(True, "no permission needed", "tool")

        path: Path | None = None
        if target_path is not None:
            path = self.workspace.resolve(target_path)
            if path is None:
                return Decision(False, f"invalid path: {target_path}", "path")
        trusted = path is not None and self.workspace.is_trusted(path)

        if self.hooks is not None:
            verdict = await self.hooks.fire("PreToolUse", {"tool": tool.name, "args": args})
            if verdict and verdict.get("decision") in ("allow", "deny"):
                allow = verdict["decision"] == "allow"
                return Decision(allow, verdict.get("reason", f"hook {verdict['decision']}"), "hook")

        match self.rules.check(tool.name, str(path) if path else None):
            case "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            case "allow":
                return Decision(True, "matches an allow rule", "rule")

        if (decision := await self._check_mode(tool, args, path, trusted, history_tail)) is not None:
            return decision

        if path is not None and self.scratch is not None and self.scratch.contains(path):
            return Decision(True, "agent scratch directory", "mode")

        if self.callback is None:
            return Decision(False, "no permission handler available", "prompt")
        request = PermissionRequest(
            tool=tool.name,
            args=args,
            summary=summary or tool.name,
            path=str(path) if path else None,
            remember_label=self._remember_label(tool, path, trusted),
        )
        decision = await self.callback(request)
        if decision.allow and decision.remember:
            self._remember(tool, path, trusted)
        return decision

    async def _check_mode(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool, history_tail: str
    ) -> Decision | None:
        if self.mode == PermissionMode.BYPASS:
            return Decision(True, "bypassPermissions mode", "mode")
        if self.mode == PermissionMode.PLAN and not tool.read_only:
            return Decision(False, "plan mode blocks mutating tools", "mode")
        if tool.read_only and (path is None or trusted):
            return Decision(True, "read-only, inside the working directory", "mode")
        if self.mode == PermissionMode.ACCEPT_EDITS and trusted:
            return Decision(True, "acceptEdits (inside the working directory)", "mode")
        if self.mode == PermissionMode.AUTO and self.classifier is not None:
            verdict = await self.classifier.classify(tool.name, args, history_tail=history_tail)
            if verdict.decision != "ask":
                return Decision(verdict.decision == "allow", verdict.reason, "classifier")
        return None

    # -- "don't ask again" ------------------------------------------------------
    # Edits inside the project switch the session to acceptEdits (visible in the
    # mode indicator). Anything outside the project is remembered per directory.

    def _remember_label(self, tool: Tool[Any], path: Path | None, trusted: bool) -> str:
        if path is None:
            return f"don't ask again for {tool.name} this session"
        if trusted:
            return "allow all edits in this project for this session"
        return f"allow {tool.name} in {_directory(path)}/ for this session"

    def _remember(self, tool: Tool[Any], path: Path | None, trusted: bool) -> None:
        if path is None:
            self.rules.add_allow(Rule(tool.name))
        elif trusted:
            self.mode = PermissionMode.ACCEPT_EDITS
        else:
            self.rules.add_allow(Rule(tool.name, f"{glob.escape(str(_directory(path)))}/*"))


def _directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent
