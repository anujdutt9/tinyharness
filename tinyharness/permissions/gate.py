"""The permission gate: decides whether a tool call may run.

Layers are checked in order and the first one with an opinion wins:

    1. tool     bookkeeping tools (TodoWrite) never need permission
    2. path     the target path must resolve
    3. hooks    PreToolUse hooks can allow or deny
    4. rules    config rules and "don't ask again" answers (deny beats allow)
    5. mode     plan / bypass / reads-are-free / acceptEdits / auto (see `_auto`)
    6. scratch  the agent's own scratch directory
    7. ask      the permission callback (interactive prompt or SDK policy)

With no callback configured, anything that reaches step 7 is denied.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from tinyharness.permissions.modes import PermissionMode
from tinyharness.permissions.rules import DOMAIN, PermissionRules, Rule

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
    reason: str = ""  # why auto mode is asking, when it is


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
        target_host: str | None = None,
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

        site = f"{DOMAIN}{target_host.lower()}" if target_host else None
        match self.rules.check(tool.name, str(path) if path else site):
            case "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            case "allow":
                return Decision(True, "matches an allow rule", "rule")

        risk = self._auto_risk(tool, args, path, trusted) if self.mode == PermissionMode.AUTO else None
        if risk is None and (decision := await self._check_mode(tool, args, path, trusted, history_tail)):
            return decision
        if risk is not None and self.classifier and (blocked := self.classifier.fast_filter(args)):
            return Decision(False, blocked.reason, "classifier")

        if self._in_scratch(path):
            return Decision(True, "agent scratch directory", "mode")

        if self.callback is None:
            return Decision(False, "no permission handler available", "prompt")
        request = PermissionRequest(
            tool=tool.name,
            args=args,
            summary=summary or tool.name,
            path=str(path) if path else None,
            remember_label=self._remember_label(tool, path, trusted, target_host),
            reason=risk or "",
        )
        decision = await self.callback(request)
        if decision.allow and decision.remember:
            self._remember(tool, path, trusted, site)
        elif decision.allow and tool.read_only and (site or path is not None):
            # A plain "yes" to reading something covers reading it again this session.
            self.rules.add_allow(Rule(tool.name, site or glob.escape(str(path))))
        return decision

    async def _check_mode(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool, history_tail: str
    ) -> Decision | None:
        if self.mode == PermissionMode.BYPASS:
            return Decision(True, "bypassPermissions mode", "mode")
        if self.mode == PermissionMode.PLAN and not tool.read_only:
            return Decision(False, "plan mode blocks mutating tools", "mode")
        if tool.read_only and not tool.network and (path is None or trusted):
            return Decision(True, "read-only, inside the working directory", "mode")
        if self.mode == PermissionMode.ACCEPT_EDITS and trusted:
            return Decision(True, "acceptEdits (inside the working directory)", "mode")
        if self.mode == PermissionMode.AUTO:
            return await self._auto(tool, args, path, trusted, history_tail)
        return None

    def _in_scratch(self, path: Path | None) -> bool:
        return path is not None and self.scratch is not None and self.scratch.contains(path)

    # -- auto mode --------------------------------------------------------------
    # Routine work runs without asking; risky actions always ask (see `_auto_risk`);
    # anything in between goes to the classifier, which asks when unsure.

    def _auto_risk(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool
    ) -> str | None:
        """Why auto mode must ask about this call, or None if it may decide itself."""
        if tool.read_only or not tool.needs_permission or self._in_scratch(path):
            return None
        if path is not None and _GIT_INTERNALS.search(str(path)):
            return "it changes version-control internals (.git)"
        if path is not None and not trusted:
            return "it changes a file outside the project"
        if (reason := tool.risk(args, path)) is not None:
            return f"it {reason}"
        return None

    async def _auto(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool, history_tail: str
    ) -> Decision | None:
        if path is not None and trusted:
            return Decision(True, "auto: routine work inside the project", "mode")
        if self._in_scratch(path):
            return Decision(True, "agent scratch directory", "mode")
        if tool.network and not urlsplit(str(args.get("url", ""))).query:
            return Decision(True, "auto: reading a public web page", "mode")
        if self.classifier is None:
            return None
        verdict = await self.classifier.classify(tool.name, args, history_tail=history_tail)
        if verdict.decision == "ask":
            return None
        return Decision(verdict.decision == "allow", verdict.reason, "classifier")

    # -- "don't ask again" ------------------------------------------------------
    # Edits inside the project switch the session to acceptEdits (visible in the
    # mode indicator). Anything outside the project is remembered per directory.

    def _remember_label(
        self, tool: Tool[Any], path: Path | None, trusted: bool, host: str | None = None
    ) -> str:
        if host:
            return f"allow {tool.name} for {host} this session"
        if path is None:
            return f"don't ask again for {tool.name} this session"
        if trusted:
            return "allow all edits in this project for this session"
        return f"allow {tool.name} in {_directory(path)}/ for this session"

    def _remember(self, tool: Tool[Any], path: Path | None, trusted: bool, site: str | None = None) -> None:
        if site:
            self.rules.add_allow(Rule(tool.name, site))
        elif path is None:
            self.rules.add_allow(Rule(tool.name))
        elif trusted:
            self.mode = PermissionMode.ACCEPT_EDITS
        else:
            self.rules.add_allow(Rule(tool.name, f"{glob.escape(str(_directory(path)))}/*"))


_GIT_INTERNALS = re.compile(r"(^|/)\.git(/|$)")


def _directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent
