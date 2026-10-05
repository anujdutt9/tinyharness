"""The permission gate: decides whether a tool call may run.

Layers are checked in order and the first one with an opinion wins:

    1. tool     bookkeeping tools (TodoWrite) never need permission
    2. path     the target path must resolve
    3. hooks    PreToolUse hooks can allow or deny
    4. rules    config rules and "don't ask again" answers (deny beats allow)
       escape   a call that leaves its sandbox (Bash with sandbox=false), or that
                changes a file that runs code outside it (hooks, plugins, git hooks,
                settings), skips allow rules. Auto may keep a non-destructive command
                contained; other escapes always ask and can't be remembered
    5. mode     plan / bypass / reads-are-free / project edits / sandboxed commands /
                auto (see `_auto`)
    6. scratch  the agent's own scratch directory
    7. ask      the permission callback (interactive prompt or SDK policy)

With no callback configured, anything that reaches step 7 is denied.
"""

from __future__ import annotations

import glob
import re
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
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
    remember_label: str  # what "don't ask again" would cover, in plain words; "" = not offered
    reason: str = ""  # why auto mode is asking, when it is
    warning: str = ""  # something the user must know before answering (e.g. no sandbox)
    agent: str = ""  # the subagent asking, if it isn't the main agent


@dataclass
class Decision:
    allow: bool
    reason: str = ""
    source: str = ""  # tool | path | hook | rule | mode | classifier | prompt | policy ...
    remember: bool = False  # allow: apply the request's "don't ask again" scope
    interrupt: bool = False  # deny: skip the rest of this turn's tool calls
    feedback: str = ""  # with interrupt: what to do instead, sent as the user's next message
    # Auto mode can replace an unnecessary sandbox escape with equivalent contained args.
    arguments: dict[str, Any] | None = None
    # allow: the resolved path this was judged on. The tool must act on exactly this path,
    # not resolve it again: a symlink swapped in between would point it somewhere else.
    path: Path | None = None


PermissionCallback = Callable[[PermissionRequest], Awaitable[Decision]]


# Least to most permissive: a subagent runs in the less permissive of the session's mode
# and its own `permission_mode` (see PermissionGate.child).
_PERMISSIVENESS = (
    PermissionMode.PLAN,
    PermissionMode.ASK,
    PermissionMode.DEFAULT,
    PermissionMode.AUTO,
    PermissionMode.BYPASS,
)


def stricter(mode: PermissionMode, limit: PermissionMode | None) -> PermissionMode:
    """`mode`, held to `limit` if that's less permissive."""
    if limit is None:
        return mode
    return min(mode, limit, key=_PERMISSIVENESS.index)


@dataclass(eq=False)  # compared by identity: gates are tracked in a WeakSet
class PermissionGate:
    workspace: Workspace
    mode: PermissionMode = PermissionMode.DEFAULT
    rules: PermissionRules = field(default_factory=PermissionRules)
    hooks: HookRunner | None = None
    classifier: AutoModeClassifier | None = None
    # Where content from other machines in the conversation came from (ToolContext.outside).
    # While there is any, calls that can send data out ask a person ("Rule of Two").
    outside: Callable[[], list[str]] = field(default=lambda: [], repr=False)
    callback: PermissionCallback | None = None
    scratch: Scratch | None = None
    protected: tuple[Path, ...] = ()  # changing anything under these always asks (see `_protected`)
    agent: str = ""  # a subagent's gate: whose (shown when it asks)
    limit: PermissionMode | None = None  # a subagent's own permission_mode: never more permissive
    _children: weakref.WeakSet[PermissionGate] = field(default_factory=weakref.WeakSet, repr=False)

    def set_mode(self, mode: PermissionMode) -> None:
        """Change mode, for this gate and every subagent gate made from it: the mode you
        pick applies to all the work, including subagents already running."""
        self.mode = stricter(mode, self.limit)
        for child in list(self._children):
            child.set_mode(mode)

    def child(self, agent: str, limit: PermissionMode | None) -> PermissionGate:
        """A subagent's gate: the same rules and prompts, the session's mode (held to the
        agent's own `limit`), and it follows later mode changes (see `set_mode`)."""
        gate = replace(
            self, mode=stricter(self.mode, limit), agent=agent, limit=limit, _children=weakref.WeakSet()
        )
        self._children.add(gate)
        return gate

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
            path = self.workspace.resolve(target_path)  # the one resolution: see Decision.path
            if path is None:
                return Decision(False, f"invalid path: {target_path}", "path")
        decision = await self._decide(tool, args, path, target_host, summary, history_tail)
        if decision.allow and path is not None:
            decision.path = path
        return decision

    async def _decide(
        self,
        tool: Tool[Any],
        args: dict[str, Any],
        path: Path | None,
        target_host: str | None,
        summary: str,
        history_tail: str,
    ) -> Decision:
        trusted = path is not None and self.workspace.is_trusted(path)

        if self.hooks is not None:
            verdict = await self.hooks.fire("PreToolUse", {"tool": tool.name, "args": args})
            if verdict and verdict.get("decision") in ("allow", "deny"):
                allow = verdict["decision"] == "allow"
                return Decision(allow, verdict.get("reason", f"hook {verdict['decision']}"), "hook")

        site = f"{DOMAIN}{target_host.lower()}" if target_host else None
        sandbox_escape = tool.escapes(args)
        protected_escape = self._protected(tool, path)
        persistent_escape = self._persists_outside(tool)
        escape = sandbox_escape or protected_escape or persistent_escape
        match self.rules.check(tool.name, str(path) if path else site):
            case "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            case "allow" if escape is None:  # rules were written with the sandbox in place
                return Decision(True, "matches an allow rule", "rule")

        if escape is not None:
            if self.mode == PermissionMode.BYPASS:
                return Decision(True, "bypassPermissions mode", "mode")
            if self.mode == PermissionMode.PLAN:
                return Decision(False, "plan mode blocks mutating tools", "mode")
            risk = None
            if (
                self.mode == PermissionMode.AUTO
                and sandbox_escape is not None
                and protected_escape is None
                and persistent_escape is None
                and not self.outside()
            ):
                risk = self._auto_risk(tool, args, path, trusted)
                contained = tool.contained_args(args)
                if risk is None and contained is not None:
                    return Decision(
                        True,
                        "kept the non-destructive command inside the sandbox",
                        "contained",
                        arguments=contained,
                    )
            request = PermissionRequest(
                agent=self.agent,
                tool=tool.name,
                args=args,
                summary=summary or tool.name,
                path=str(path) if path else None,
                remember_label="",
                reason=risk or "",
                warning=escape + self._outside_warning(),
            )
            return await self._ask(request)

        risk = self._auto_risk(tool, args, path, trusted) if self.mode == PermissionMode.AUTO else None
        if risk is None and (decision := await self._check_mode(tool, args, path, trusted, history_tail)):
            return decision
        if risk is not None and self.classifier and (blocked := self.classifier.fast_filter(args)):
            return Decision(False, blocked.reason, "classifier")

        if self._in_scratch(path):
            return Decision(True, "agent scratch directory", "mode")

        # Auto mode asks about every risky action case by case: a "don't ask again"
        # would stop it judging the next one (and used to switch modes).
        case_by_case = risk is not None
        request = PermissionRequest(
            agent=self.agent,
            tool=tool.name,
            args=args,
            summary=summary or tool.name,
            path=str(path) if path else None,
            remember_label="" if case_by_case else self._remember_label(tool, path, trusted, target_host),
            reason=risk or "",
            warning=self._outside_warning().removeprefix("; ") if tool.network else "",
        )
        decision = await self._ask(request)
        if decision.allow and decision.remember and request.remember_label:
            self._remember(tool, path, trusted, site)
        elif decision.allow and tool.read_only and (site or path is not None):
            # A plain "yes" to reading something covers reading it again this session.
            for name in _readers(tool) if path is not None and not site else (tool.name,):
                self.rules.add_allow(Rule(name, site or glob.escape(str(path))))
        return decision

    def _outside_warning(self) -> str:
        """Said when a call could send data out while outside content is in the conversation."""
        sources = self.outside()
        if not sources:
            return ""
        shown = ", ".join(sources[:3]) + (f" and {len(sources) - 3} more" if len(sources) > 3 else "")
        return f"; content from outside ({shown}) is in this conversation and could be steering this"

    def _protected(self, tool: Tool[Any], path: Path | None) -> str | None:
        """Why changing `path` needs the user every time: it's read as code or settings
        later, outside the sandbox (.tinyharness/, .git/hooks, .git/config, ~/.tinyharness)."""
        if path is None or tool.read_only or self._in_scratch(path):
            return None
        for root in self.protected:
            if path == root or path.is_relative_to(root):
                shown = self.workspace.relative(root)
                return f"this changes {shown}, which can run code outside the sandbox (hooks, plugins or settings)"
        return None

    def _persists_outside(self, tool: Tool[Any]) -> str | None:
        """Why saving to memory needs the user now: outside content in the conversation
        could be planting a note that every later session would read as context."""
        if tool.persists and self.outside():
            return "it saves a note that every later session reads"
        return None

    async def _ask(self, request: PermissionRequest) -> Decision:
        if self.callback is None:
            return Decision(
                False, "no one is here to approve it (a non-interactive run): do without it", "prompt"
            )
        return await self.callback(request)

    async def _check_mode(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool, history_tail: str
    ) -> Decision | None:
        if self.mode == PermissionMode.BYPASS:
            return Decision(True, "bypassPermissions mode", "mode")
        if self.mode == PermissionMode.PLAN and not tool.read_only:
            return Decision(False, "plan mode blocks mutating tools", "mode")
        # The agent's own notes; with outside content in the conversation, see _persists_outside.
        if tool.persists and self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO):
            return Decision(True, f"{self.mode.value}: the agent's memory", "mode")
        if tool.read_only and not tool.network and (path is None or trusted):
            return Decision(True, "read-only, inside the working directory", "mode")
        if self.mode == PermissionMode.DEFAULT and trusted:
            return Decision(True, "default: inside the working directory", "mode")
        if self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO) and tool.contained(args):
            return Decision(True, f"{self.mode.value}: runs in the sandbox", "mode")
        # Reading a public page is routine; a URL with a query string could carry data out.
        if tool.network and self.outside():
            return None  # outside content could be steering this: a person decides (see check)
        if self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO) and _public_read(tool, args):
            return Decision(True, f"{self.mode.value}: reading a public web page", "mode")
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
        if self.classifier is None:
            return None
        verdict = await self.classifier.classify(tool.name, args, history_tail=history_tail)
        if verdict.decision == "ask":
            return None
        return Decision(verdict.decision == "allow", verdict.reason, "classifier")

    # -- "don't ask again" ------------------------------------------------------
    # In ask mode, edits inside the project switch the session to default (visible in the
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
        if tool.name in READING_TOOLS:
            return f"allow reading in {_directory(path)}/ for this session"
        return f"allow {tool.name} in {_directory(path)}/ for this session"

    def _remember(self, tool: Tool[Any], path: Path | None, trusted: bool, site: str | None = None) -> None:
        if site:
            self.rules.add_allow(Rule(tool.name, site))
        elif path is None:
            self.rules.add_allow(Rule(tool.name))
        elif trusted:
            self.set_mode(PermissionMode.DEFAULT)
        else:
            folder = glob.escape(str(_directory(path)))
            for name in _readers(tool):
                self.rules.add_allow(Rule(name, f"{folder}/*"))
                if name in READING_TOOLS:
                    self.rules.add_allow(Rule(name, folder))  # Grep or Glob on the folder itself


_GIT_INTERNALS = re.compile(r"(^|/)\.git(/|$)")
# Tools that only look at files: an answer about one covers them all, so reading a folder
# outside the project is one question, not one per tool (Read, then Grep, then Glob).
READING_TOOLS = ("Read", "Grep", "Glob", "Pdf")


def _readers(tool: Tool[Any]) -> tuple[str, ...]:
    """The tools an approval for `tool` covers: every reading tool, if it is one."""
    return READING_TOOLS if tool.name in READING_TOOLS else (tool.name,)


def _directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _public_read(tool: Tool[Any], args: dict[str, Any]) -> bool:
    """A read-only network call to a URL without a query string (WebFetch of a page)."""
    return tool.network and tool.read_only and not urlsplit(str(args.get("url", ""))).query
