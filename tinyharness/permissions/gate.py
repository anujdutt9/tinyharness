"""The permission gate: decides whether a tool call may run.

Layers are checked in order and the first one with an opinion wins:

    1. tool     bookkeeping tools (TodoWrite) never need permission
    2. path     the target path must resolve
    3. hooks    PreToolUse hooks can allow or deny
    4. boundary contained acquisition capabilities ask before execution; a matching
                session approval may cover the exact operation
       rules    config rules and "don't ask again" answers (deny beats allow)
       escape   a call that crosses a hard boundary, or changes a file that runs code
                outside it (hooks, plugins, git hooks, settings), skips allow rules and
                always asks
    5. mode     plan / bypass / reads-are-free / project edits / sandboxed commands /
                auto (see `_auto`)
    6. scratch  the agent's own scratch directory
    7. ask      the permission callback (interactive prompt or SDK policy)

With no callback configured, anything that reaches step 7 is denied.
"""

from __future__ import annotations

import glob
import json
import re
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from tinyharness.permissions.modes import PermissionMode
from tinyharness.permissions.rules import CAPABILITY, DOMAIN, PermissionRules, Rule
from tinyharness.sandbox.proxy import SERVICE_HOSTS, host_allowed

# What "don't ask again" for an unsandboxed command remembers, so the answer doesn't
# reach sandboxed commands either. A config rule (`allow = ["Bash"]`) covers both.
UNSANDBOXED = CAPABILITY + "unsandboxed"

if TYPE_CHECKING:
    from tinyharness.hooks import HookRunner
    from tinyharness.permissions.classifier import PermissionClassifier
    from tinyharness.tools.base import ApprovalBoundary, Tool
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
    """Describe the permission gate's decision for one tool call."""

    allow: bool
    reason: str = ""
    source: str = ""  # tool | path | hook | rule | mode | classifier | prompt | policy ...
    remember: bool = False  # allow: apply the request's "don't ask again" scope
    interrupt: bool = False  # deny: skip the rest of this turn's tool calls
    feedback: str = ""  # with interrupt: what to do instead, sent as the user's next message
    # allow: the resolved path this was judged on. The tool must act on exactly this path,
    # not resolve it again: a symlink swapped in between would point it somewhere else.
    path: Path | None = None


@dataclass
class _AutoBlocks:
    """Classifier blocks shared by the lead and its subagents in one session."""

    by_action: dict[str, int] = field(default_factory=dict)
    total: int = 0


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
    classifier: PermissionClassifier | None = None
    # Where content from other machines in the conversation came from (ToolContext.outside).
    # While there is any, calls that can send data out are reviewed again.
    outside: Callable[[], list[str]] = field(default=lambda: [], repr=False)
    callback: PermissionCallback | None = None
    # Bash and the gate share this set. Remembering a host must expand the sandbox's
    # actual policy too; remembering only the prompt decision leaves later commands blocked.
    session_network_hosts: set[str] = field(default_factory=set, repr=False)
    # "Don't ask again" answers given while the sandbox was on. They don't carry over to
    # `/sandbox off`: the person agreed to sandboxed commands, not to any command.
    sandboxed_grants: set[Rule] = field(default_factory=set, repr=False)
    # The sandbox's allowlist. Reading a page there needs no review, even with outside
    # content in the conversation: it sends the host nothing but the URL.
    allowed_hosts: tuple[str, ...] = ()
    # An explicitly invoked skill may declare a reviewed set of exact destinations. The
    # first host still asks; choosing the session option grants that finite set, so a
    # redirect chain or multi-source workflow does not turn into one prompt per host.
    # Known services that use several hosts (sandbox/proxy.py SERVICE_HOSTS) work the same way.
    network_bundles: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: dict(SERVICE_HOSTS), repr=False
    )
    scratch: Scratch | None = None
    protected: tuple[Path, ...] = ()  # changing anything under these always asks (see `_protected`)
    agent: str = ""  # a subagent's gate: whose (shown when it asks)
    limit: PermissionMode | None = None  # a subagent's own permission_mode: never more permissive
    _children: weakref.WeakSet[PermissionGate] = field(default_factory=weakref.WeakSet, repr=False)
    _auto_blocks: _AutoBlocks = field(default_factory=_AutoBlocks, repr=False)

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
        approval_boundary: ApprovalBoundary | None = None,
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
        decision = await self._decide(tool, args, path, target_host, approval_boundary, summary, history_tail)
        if decision.allow and path is not None:
            decision.path = path
        return decision

    async def _decide(
        self,
        tool: Tool[Any],
        args: dict[str, Any],
        path: Path | None,
        target_host: str | None,
        approval_boundary: ApprovalBoundary | None,
        summary: str,
        history_tail: str,
    ) -> Decision:
        trusted = path is not None and self.workspace.is_trusted(path)

        site = f"{DOMAIN}{target_host.lower()}" if target_host else None
        sandbox_escape = tool.escapes(args)
        protected_escape = self._protected(tool, path)
        persistent_escape = self._persists_outside(tool)
        escape = sandbox_escape or protected_escape or persistent_escape
        scoped_escape = escape is not None and tool.rememberable_escape(args)
        if self.hooks is not None:
            verdict = await self.hooks.fire("PreToolUse", {"tool": tool.name, "args": args})
            if verdict and verdict.get("decision") in ("allow", "deny"):
                allow = verdict["decision"] == "allow"
                # Hooks may deny anything, but an allow hook cannot weaken containment or
                # approve code/config that will execute later outside it.
                if not allow or (escape is None and approval_boundary is None):
                    return Decision(allow, verdict.get("reason", f"hook {verdict['decision']}"), "hook")

        separate_boundary = approval_boundary is not None and (escape is None or scoped_escape)
        auto_install: Decision | None = None
        install_answer: Decision | None = None  # the person approved this exact command already
        if separate_boundary:
            assert approval_boundary is not None
            capability = CAPABILITY + approval_boundary.key
            capability_verdict = self.rules.check(tool.name, capability)
            if capability_verdict == "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            remembered = Rule(tool.name, capability) in self.rules.allow
            if not remembered and self.mode == PermissionMode.PLAN:
                return Decision(False, "plan mode blocks software acquisition", "mode")
            if not remembered and not escape:
                auto_install = await self._auto_approves_install(tool, args, history_tail)
                if auto_install is not None and not auto_install.allow:
                    return auto_install
                remembered = auto_install is not None  # this install only: judge the next one again
            # An install that also needs a new host (Playwright fetching Chrome) is one
            # question about both, not two in a row. Seen live: "allow browser runtime
            # installs", then straight away "allow storage.googleapis.com" for the same command.
            host_too = (
                not remembered
                and scoped_escape
                and escape == sandbox_escape
                and site is not None
                and tool.risk(args, path) is None
            )
            if host_too and self.rules.check(tool.name, site) == "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            if not remembered:
                request = PermissionRequest(
                    agent=self.agent,
                    tool=tool.name,
                    args=args,
                    summary=summary or tool.name,
                    path=str(path) if path else None,
                    remember_label=approval_boundary.remember_label
                    + (f" and {tool.name} for {target_host}" if host_too else ""),
                    warning=approval_boundary.warning
                    + (f"; {escape}" if host_too else "")
                    + self._outside_warning(),
                )
                answer = await self._ask(request)
                if not answer.allow:
                    return answer
                install_answer = answer
                if answer.remember:
                    self.rules.add_allow(Rule(tool.name, capability))
                    if host_too:
                        self._remember(tool, path, trusted, site, args)
                if host_too:
                    return answer  # the one answer covered the install and the host
        target = str(path) if path else site
        unsandboxed = target is None and tool.runs_unsandboxed(args)
        match self.rules.check(tool.name, UNSANDBOXED if unsandboxed else target):
            case "deny":
                return Decision(False, "blocked by a deny rule", "rule")
            case "allow" if escape is None and (
                target is not None or self._blanket_covers(tool, args, target_host, history_tail)
            ):
                return auto_install or Decision(True, "matches an allow rule", "rule")
            case "allow" if scoped_escape:
                # The remembered rule grants only this sandbox-enforced capability. The
                # command still passes through mode and risk checks below.
                if tool.name == "Bash" and site and site.startswith(DOMAIN):
                    self.session_network_hosts.add(site.removeprefix(DOMAIN))
                escape = None

        if escape is not None:
            if self.mode == PermissionMode.PLAN:
                return Decision(False, "plan mode blocks mutating tools", "mode")
            risk = self._auto_risk(tool, args, path, trusted) if self.mode == PermissionMode.AUTO else None
            if (
                risk is None
                and target_host
                and scoped_escape
                and escape == sandbox_escape
                and approval_boundary is None
                and (approved := await self._auto_approves_host(tool, args, target_host, history_tail))
            ):
                return approved
            request = PermissionRequest(
                agent=self.agent,
                tool=tool.name,
                args=args,
                summary=summary or tool.name,
                path=str(path) if path else None,
                remember_label=(
                    self._remember_label(tool, path, trusted, target_host)
                    if scoped_escape and risk is None
                    else ""
                ),
                reason=risk or "",
                warning=(
                    (
                        f"{approval_boundary.warning}; {escape}"
                        if approval_boundary is not None and not separate_boundary
                        else escape
                    )
                    + self._network_bundle_warning(tool, target_host)
                    + self._outside_warning()
                ),
            )
            answer = await self._ask(request)
            if answer.allow and answer.remember and request.remember_label:
                self._remember(tool, path, trusted, site, args)
            return answer

        risk = self._auto_risk(tool, args, path, trusted) if self.mode == PermissionMode.AUTO else None
        # The install review saw the complete command and was told about outside content.
        # Do not ask the same classifier to judge that command a second time below.
        if risk is None and auto_install is not None:
            return auto_install
        if risk is None and install_answer is not None:  # one prompt per command, not two
            return install_answer
        if risk is None and (
            decision := await self._check_mode(tool, args, path, trusted, history_tail, target_host)
        ):
            return decision
        if risk is not None and self.classifier and (blocked := self.classifier.fast_filter(args)):
            return Decision(False, blocked.reason, "classifier")
        if (
            risk is not None
            and self.classifier is not None
            and not self.outside()  # outside content could have suggested the deletion
            and (note := tool.nameable_risk(args)) is not None
        ):
            review = await self.classifier.classify(tool.name, args, history_tail=history_tail, note=note)
            if review.decision == "allow":
                return Decision(True, review.reason or "the user asked for this", review.provider)

        if self._in_scratch(path):
            return Decision(True, "agent scratch directory", "mode")

        # Auto mode asks about every risky action case by case: a "don't ask again"
        # would stop it judging the next one (and used to switch modes).
        # So does a blanket rule while outside content could be steering a call that can
        # send data out: `_blanket_covers` wouldn't honor the answer.
        case_by_case = risk is not None or (target is None and self.outside() and tool.can_send_out(args))
        request = PermissionRequest(
            agent=self.agent,
            tool=tool.name,
            args=args,
            summary=summary or tool.name,
            path=str(path) if path else None,
            remember_label=""
            if case_by_case
            else self._remember_label(tool, path, trusted, target_host, args),
            reason=risk or "",
            warning=self._outside_warning().removeprefix("; ") if tool.can_send_out(args) else "",
        )
        decision = await self._ask(request)
        if decision.allow and decision.remember and request.remember_label:
            self._remember(tool, path, trusted, site, args)
        elif decision.allow and tool.read_only and (site or path is not None):
            # A plain "yes" to reading something covers reading it again this session.
            for name in _readers(tool) if path is not None and not site else (tool.name,):
                self.rules.add_allow(Rule(name, site or glob.escape(str(path))))
        return decision

    def _blanket_covers(
        self, tool: Tool[Any], args: dict[str, Any], target_host: str | None, history_tail: str
    ) -> bool:
        """Whether a rule for the whole tool ("don't ask again for Bash") covers this call.

        It never covers what must always ask (the tool's `risk`: deleting files, sudo,
        credentials), an unsandboxed command on the strength of an answer given while the
        sandbox was on, nor a call that could send data out while outside content is in
        the conversation: outside-content protection holds whatever was answered before. Seen in
        review: with the sandbox off, a remembered `Bash` let `curl -d @src/main.py`
        through after a WebFetch.
        """
        if tool.risk(args, None) is not None:
            return False
        if tool.runs_unsandboxed(args) and not any(
            rule.matches(tool.name, UNSANDBOXED) and rule not in self.sandboxed_grants
            for rule in self.rules.allow
        ):
            return False  # only answers given while the sandbox was on: see `sandboxed_grants`
        return not (
            self.outside()
            and tool.can_send_out(args)
            and not self._trusted_read(tool, args, target_host, history_tail)
        )

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
        self,
        tool: Tool[Any],
        args: dict[str, Any],
        path: Path | None,
        trusted: bool,
        history_tail: str,
        target_host: str | None = None,
    ) -> Decision | None:
        # A sandbox limits destinations, not what an allowed destination receives. Auto
        # lets the classifier review an outbound-capable call without seeing tool output.
        # An uncertain or blocked action goes back to the agent to recover; repeated
        # blocks eventually fall through to the person instead of looping forever.
        if (
            self.outside()
            and tool.can_send_out(args)
            and not self._trusted_read(tool, args, target_host, history_tail)
        ):
            if self.mode == PermissionMode.AUTO and self.classifier is not None:
                verdict = await self.classifier.classify(
                    tool.name,
                    args,
                    history_tail=history_tail,
                    note=(
                        "Content from outside (web pages or downloads) is in the conversation, "
                        "and this call could send data out."
                    ),
                )
                decision = self._auto_verdict(verdict, "serves the request", tool, args)
                if decision is not None:
                    return decision
            return None
        if self.mode == PermissionMode.BYPASS:
            return Decision(True, "bypassPermissions mode", "mode")
        if self.mode == PermissionMode.PLAN and not tool.read_only:
            return Decision(False, "plan mode blocks mutating tools", "mode")
        # The agent's own notes; with outside content in the conversation, see _persists_outside.
        if tool.persists and self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO):
            return Decision(True, f"{self.mode.value}: the agent's memory", "mode")
        if tool.read_only and not tool.network and (path is None or trusted):
            return Decision(True, "read-only, inside the working directory", "mode")
        # An unsandboxed command that only reads, such as `ls ../other` or `git log`, runs like
        # a read; one that touches credentials still asks (`risk`).
        if (
            self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO)
            and path is None
            and not tool.contained(args)
            and not tool.writes(args)
            and not tool.network
            and tool.risk(args, path) is None
        ):
            return Decision(True, f"{self.mode.value}: a command that only reads", "mode")
        if self.mode == PermissionMode.DEFAULT and trusted:
            return Decision(True, "default: inside the working directory", "mode")
        if self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO) and tool.contained(args):
            return Decision(True, f"{self.mode.value}: runs in the sandbox", "mode")
        # Plan mode too: researching is what it's for. Seen live: in plan mode, reading the
        # PyPI page of the package being planned for asked every time, though default and
        # auto mode read it freely. Outside content still gets the review above.
        if self.mode in (PermissionMode.DEFAULT, PermissionMode.AUTO, PermissionMode.PLAN) and _public_read(
            tool, args
        ):
            return Decision(True, f"{self.mode.value}: reading a public web page", "mode")
        if self.mode == PermissionMode.AUTO:
            return await self._auto(tool, args, path, trusted, history_tail)
        return None

    def _trusted_read(
        self, tool: Tool[Any], args: dict[str, Any], host: str | None, history_tail: str
    ) -> bool:
        """Reading a page on a host the sandbox allows, or one the user named themselves.

        The destination didn't come from outside content, so a fetch there can't carry data
        to whoever wrote it. Seen live: with a web page in context, fetching the docs the
        user asked about was held back as "reconnaissance".
        """
        if host is None or not _public_read(tool, args):
            return False
        host = host.lower()
        if host_allowed(host, (*self.allowed_hosts, *self.session_network_hosts)):
            return True
        # The whole name, so `e.com` isn't found inside `some.com`.
        named = re.compile(rf"(?<![\w.-]){re.escape(host.removeprefix('www.'))}(?![\w-])")
        return any(
            named.search(line.lower()) for line in history_tail.splitlines() if line.startswith("[user")
        )

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
        if path is not None and _GIT_INTERNALS.search(path.as_posix()):
            return "it changes version-control internals (.git)"
        if path is not None and not trusted:
            return "it changes a file outside the project"
        if (reason := tool.risk(args, path)) is not None:
            return f"it {reason}"
        return None

    async def _auto_approves_install(
        self, tool: Tool[Any], args: dict[str, Any], history_tail: str
    ) -> Decision | None:
        """Let Auto approve an aligned install or return a recoverable classifier block."""
        if self.mode != PermissionMode.AUTO or self.classifier is None:
            return None
        outside = (
            " Content from outside is in the conversation; tool output is excluded from this review."
            if self.outside()
            else ""
        )
        verdict = await self.classifier.classify(
            tool.name,
            args,
            history_tail=history_tail,
            note="This sandboxed command installs software from a package registry." + outside,
        )
        return self._auto_verdict(verdict, "software acquisition matches the user's request", tool, args)

    async def _auto_approves_host(
        self, tool: Tool[Any], args: dict[str, Any], host: str, history_tail: str
    ) -> Decision | None:
        """Let Auto open one more host for this one command.

        Nothing is remembered: the next command that needs the host is judged again. With
        outside content in the conversation the person decides, since a new destination is
        the way that content could get data sent out.
        """
        local = host == "localhost"
        # Loopback never leaves this machine, so outside content can't use it to send data
        # away; what it can reach is local services, which the review judges.
        if self.mode != PermissionMode.AUTO or self.classifier is None or (self.outside() and not local):
            return None
        bundle = self._network_bundle(tool, host)
        reach = f"{host} (and {bundle[0]}'s other download hosts)" if bundle else host
        verdict = await self.classifier.classify(
            tool.name,
            args,
            history_tail=history_tail,
            note=(
                "This sandboxed command wants loopback: to run a local server and talk to it, and "
                "to reach other services already running on this machine (Docker's API stays closed)."
                if local
                else f"This sandboxed command needs {reach}, which isn't on the sandbox's allowlist. "
                "If allowed, it can reach that for this one command only."
            ),
        )
        decision = self._auto_verdict(
            verdict, f"{host} serves the request, for this command only", tool, args
        )
        if local and decision is not None and decision.allow:
            # A project that runs a local server runs it again and again: once is enough.
            self._remember(tool, None, False, f"{DOMAIN}{host}")
            decision.reason = verdict.reason or "loopback serves the request, for this session"
        return decision

    async def _auto(
        self, tool: Tool[Any], args: dict[str, Any], path: Path | None, trusted: bool, history_tail: str
    ) -> Decision | None:
        if path is not None and trusted:
            return Decision(True, "auto: routine work inside the project", "mode")
        if self._in_scratch(path):
            return Decision(True, "agent scratch directory", "mode")
        if self.classifier is None:
            return None
        # Seen live: with the sandbox off, the review still called a command "within the
        # project sandbox". It must judge how far the command can actually reach.
        note = (
            "This runs without a sandbox: it can change any of the user's files and reach any host."
            if path is None and not tool.contained(args)
            else ""
        )
        verdict = await self.classifier.classify(tool.name, args, history_tail=history_tail, note=note)
        return self._auto_verdict(verdict, "routine action matches the user's request", tool, args)

    def _auto_verdict(
        self, verdict: Any, allow_reason: str, tool: Tool[Any], args: dict[str, Any]
    ) -> Decision | None:
        """Turn uncertainty into recovery, falling back to a prompt only after repetition.

        A denial is final and says not to repackage the call. An uncertain verdict holds
        the call back and says the agent may try a narrower one; the same call held three
        times, or twenty held in all, goes to the person instead.
        """
        key = f"{tool.name}:{json.dumps(args, sort_keys=True, default=str)}"
        if verdict.decision == "allow":
            self._auto_blocks.by_action.pop(key, None)
            return Decision(True, verdict.reason or allow_reason, verdict.provider)
        if verdict.decision == "deny":
            return Decision(
                False,
                f"auto review denied this call ({verdict.reason or 'not allowed'}). It did not run, "
                "and this was not the sandbox. Don't attempt it another way; if the task needs it, "
                "tell the user what you need and why.",
                verdict.provider,
            )
        self._auto_blocks.by_action[key] = self._auto_blocks.by_action.get(key, 0) + 1
        self._auto_blocks.total += 1
        if self._auto_blocks.by_action[key] >= 3 or self._auto_blocks.total >= 20:
            self._auto_blocks.by_action.pop(key, None)
            if self._auto_blocks.total >= 20:
                self._auto_blocks.total = 0
            return None
        if getattr(verdict, "failed", False):
            return Decision(
                False,
                f"auto review couldn't run ({verdict.reason}). It did not run; this is not a judgement "
                "that the call is unsafe. Retry it once, or ask the user.",
                verdict.provider,
            )
        concern = verdict.reason or "it couldn't tell that the user's request calls for this"
        return Decision(
            False,
            f"auto review held this call back ({concern}). It did not run, and this was not the "
            "sandbox. You may make a narrower call that addresses that concern, or ask the user; "
            "don't repeat or disguise the same call.",
            verdict.provider,
        )

    # -- "don't ask again" ------------------------------------------------------
    # In ask mode, edits inside the project switch the session to default (visible in the
    # mode indicator). Anything outside the project is remembered per directory.

    def _remember_label(
        self,
        tool: Tool[Any],
        path: Path | None,
        trusted: bool,
        host: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> str:
        if host:
            if bundle := self._network_bundle(tool, host):
                name, hosts = bundle
                if name in SERVICE_HOSTS:
                    return f"allow {name}'s {len(hosts)} download hosts this session"
                return f"allow {len(hosts)} exact destinations declared by /{name} this session"
            return f"allow {tool.name} for {host} this session"
        if path is None:
            if args is not None and tool.runs_unsandboxed(args):
                return f"don't ask again for unsandboxed {tool.name} commands this session"
            return f"don't ask again for {tool.name} this session"
        if trusted:
            return "allow all edits in this project for this session"
        if tool.name in READING_TOOLS:
            return f"allow reading in {_directory(path)}/ for this session"
        return f"allow {tool.name} in {_directory(path)}/ for this session"

    def _remember(
        self,
        tool: Tool[Any],
        path: Path | None,
        trusted: bool,
        site: str | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        if site:
            host = site.removeprefix(DOMAIN)
            bundle = self._network_bundle(tool, host)
            hosts = bundle[1] if bundle else (host,)
            for allowed_host in hosts:
                self.rules.add_allow(Rule(tool.name, f"{DOMAIN}{allowed_host}"))
                if tool.name == "Bash":
                    self.session_network_hosts.add(allowed_host)
        elif path is None:
            if args is not None and tool.runs_unsandboxed(args):
                self.rules.add_allow(Rule(tool.name, UNSANDBOXED))
            elif (rule := Rule(tool.name)) not in self.rules.allow:
                self.rules.add_allow(rule)
                self.sandboxed_grants.add(rule)
        elif trusted:
            self.set_mode(PermissionMode.DEFAULT)
        else:
            folder = glob.escape(_directory(path).as_posix())
            for name in _readers(tool):
                self.rules.add_allow(Rule(name, f"{folder}/*"))
                if name in READING_TOOLS:
                    self.rules.add_allow(Rule(name, folder))  # Grep or Glob on the folder itself

    def add_network_bundle(self, name: str, hosts: tuple[str, ...]) -> None:
        """Make a skill's finite destination set available to the session approval UI."""
        if hosts:
            self.network_bundles[name] = tuple(dict.fromkeys(hosts))

    def _network_bundle(self, tool: Tool[Any], host: str) -> tuple[str, tuple[str, ...]] | None:
        if tool.name != "Bash":
            return None
        lowered = host.lower()
        return next(
            ((name, hosts) for name, hosts in self.network_bundles.items() if lowered in hosts),
            None,
        )

    def _network_bundle_warning(self, tool: Tool[Any], host: str | None) -> str:
        bundle = self._network_bundle(tool, host) if host else None
        if bundle is None:
            return ""
        name, hosts = bundle
        who = f"{name} downloads use" if name in SERVICE_HOSTS else f"/{name} declares"
        return f"; {who} this exact destination set: {', '.join(hosts)}"


_GIT_INTERNALS = re.compile(r"(^|/)\.git(/|$)")
# Tools that only look at files: an answer about one covers them all, so reading a folder
# outside the project is one question, not one per tool (Read, then Grep, then Glob).
READING_TOOLS = ("Read", "Grep", "Glob")


def _readers(tool: Tool[Any]) -> tuple[str, ...]:
    """The tools an approval for `tool` covers: every reading tool, if it is one."""
    return READING_TOOLS if tool.name in READING_TOOLS else (tool.name,)


def _directory(path: Path) -> Path:
    return path if path.is_dir() else path.parent


def _public_read(tool: Tool[Any], args: dict[str, Any]) -> bool:
    """A read-only network call to a URL without a query string (WebFetch of a page)."""
    return tool.network and tool.read_only and not urlsplit(str(args.get("url", ""))).query
