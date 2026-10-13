"""A Session wires the harness together and holds state across prompts.

    session = Session(SessionOptions(permission_mode="ask"))
    async for event in session.run("add a --verbose flag to cli.py"):
        ...
    async for event in session.run("now add a test for it"):   # same history & permissions
        ...
    session.close()

`query()` is the one-shot shortcut for scripts.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import re
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

from tinyharness import trust
from tinyharness.agent import world
from tinyharness.agent.compaction import SUMMARY_MARK, Compactor
from tinyharness.agent.history import History, Image, Message
from tinyharness.agent.loop import LoopDeps, record_usage, run_agent_loop, summary_notes
from tinyharness.agent.prompts import build_system_prompt, load_instructions
from tinyharness.agent.sessionlog import (
    FileSessionStore,
    Replayed,
    SessionInfo,
    SessionLog,
    SessionStore,
    replay,
)
from tinyharness.agent.transcript import Transcript, trace_id
from tinyharness.config import Config, dotenv_paths, dotenv_values, load_config, resolve_credentials
from tinyharness.events import Compacted, Done, Event, SkillLoaded, ToolResult, ToolUse
from tinyharness.ext import Command as ExtCommand
from tinyharness.ext import Extension, LoadError, NeedsConsent, RunResult, load_extensions
from tinyharness.hooks import Hook, HookRunner, load_hooks
from tinyharness.llm import identity
from tinyharness.llm.calibration import CalibrationStore
from tinyharness.llm.client import LLMClient, Ping
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.providers import DEFAULT_CONTEXT, Provider
from tinyharness.llm.textcalls import recover_calls
from tinyharness.llm.tokens import count_tokens, estimate_tokens, model_counter
from tinyharness.llm.types import Effort
from tinyharness.llm.usage import Usage, UsageMeter
from tinyharness.paths import default_config_path, home_dir, move_legacy_state, project_dir
from tinyharness.permissions import (
    Decision,
    PermissionCallback,
    PermissionGate,
    PermissionMode,
    PermissionRequest,
    PermissionRules,
)
from tinyharness.permissions.classifier import AutoModeClassifier
from tinyharness.permissions.rules import DOMAIN, Rule
from tinyharness.plugins import load_plugins
from tinyharness.prompts import load_prompt
from tinyharness.sandbox import Sandbox, SandboxPolicy, detect_sandbox
from tinyharness.sandbox.proxy import AllowlistProxy
from tinyharness.skills import Skill, SkillRegistry, build_skill_prompt, load_skills, skills_section
from tinyharness.subagents import AgentSpec, load_agents
from tinyharness.subagents.runner import run_subagent
from tinyharness.tools.base import Tool, ToolContext, ToolError, file_version
from tinyharness.tools.bash import BashInput, BashTool
from tinyharness.tools.registry import default_registry
from tinyharness.trust import TrustStore
from tinyharness.workspace import Scratch, Workspace

MAX_SUGGESTION = 120  # chars: a longer "next message" is the model explaining, not a suggestion


def _one_at_a_time(callback: PermissionCallback | None) -> PermissionCallback | None:
    """Parallel subagents may all need permission at once: ask one question at a time."""
    if callback is None:
        return None
    lock = asyncio.Lock()

    async def ask(request: PermissionRequest) -> Decision:
        async with lock:
            return await callback(request)

    return ask


log = logging.getLogger(__name__)


def _capped(spec: AgentSpec, root: Path) -> AgentSpec:
    """An agent that came with the project (its files or plugins) can't grant itself
    bypassPermissions: that's for agents you define yourself (~/.tinyharness, SDK)."""
    project = project_dir(root).resolve()
    from_project = spec.source is not None and spec.source.resolve().is_relative_to(project)
    if not from_project or project == home_dir().expanduser().resolve():
        return spec
    if spec.permission_mode != PermissionMode.BYPASS:
        return spec
    log.warning("agent %r from this project asks for bypassPermissions; using default", spec.name)
    return replace(spec, permission_mode=PermissionMode.DEFAULT)


def _new_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


class MissingAPIKeyError(RuntimeError):
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        super().__init__(
            f"No API key for {provider.label}. Add one with /key {provider.name} (or run "
            f"`tinyharness --setup`), or set {provider.api_key_env} in your shell."
        )


class SandboxRequiredError(RuntimeError):
    """Raised before agent work when required containment is unavailable."""


@dataclass(frozen=True)
class EvaluationContext:
    """Provide a read-only snapshot to a post-turn evaluator.

    Attributes
    ----------
    session_id : str
        Identifier of the session being evaluated.
    turn_id : str
        Correlation identifier for the completed turn.
    prompt : str
        User prompt that started the turn.
    reason : str
        Normalized reason the turn ended.
    working_dir : pathlib.Path
        Session working directory.
    messages : tuple[Message, ...]
        Copy of the normalized conversation history.
    duration_ms : float
        Turn duration in milliseconds.
    """

    session_id: str
    turn_id: str
    prompt: str
    reason: str
    working_dir: Path
    messages: tuple[Message, ...]
    duration_ms: float


Evaluator = Callable[[EvaluationContext], Any | Awaitable[Any]]


async def _invoke_evaluator(evaluator: Evaluator, context: EvaluationContext) -> Any:
    """Run synchronous observers off the event loop; await asynchronous ones normally."""
    if inspect.iscoroutinefunction(evaluator):
        return await evaluator(context)
    result = await asyncio.to_thread(evaluator, context)
    return await result if inspect.isawaitable(result) else result


@dataclass
class SessionOptions:
    """Configure one session.

    Values set to ``None`` inherit from the loaded :class:`Config` where applicable.
    """

    config_path: Path | None = None
    working_dir: Path | None = None
    provider: str | None = None
    model: str | None = None
    permission_mode: PermissionMode | str | None = None
    effort: Effort | None = None
    api_key: str | None = None
    use_last_model: bool = False  # start with the model last picked in the CLI (`/model`)
    todo_list: bool | None = None  # offer the TodoWrite checklist; None: the config's `todo_list`
    resume: str | None = None  # continue a saved conversation: its id, or "latest"
    # Load the project's own hooks and plugins (code that runs outside the sandbox)?
    # None: only if you've trusted this folder (the CLI asks); see tinyharness/trust.py.
    trust_project: bool | None = None
    store: SessionStore | None = None  # where conversations are saved (default: files in the project)

    permission_callback: PermissionCallback | None = None
    on_event: Callable[[Event], None] | None = None  # progress of subagents/teammates as it happens
    hooks: list[Hook] = field(default_factory=list)
    evaluators: list[Evaluator] = field(default_factory=list)
    evaluator_timeout: float = 30.0
    tools: list[Tool[Any]] = field(default_factory=list)  # your own tools; a built-in's name replaces it
    # Extensions given in code: modules with setup(th), or setup functions (tinyharness/ext.py).
    extensions: list[Any] = field(default_factory=list)
    allowed_tools: list[str] | None = None
    agents: list[AgentSpec] = field(default_factory=list)  # more subagents; same name replaces one
    extra_system: str = ""
    max_steps: int = 200

    enable_skills: bool = True
    enable_subagents: bool = True  # load agents from files (built-in, ~/.tinyharness, project)
    enable_plugins: bool = True
    expose_skill_tool: bool | None = None  # list skills in the prompt; None: the config's `skill_tool`

    def resolve_config(self) -> Config:
        cfg = load_config(self.config_path, use_last_model=self.use_last_model)
        if self.working_dir is not None:
            cfg.working_dir = Path(self.working_dir)
            cfg.working_dir.mkdir(parents=True, exist_ok=True)
        if self.provider:
            cfg.use_provider(self.provider)
        if self.model:
            cfg.select(self.model)
        if self.permission_mode:
            cfg.permission_mode = PermissionMode.parse(self.permission_mode)
        if self.effort is not None:
            cfg.effort = self.effort
        if self.api_key:
            cfg.set_api_key(self.api_key)
        if self.todo_list is not None:
            cfg.todo_list = self.todo_list
        if self.expose_skill_tool is not None:
            cfg.skill_tool = self.expose_skill_tool
        return cfg


class Session:
    """Coordinate models, tools, permissions, and conversation state.

    Parameters
    ----------
    options : SessionOptions, optional
        Per-session configuration and integrations.
    config : Config, optional
        Preloaded configuration. When omitted, the session resolves it from ``options``.

    Raises
    ------
    ValueError
        If evaluator timeout or other session settings are invalid.
    SandboxRequiredError
        If containment is required but unavailable.
    """

    def __init__(self, options: SessionOptions | None = None, config: Config | None = None) -> None:
        self.options = options or SessionOptions()
        if self.options.evaluator_timeout <= 0:
            raise ValueError("evaluator_timeout must be greater than zero")
        self.config = config or self.options.resolve_config()
        root = self.config.working_dir
        move_legacy_state(root)
        self.store: SessionStore = self.options.store or FileSessionStore(root)
        self.id, saved = self._load(self.options.resume) if self.options.resume else (_new_id(), None)
        if saved is not None and saved.model and not (self.options.model or self.options.provider):
            self._restore_model(saved.model)

        self.workspace = Workspace(root)
        self.scratch = Scratch(root, self.id, max_age_hours=self.config.scratch_max_age_hours)
        self.transcript = self.store.writer(self.id)
        if isinstance(self.transcript, Transcript):
            self.transcript.set_secrets(self.config.credential_values())
        self.log = SessionLog(self.transcript)
        self.history = History()
        self.ctx = ToolContext(
            workspace=self.workspace, config=self.config, scratch=self.scratch, notify=self.options.on_event
        )

        # Extensions: skills, agents, hooks (plugins contribute to all three).
        self.skills = SkillRegistry(load_skills(root) if self.options.enable_skills else [])
        self.agents: dict[str, AgentSpec] = load_agents(root) if self.options.enable_subagents else {}
        self.project_code = trust.scan(root)
        trusted = self.options.trust_project
        self.project_trusted = TrustStore().trusts(self.project_code) if trusted is None else trusted
        if self.project_code.needs_trust and not self.project_trusted:
            log.warning(
                "not loading this project's hooks and plugins: %s",
                self.project_code.too_large or "the folder isn't trusted",
            )
        hooks = load_hooks(root, project=self.project_trusted)
        if self.options.enable_plugins:
            for plugin in load_plugins(root, project=self.project_trusted):
                for skill in plugin.skills:
                    self.skills.add(skill)
                self.agents.update({a.name: a for a in plugin.agents})
                hooks.extend(plugin.hooks)
        self.agents = {name: _capped(spec, root) for name, spec in self.agents.items()}
        self.agents.update({spec.name: spec for spec in self.options.agents})  # yours, from code: as given
        self.hooks = HookRunner([*hooks, *self.options.hooks], cwd=root)

        # Skills are listed in the prompt and loaded with Read (subagents get the list too).
        self.ctx.skills_prompt = skills_section(self.skills.list()) if self.config.skill_tool else ""
        # Skills may ship reference files; let tools read them without prompting.
        for skill in self.skills.list():
            if skill.directory is not None:
                self.workspace.add_read_root(skill.directory)

        self.sandbox: Sandbox | None = detect_sandbox() if self.config.sandbox.enabled else None
        if self.config.sandbox.required and self.sandbox is None:
            self.transcript.write("sandbox_unavailable", required=True)
            self.transcript.close()
            self.scratch.cleanup()
            raise SandboxRequiredError(
                "sandbox.required is true, but no supported sandbox backend is available; "
                "install the platform sandbox or set sandbox.required = false"
            )
        self.sandbox_policy = SandboxPolicy.for_project(
            root,
            scratch=self.scratch.dir,
            cache_dir=home_dir() / "sandbox-cache",
            extra_write=self.config.sandbox.allow_write,
            allow_localhost=self.config.sandbox.allow_localhost,
        )
        hosts = self.config.sandbox.allow_hosts
        self.proxy = AllowlistProxy(tuple(hosts)) if self.sandbox is not None and hosts else None
        self.extensions: list[Extension] = []
        self.extension_errors: list[LoadError] = []
        enabled = list(self.config.extensions.enabled)
        if self.config.todo_list and "todo" not in enabled:  # the older setting for the same thing
            enabled.append("todo")
        self._load_extensions(enabled)

        # One gate for the whole session, so "don't ask again" answers and mode
        # changes (e.g. Shift+Tab in the CLI) carry over from prompt to prompt.
        cfg = self.config
        self.gate = PermissionGate(
            workspace=self.workspace,
            mode=cfg.permission_mode,
            rules=PermissionRules.parse(cfg.permissions.allow, cfg.permissions.deny, root),
            hooks=self.hooks,
            callback=_one_at_a_time(self.options.permission_callback),
            scratch=self.scratch,
            # Files that would run code outside the sandbox: writing them always asks.
            protected=(
                *self.sandbox_policy.protected,
                home_dir().expanduser().resolve(),
                default_config_path().parent.expanduser().resolve(),
            ),
        )

        self.gate.outside = lambda: self.ctx.outside
        self._needs_rules: list[Rule] = []
        self._extension_proxies: dict[str, AllowlistProxy] = {}
        self._apply_needs()

        self.models = ModelDirectory(home_dir() / "cache" / "models")
        self._calibration = CalibrationStore(home_dir() / "cache" / "tool-calls.json")
        self._connect_llm()
        self._started = False
        self._steering: list[str] = []  # messages for the agent while a turn runs (see steer)
        # Subagents run side by side; their calls that change things take turns on this.
        self._agent_writes = asyncio.Lock()
        self.resume_report: world.ResumeReport | None = None  # what changed while away, when resumed
        self._resume_note = ""
        if saved is not None:
            self._restore(saved)
        self.transcript.write(
            "session_resumed" if saved else "session_start",
            model=cfg.model,
            mode=self.mode,
            working_dir=str(root),
        )
        self._log_model()
        self._log_world()

    # ------------------------------------------------------------------ saving & resuming

    def _load(self, which: str) -> tuple[str, Replayed]:
        session_id = which
        if which == "latest":
            found = self.store.sessions(limit=1)
            if not found:
                raise ValueError(f"no saved conversation to resume in {self.config.working_dir}")
            session_id = found[0].id
        try:
            saved = replay(self.store.load(session_id))
        except KeyError:
            raise ValueError(f"no saved conversation {session_id!r} in {self.config.working_dir}") from None
        if not saved.history.messages:
            raise ValueError(f"conversation {session_id!r} has nothing to resume")
        return session_id, saved

    def _restore_model(self, spec: str) -> None:
        """Resume on the model the conversation was using, if there's still a key for it."""
        provider = spec.partition(":")[0]
        if provider in self.config.providers and self.config.has_key(provider):
            with contextlib.suppress(ValueError):  # a model id the provider no longer accepts
                self.config.select(spec)

    def _restore(self, saved: Replayed) -> None:
        history = saved.history
        self.log.seed(history, saved.todos)  # what's on disk; the changes below get logged
        answered = {m.tool_call_id for m in history.messages if m.role == "tool"}
        unfinished = sum(1 for m in history.messages for tc in m.tool_calls if tc.id not in answered)
        history.repair(world.UNFINISHED)
        report = world.check(
            history.messages,
            saved.every_message,
            root=self.workspace.root,
            reported=saved.reported,
            git_before=saved.world.get("git"),
            instructions_before=saved.world.get("agent_md"),
            instructions_now=world.text_hash(load_instructions(self.workspace.root)),
        )
        report.unfinished = unfinished
        # A fresh system prompt: the model, scratch dir or AGENTS.md may have changed.
        first = history.messages[0]
        if first.role == "system" and not (first.content or "").startswith(SUMMARY_MARK):
            history.messages[0] = Message(role="system", content=self._system_prompt())
        self.history = history
        self.ctx.todos = list(saved.todos)
        self.ctx.read_paths = set(report.current_reads)
        # Those files are as the agent last saw them, so a change from now on (the user
        # editing one) must make Edit and Write ask for a fresh Read, as in a live session.
        self.ctx.seen_versions = {p: v for p in report.current_reads if (v := file_version(Path(p)))}
        self.resume_report = report
        self._resume_note = report.note()
        self._sync()
        if report.now:
            self.transcript.write("files", files=report.now)

    def _sync(self) -> None:
        self.log.sync(self.history, self.ctx.todos)

    def _log_model(self) -> None:
        self.transcript.write("model", spec=f"{self.config.provider}:{self.config.model}")

    def _log_world(self) -> None:
        root = self.workspace.root
        self.transcript.write(
            "world", git=world.git_state(root), agent_md=world.text_hash(load_instructions(root))
        )

    # ------------------------------------------------------------------ setup

    def _connect_llm(self) -> None:
        """(Re)build everything that depends on the provider/model."""
        cfg = self.config
        api_key = cfg.resolve_api_key()
        if not api_key and cfg.provider_info.needs_key:
            raise MissingAPIKeyError(cfg.provider_info)
        self.llm = LLMClient(
            cfg.provider_info,
            api_key,
            cfg.model,
            models=self.models,
            effort=cfg.effort,
            timeout=cfg.request_timeout,
            retries=cfg.request_retries,
            tool_calls=cfg.tool_calls,
            calibration=self._calibration,
        )
        self.llm.show_reasoning = cfg.show_thinking
        self.gate.classifier = AutoModeClassifier(self.llm, model=cfg.role_model("safety_classifier"))
        self.compactor = Compactor(
            self.llm,
            context_window=lambda: self.context_window,
            max_output=lambda: info.max_output if (info := self.llm.model_info()) else None,
            counter=lambda: model_counter((info := self.llm.model_info()) and info.tokenizer),
            reserved=cfg.compaction_reserved,
            preserve_recent_tokens=cfg.compaction_preserve_recent_tokens,
            threshold=cfg.auto_compact_threshold,
            model=cfg.role_model("compaction"),
        )

    def _identify(self) -> identity.CallIdentity:
        """This session's lead identity for the model calls that follow (llm/identity.py)."""
        who = identity.CURRENT.get()
        if who is None or who.session != self.id:
            who = identity.for_session(self.id)
            identity.CURRENT.set(who)
        return who

    async def _spawn_subagent(
        self,
        spec: AgentSpec,
        task: str,
        ctx: ToolContext,
        *,
        label: str = "",
        extra_tools: Iterable[Tool[Any]] = (),
        inbox: Callable[[], list[str]] | None = None,
    ) -> str:
        # The session's rules and mode, live (switching mode reaches running subagents);
        # the agent's own permission_mode can only make it stricter.
        gate = self.gate.child(spec.name, spec.permission_mode)
        try:
            llm, model = self.llm_for(spec.model)
        except (MissingAPIKeyError, ValueError) as e:
            raise ToolError(f"can't start the {spec.name} agent: {e}") from None
        parent = identity.CURRENT.get() or self._identify()
        token = identity.CURRENT.set(identity.child(parent, spec.name))  # this task only
        try:
            return await run_subagent(
                spec,
                task,
                ctx=ctx,
                llm=llm,
                model=model,
                tools=self.tools,
                gate=gate,
                label=label,
                extra_tools=extra_tools,
                inbox=inbox,
                write_lock=self._agent_writes,
                **({"max_steps": spec.max_steps} if spec.max_steps else {}),
            )
        finally:
            identity.CURRENT.reset(token)

    def llm_for(self, spec: str | None) -> tuple[LLMClient, str | None]:
        """The client and model id for a subagent's `model` setting:

        None, "inherit"      the session's current model
        "small"              the session provider's small model (the one auto mode uses)
        "provider:model"     that model, from any provider you have a key for
        "provider:"          that provider's default model
        "model-id"           that model, from the session's provider
        """
        if not spec or spec == "inherit":
            return self.llm, None
        if spec == "small":
            return self.llm, self.config.small_model
        provider, model = self.config.parse_model(spec)
        if provider == self.config.provider:
            return self.llm, model or None
        self._require_key(provider)
        info = self.config.providers[provider]
        client = LLMClient(
            info,
            info.api_key,
            model or info.default_model,
            models=self.models,
            effort=self.config.effort,
            timeout=self.config.request_timeout,
            retries=self.config.request_retries,
            transport=self.llm.transport,
            tool_calls=self.config.tool_calls,
            calibration=self._calibration,
        )
        return client, None

    # ------------------------------------------------------------------ settings

    def reload_keys(self) -> None:
        """Resolve credentials again after a source changes."""
        dotenv = dotenv_values(*dotenv_paths(self.config.working_dir))
        resolve_credentials(self.config.providers, dotenv)
        if isinstance(self.transcript, Transcript):
            self.transcript.set_secrets(self.config.credential_values())

    async def check_model(self, spec: str) -> Ping:
        """Send a tiny test request to a model before switching to it ("provider:" = its default).
        Raises ValueError / MissingAPIKeyError like set_model, without changing anything."""
        provider, model = self.config.parse_model(spec)
        self._require_key(provider)
        info = self.config.providers[provider]
        return await self.client_for(provider).ping(model or info.default_model)

    def set_model(self, spec: str) -> None:
        """Switch model; `provider:model` switches provider too. The conversation is kept."""
        provider, _ = self.config.parse_model(spec)
        self._switch(lambda: self.config.select(spec), provider)

    def set_provider(self, name: str) -> None:
        """Switch provider, using its default model. The conversation is kept."""
        self._switch(lambda: self.config.use_provider(name), name)

    def _switch(self, apply: Callable[[], None], provider: str) -> None:
        self._require_key(provider)
        apply()
        self._connect_llm()
        # The system prompt names the model; keep it true. (A new model starts a new
        # prompt cache anyway, so this costs nothing extra.)
        if self.history.messages and self.history.messages[0].role == "system":
            self.history.messages[0].content = self._system_prompt()
        self._log_model()
        self._sync()

    def _system_prompt(self) -> str:
        # Extensions' sections come after the core prompt, in the order they were turned
        # on: the same every time, so the cached prefix holds (docs/extension-api.md).
        sections = [self.ctx.skills_prompt] if self.ctx.skills_prompt else []
        sections += [f"# {name}\n\n{text}" for ext in self.extensions for name, text in ext.sections]
        return build_system_prompt(
            self.workspace.root,
            self.scratch.dir,
            model_name=self.model_label,
            model_id=self.config.model,
            provider=self.config.provider_info.label,
            extra="\n\n".join(filter(None, [*sections, self.options.extra_system])),
        )

    # ------------------------------------------------------------------ extensions

    def _load_extensions(self, names: list[str]) -> None:
        """Load `names` and build the tool set: core tools, your SDK tools, then each
        extension's, in order. Also used to switch extensions mid-session."""
        ext_hooks = {id(h) for ext in self.extensions for h in ext.hooks}
        self.hooks.hooks = [h for h in self.hooks.hooks if id(h) not in ext_hooks]
        self.extensions, self.extension_errors = load_extensions(
            names,
            self,
            self.config.extensions.settings(),
            project=self.project_trusted,
            given=self.options.extensions,
        )
        self.tools = default_registry()
        if shutil.which("bash") or shutil.which("sh"):
            self.tools.register(BashTool(self.sandbox, self.sandbox_policy, proxy=self.proxy))
        for tool in self.options.tools:  # before the Agent tool, so subagents can use them too
            self.tools.register(tool)
        for ext in self.extensions:
            for name in ext.replaces:
                self.tools.remove(name)
            for tool in ext.tools:
                self.tools.register(tool)
            for hook in ext.hooks:
                self.hooks.add(hook)

    def set_extensions(self, names: list[str]) -> None:
        """Switch to these extensions now. The tools and system prompt change, so the next
        request rebuilds the provider's prompt cache once."""
        self.config.extensions.enabled = list(names)
        self._load_extensions(self.config.extensions.enabled)
        self._apply_needs()
        if self.history.messages and self.history.messages[0].role == "system":
            first = self.history.messages[0].content or ""
            if not first.startswith(SUMMARY_MARK):
                self.history.messages[0] = Message(role="system", content=self._system_prompt())
        self.transcript.write("extensions", enabled=[e.name for e in self.extensions])

    def _apply_needs(self) -> None:
        """Let each extension's own tools do what its approved NEEDS say without asking:
        reach its hosts, write in its folders. Other tools get nothing from them."""
        for rule in self._needs_rules:
            self.gate.rules.remove_allow(rule)
        self._needs_rules = [
            Rule(tool.name, pattern)
            for ext in self.extensions
            if ext.granted
            for tool in ext.tools
            for pattern in [*(f"{DOMAIN}{h.lower()}" for h in ext.hosts), *(f"{w}/*" for w in ext.writes)]
        ]
        for rule in self._needs_rules:
            self.gate.rules.add_allow(rule)

    def grant_needs(self, name: str) -> None:
        """Approve what an extension's NEEDS ask for (remembered with its code's fingerprint)."""
        ext = next((e for e in self.extensions if e.name == name), None)
        if ext is None:
            raise ValueError(f"no extension {name!r} is loaded")
        NeedsConsent().grant(ext)
        self._apply_needs()

    async def run_for_extension(self, ext: Extension, command: str, *, timeout: float = 120) -> RunResult:
        """`th.run`: a command in the sandbox, with the extension's approved needs added to
        what the sandbox allows (its hosts through its own proxy, its folders writable)."""
        policy, proxy = self.sandbox_policy, self.proxy
        if ext.granted and ext.writes:
            policy = replace(policy, write_roots=(*policy.write_roots, *ext.writes))
        if ext.granted and ext.hosts and self.sandbox is not None:
            if ext.name not in self._extension_proxies:
                base = tuple(self.config.sandbox.allow_hosts)
                self._extension_proxies[ext.name] = AllowlistProxy((*base, *ext.hosts))
            proxy = self._extension_proxies[ext.name]
        bash = BashTool(self.sandbox, policy, proxy=proxy)
        run = await bash.execute(BashInput(command=command, timeout=max(1, min(600, int(timeout)))), self.ctx)
        return RunResult(run.code, run.output)

    def extension_command(self, name: str) -> ExtCommand | None:
        return next((c for ext in self.extensions for c in ext.commands if c.name == name), None)

    def prefix_tokens(self) -> int:
        """Tokens sent before the conversation on every request: the system prompt and the
        tool definitions. The provider caches them, as long as they stay the same."""
        tools = [t.spec().__dict__ for t in self.tools.available(self.mode)]
        return count_tokens(self._system_prompt() + json.dumps(tools, default=str))

    def extension_tokens(self, ext: Extension) -> int:
        """Tokens this extension adds to every request: its tool schemas and prompt sections."""
        text = json.dumps([t.spec().__dict__ for t in ext.tools], default=str)
        return count_tokens(text + "".join(t for _, t in ext.sections))

    async def run_agent(
        self,
        agent: str,
        task: str,
        *,
        model: str | None = None,
        label: str = "",
        extra_tools: Iterable[Tool[Any]] = (),
        inbox: Callable[[], list[str]] | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        """A subagent run to completion under this session's gate and sandbox: its answer.
        The primitive the agents extension builds Agent and Team on."""
        spec = self.agents.get(agent)
        if spec is None:
            raise ToolError(f"unknown agent {agent!r}; available: {', '.join(self.agents) or 'none'}")
        if model:
            spec = replace(spec, model=model)
        return await self._spawn_subagent(
            spec, task, ctx or self.ctx, label=label, extra_tools=extra_tools, inbox=inbox
        )

    def _require_key(self, provider: str) -> None:
        if provider not in self.config.providers:
            raise ValueError(
                f"unknown provider {provider!r}; choose one of: {', '.join(self.config.providers)}"
            )
        if not self.config.has_key(provider):
            self.reload_keys()
        if not self.config.has_key(provider):
            raise MissingAPIKeyError(self.config.providers[provider])

    @property
    def usage(self) -> UsageMeter:
        """Tokens and cost so far, subagents included: `.total` (a Usage), `.cost` (None if unknown)."""
        return self.ctx.usage

    @property
    def mode(self) -> PermissionMode:
        return self.gate.mode

    def set_mode(self, mode: PermissionMode | str) -> None:
        """For the session and its subagents, including ones already running."""
        self.gate.set_mode(PermissionMode.parse(mode))
        self.transcript.write("mode", mode=self.gate.mode.value)

    def set_show_thinking(self, show: bool) -> None:
        """Ask for the model's reasoning where a provider only sends it when asked (OpenAI's
        summaries); others send it anyway. Showing it is up to whoever renders the events."""
        self.config.show_thinking = show
        self.llm.show_reasoning = show

    def set_effort(self, effort: Effort | None) -> None:
        """Reasoning effort for every model: none | low | medium | high | max, or None for the
        model's default (medium for models whose reasoning is off unless asked for)."""
        self.config.effort = effort
        self.llm.effort = effort

    @property
    def model_label(self) -> str:
        """The model's display name from the provider's list, else its id."""
        info = self.llm.model_info()
        return info.label if info and info.label else self.config.model

    @property
    def context_window(self) -> int:
        """The configured window, else what the provider reports for the model, else a default."""
        info = self.llm.model_info()
        return self.config.context_window_tokens or (
            info.context if info and info.context else DEFAULT_CONTEXT
        )

    def client_for(self, provider: str) -> LLMClient:
        """A client for another provider (e.g. to list its models), sharing the model cache."""
        info = self.config.providers[provider]
        return LLMClient(info, info.api_key, info.default_model, models=self.models)

    async def compact(self) -> bool:
        """Summarize the conversation now, keeping only the latest exchange as is (the
        automatic version runs when the context nears full). False if there was nothing to do,
        including when a short conversation's summary would be longer than the conversation."""
        self.compactor.budget_cap(self.history)
        self.compactor.microcompact(self.history)
        before = list(self.history.messages)
        if not await self.compactor.summarize(self.history, keep_tail=1, notes=summary_notes(self.ctx)):
            return False
        if estimate_tokens(self.history.messages) >= estimate_tokens(before):
            self.history.messages = before  # the summary didn't save anything: keep the original
            return False
        self.transcript.write("compacted", kind="manual")
        self._sync()
        return True

    def clear(self) -> None:
        """Start a new conversation: the old one stays saved as it was (`tinyharness -r`
        picks it), and this one gets its own id and log.
        Permissions and scratch space are kept, and so are the token and cost totals: they're
        what the session spent (see UsageMeter.since_clear)."""
        had_conversation = any(m.role != "system" for m in self.history.messages)
        self.history.reset()
        self.ctx.outside.clear()  # a new conversation has no outside content in it yet
        self.ctx.usage.mark_clear()
        self.resume_report, self._resume_note = None, ""
        if had_conversation:
            previous = self.id
            self.transcript.close()
            self.id = _new_id()
            self.transcript = self.store.writer(self.id)
            if isinstance(self.transcript, Transcript):
                self.transcript.set_secrets(self.config.credential_values())
            self.log = SessionLog(self.transcript)
            self.transcript.write(
                "session_start",
                model=self.config.model,
                mode=self.mode,
                working_dir=str(self.workspace.root),
                cleared=previous,
            )
            self._log_model()
            self._log_world()
        self._sync()

    # ------------------------------------------------------------------ running

    async def run(self, prompt: str, images: Sequence[Image] = ()) -> AsyncGenerator[Event, None]:
        """Send a message and stream session events.

        A message with an installed skill's `/name` anywhere in it loads that skill with it,
        unless it's already in the conversation: asked for that way, it shouldn't depend on
        the model choosing to read the SKILL.md. A bare name doesn't count ("don't use the
        some-skill" names it too); the model sees those in the skills list.

        Parameters
        ----------
        prompt : str
            Message to send to the agent.
        images : Sequence[Image], optional
            Images supplied with the message.

        Yields
        ------
        Event
            Model text, tool activity, usage, and the final turn outcome.
        """
        named = self.skills_named(prompt)
        if named:
            prompt = "\n\n".join([*(build_skill_prompt(s) for s in named), f"User message: {prompt}"])
        for skill in named:
            yield SkillLoaded(skill.name, count_tokens(build_skill_prompt(skill)), "named")
        self._identify()
        async for event in self._turn(prompt, images):
            yield event

    def skills_named(self, prompt: str) -> list[Skill]:
        """Installed skills `prompt` asks for as `/name` that the conversation doesn't hold yet
        (not part of a path: `~/.tinyharness/skills/x/SKILL.md` doesn't ask for x)."""
        return [
            skill
            for skill in self.skills.list()
            if re.search(rf"(?<![\w/.~-])/{re.escape(skill.name)}(?![\w/-])", prompt)
            and not self._skill_in_context(skill)
        ]

    def _skill_in_context(self, skill: Skill) -> bool:
        for m in self.history.messages:
            if m.role == "user" and f"# Skill: {skill.name}\n" in (m.content or ""):
                return True
            if skill.source is not None and any(
                c.name == "Read" and str(skill.source) in c.arguments for c in m.tool_calls
            ):
                return True
        return False

    def _skill_read(self, call: ToolUse) -> Skill | None:
        """The skill whose SKILL.md this Read call reads, if any."""
        path = call.args.get("path")
        if call.name != "Read" or not isinstance(path, str) or not path.endswith("SKILL.md"):
            return None
        target = (self.workspace.root / Path(path).expanduser()).resolve()
        return next((s for s in self.skills.list() if s.source and s.source.resolve() == target), None)

    async def _turn(self, prompt: str, images: Sequence[Image] = ()) -> AsyncGenerator[Event, None]:
        turn_id = trace_id("turn")
        turn_started = time.monotonic()
        if not self._started:
            await self.hooks.fire(
                "SessionStart", {"session_id": self.id, "resumed": self.resume_report is not None}
            )
            self._started = True
        if not self.history.messages:
            self.history.append(Message(role="system", content=self._system_prompt()))
        self.ctx.request = prompt
        if self._resume_note:  # tell the model once what changed while the session was away
            prompt = f"<session-resumed>\n{self._resume_note}\n</session-resumed>\n\n{prompt}"
            self._resume_note = ""

        deps = LoopDeps(
            llm=self.llm,
            tools=self.tools,
            gate=self.gate,
            compactor=self.compactor,
            transcript=self.transcript,
            ctx=self.ctx,
            hooks=self.hooks,
            allowed_tools=set(self.options.allowed_tools) if self.options.allowed_tools else None,
            max_steps=self.options.max_steps,
            inbox=self.take_steering,
            turn_id=turn_id,
        )
        skill_reads: dict[str, Skill] = {}  # Read calls of a SKILL.md, by call id
        try:
            async for event in run_agent_loop(prompt, self.history, deps, images):
                if isinstance(event, (ToolUse, ToolResult, Compacted, Done)):
                    self._sync()  # saved as it goes, so a crash loses at most one step
                if isinstance(event, Done):
                    duration_ms = round((time.monotonic() - turn_started) * 1000, 3)
                    await self._evaluate_turn(prompt, turn_id, event.reason or "done", duration_ms)
                    self.transcript.write(
                        "turn_end",
                        turn=turn_id,
                        reason=event.reason or "done",
                        duration_ms=duration_ms,
                    )
                if isinstance(event, ToolUse) and (skill := self._skill_read(event)):
                    skill_reads[event.id] = skill
                yield event
                read = skill_reads.pop(event.id, None) if isinstance(event, ToolResult) else None
                if read is not None and isinstance(event, ToolResult) and not event.is_error:
                    yield SkillLoaded(read.name, count_tokens(event.output), "read")
        except (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit):
            self.transcript.write(
                "turn_end",
                turn=turn_id,
                reason="interrupted",
                duration_ms=round((time.monotonic() - turn_started) * 1000, 3),
            )
            raise
        except Exception as e:  # recorded, so a failed turn can be told apart from a stopped one
            self.transcript.write(
                "turn_end",
                turn=turn_id,
                reason="error",
                error=f"{type(e).__name__}: {e}"[:1000],
                duration_ms=round((time.monotonic() - turn_started) * 1000, 3),
            )
            raise
        finally:
            # If the turn was interrupted mid tool call, keep the history valid.
            self.history.repair()
            self._sync()
            self._log_world()

    async def _evaluate_turn(self, prompt: str, turn_id: str, reason: str, duration_ms: float) -> None:
        """Run observers after the trajectory is complete; failures never change its outcome."""
        if not self.options.evaluators:
            return
        context = EvaluationContext(
            session_id=self.id,
            turn_id=turn_id,
            prompt=prompt,
            reason=reason,
            working_dir=self.workspace.root,
            messages=tuple(copy.deepcopy(self.history.messages)),
            duration_ms=duration_ms,
        )
        for evaluator in self.options.evaluators:
            name = getattr(evaluator, "__name__", type(evaluator).__name__)
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    _invoke_evaluator(evaluator, context), timeout=self.options.evaluator_timeout
                )
                if isinstance(result, Mapping):
                    data = dict(result)
                elif is_dataclass(result) and not isinstance(result, type):
                    data = asdict(result)
                else:
                    data = {"result": result}
                self.transcript.write(
                    "evaluation",
                    turn=turn_id,
                    evaluator=name,
                    status="ok",
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    result=data,
                )
            except TimeoutError:
                self.transcript.write(
                    "evaluation",
                    turn=turn_id,
                    evaluator=name,
                    status="timeout",
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    timeout_seconds=self.options.evaluator_timeout,
                )
            except Exception as e:
                self.transcript.write(
                    "evaluation",
                    turn=turn_id,
                    evaluator=name,
                    status="error",
                    duration_ms=round((time.monotonic() - started) * 1000, 3),
                    error=f"{type(e).__name__}: {e}"[:1000],
                )

    def steer(self, text: str) -> None:
        """Queue a user message for a running turn.

        Parameters
        ----------
        text : str
            Message delivered before the agent's next step. It remains pending if the turn
            ends before delivery.
        """
        self._steering.append(text)

    def record_suggestion(self, action: str, *, chars: int = 0) -> None:
        """Record suggestion UX outcomes without saving the suggested text."""
        if action not in ("shown", "accepted"):
            raise ValueError("suggestion action must be 'shown' or 'accepted'")
        self.transcript.write("suggestion", action=action, chars=chars)

    @property
    def pending_steering(self) -> tuple[str, ...]:
        """Messages from `steer` waiting for the agent's next step."""
        return tuple(self._steering)

    def take_steering(self) -> list[str]:
        """Messages from `steer` not delivered yet (and forget them)."""
        taken, self._steering = self._steering, []
        return taken

    # ------------------------------------------------------------------ side calls

    async def aside(self, question: str) -> str:
        """Answer a side question without changing the conversation.

        Parameters
        ----------
        question : str
            Question about the current conversation.

        Returns
        -------
        str
            The model's answer, or a short explanation when a tool is required.

        Raises
        ------
        ValueError
            If ``question`` is empty.
        """
        question = question.strip()
        if not question:
            raise ValueError("ask a question, e.g. /btw why this approach?")
        answer, wanted_tool = await self._side_call(
            f"{load_prompt('aside')}\n\nQuestion: {question}", "aside"
        )
        if not answer and wanted_tool:
            return "That needs a tool (reading or running something): ask it in the conversation instead."
        return answer or "(no answer)"

    async def suggest_next(self) -> str:
        """A guess at the user's next message, for the input box once a turn ends; "" when
        there's no conversation yet or no clear next step."""
        if not any(m.role == "assistant" for m in self.history.messages):
            return ""
        text, _ = await self._side_call(load_prompt("next_prompt"), "suggestion")
        line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        line = line.strip("\"'` ")
        return line if len(line) <= MAX_SUGGESTION else ""

    async def _side_call(self, note: str, kind: str) -> tuple[str, bool]:
        """One model call on a copy of the conversation plus `note`, kept out of the
        conversation. It sends the agent's own system prompt and tool definitions, so the
        provider's cached prefix covers nearly all of it; any tool call is dropped.
        Returns the reply's text and whether the model tried to call a tool."""
        messages = list(self.history.messages) or [Message(role="system", content=self._system_prompt())]
        last = messages[-1]
        if last.role == "user":  # a turn that just started: one user message, not two
            note = "\n\n".join(filter(None, [last.content or "", note]))
            messages[-1] = Message(role="user", content=note, images=list(last.images))
        else:
            messages.append(Message(role="user", content=note))
        allowed = set(self.options.allowed_tools) if self.options.allowed_tools else None
        available = {t.name: t.spec() for t in self.tools.available(self.mode, allowed)}
        text: list[str] = []
        called = False
        usage: Usage | None = None
        async for chunk in self.llm.stream(messages, list(available.values())):
            text.append(chunk.text)
            called = called or chunk.tool_call_delta is not None
            usage = chunk.usage or usage
        if usage is not None:
            record_usage(usage, self.llm, self.ctx, self.transcript, kind=kind)
        answer, calls = recover_calls(
            "".join(text), lambda name: available[name].parameters if name in available else None
        )
        return answer.strip(), called or bool(calls)

    async def run_skill(self, name: str, args: str = "") -> AsyncGenerator[Event, None]:
        """Run an installed skill as a turn.

        Parameters
        ----------
        name : str
            Installed skill name.
        args : str, optional
            Additional instructions passed to the skill.

        Yields
        ------
        Event
            Skill loading, model, tool, and completion events.

        Raises
        ------
        ValueError
            If ``name`` is not installed.
        """
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(s.name for s in self.skills.list()) or "(none)"
            raise ValueError(f"unknown skill {name!r}; available: {available}")
        prompt = build_skill_prompt(skill, args)
        yield SkillLoaded(skill.name, count_tokens(prompt), "command")
        async for event in self._turn(prompt):
            yield event

    def close(self) -> None:
        """Release proxies, logs, and temporary session data."""
        for proxy in [self.proxy, *self._extension_proxies.values()]:
            if proxy is not None:
                proxy.close()
        self.transcript.close()
        self.scratch.cleanup()

    # `with Session(...) as session:` / `async with ...`: closed however the block ends.
    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    async def __aenter__(self) -> Session:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()


def list_sessions(options: SessionOptions | None = None, limit: int = 20) -> list[SessionInfo]:
    """List resumable conversations from newest to oldest.

    Parameters
    ----------
    options : SessionOptions, optional
        Selects the session store and working directory.
    limit : int, default=20
        Maximum number of conversations to return.

    Returns
    -------
    list[SessionInfo]
        Saved conversation summaries.
    """
    options = options or SessionOptions()
    if options.store is not None:
        return options.store.sessions(limit)
    working_dir = Path(options.working_dir or load_config(options.config_path).working_dir)
    return FileSessionStore(working_dir.expanduser().resolve()).sessions(limit)


async def query(prompt: str, options: SessionOptions | None = None) -> AsyncGenerator[Event, None]:
    """Run one prompt in a fresh session.

    Parameters
    ----------
    prompt : str
        Message to send to the agent.
    options : SessionOptions, optional
        Configuration for the temporary session.

    Yields
    ------
    Event
        Events emitted during the turn.
    """
    session = Session(options)
    try:
        async for event in session.run(prompt):
            yield event
    finally:
        session.close()


async def invoke_skill(
    name: str, args: str = "", options: SessionOptions | None = None
) -> AsyncGenerator[Event, None]:
    """Run one skill in a fresh session.

    Parameters
    ----------
    name : str
        Installed skill name.
    args : str, optional
        Additional instructions passed to the skill.
    options : SessionOptions, optional
        Configuration for the temporary session.

    Yields
    ------
    Event
        Events emitted during the skill turn.
    """
    session = Session(options)
    try:
        async for event in session.run_skill(name, args):
            yield event
    finally:
        session.close()
