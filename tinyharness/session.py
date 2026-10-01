"""A Session wires the harness together and holds state across prompts.

    session = Session(SessionOptions(permission_mode="acceptEdits"))
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
import logging
import shutil
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tinyharness import trust
from tinyharness.agent import world
from tinyharness.agent.compaction import SUMMARY_MARK, Compactor
from tinyharness.agent.history import History, Image, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop, summary_notes
from tinyharness.agent.prompts import build_system_prompt, load_agent_md
from tinyharness.agent.sessionlog import (
    FileSessionStore,
    Replayed,
    SessionInfo,
    SessionLog,
    SessionStore,
    replay,
)
from tinyharness.config import Config, dotenv_paths, load_auth, load_config, load_dotenv_files
from tinyharness.events import Compacted, Done, Event, ToolResult, ToolUse
from tinyharness.hooks import Hook, HookRunner, load_hooks
from tinyharness.llm.client import LLMClient, Ping
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.providers import DEFAULT_CONTEXT, Provider
from tinyharness.llm.tokens import estimate_tokens
from tinyharness.llm.types import Effort
from tinyharness.llm.usage import UsageMeter
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
from tinyharness.plugins import load_plugins
from tinyharness.sandbox import Sandbox, SandboxPolicy, detect_sandbox
from tinyharness.skills import SkillRegistry, build_skill_prompt, load_skills
from tinyharness.subagents import AgentSpec, load_agents
from tinyharness.subagents.runner import run_subagent
from tinyharness.tools.agent import AgentTool
from tinyharness.tools.base import Tool, ToolContext, ToolError, file_version
from tinyharness.tools.bash import BashTool
from tinyharness.tools.registry import default_registry
from tinyharness.tools.screenshot import ScreenshotTool
from tinyharness.tools.skill import SkillTool
from tinyharness.tools.team import TeamTool
from tinyharness.tools.web import WebSearchTool
from tinyharness.trust import TrustStore
from tinyharness.workspace import Scratch, Workspace


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
    log.warning("agent %r from this project asks for bypassPermissions; using acceptEdits", spec.name)
    return replace(spec, permission_mode=PermissionMode.ACCEPT_EDITS)


def _new_id() -> str:
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


class MissingAPIKeyError(RuntimeError):
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        super().__init__(
            f"No API key for {provider.label}. Add one with /key {provider.name} (or run "
            f"`tinyharness --setup`), or set {provider.api_key_env} in your shell."
        )


@dataclass
class SessionOptions:
    """Per-session overrides on top of the loaded Config. `None` means "use the config"."""

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
    tools: list[Tool[Any]] = field(default_factory=list)  # your own tools; a built-in's name replaces it
    allowed_tools: list[str] | None = None
    agents: list[AgentSpec] = field(default_factory=list)  # more subagents; same name replaces one
    extra_system: str = ""
    max_steps: int = 50

    enable_skills: bool = True
    enable_subagents: bool = True  # load agents from files (built-in, ~/.tinyharness, project)
    enable_plugins: bool = True
    expose_skill_tool: bool = False  # let the model invoke skills itself (costs tokens every turn)

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
            cfg.api_key = self.api_key
        if self.todo_list is not None:
            cfg.todo_list = self.todo_list
        return cfg


class Session:
    def __init__(self, options: SessionOptions | None = None, config: Config | None = None) -> None:
        self.options = options or SessionOptions()
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

        # Skills may ship reference files; let tools read them without prompting.
        for skill in self.skills.list():
            if skill.directory is not None:
                self.workspace.add_read_root(skill.directory)

        self.tools = default_registry()
        if not self.config.todo_list:
            self.tools.remove("TodoWrite")
        self.tools.register(WebSearchTool(lambda: self.llm))
        self.sandbox: Sandbox | None = detect_sandbox() if self.config.sandbox.enabled else None
        self.sandbox_policy = SandboxPolicy.for_project(
            root,
            scratch=self.scratch.dir,
            cache_dir=home_dir() / "sandbox-cache",
            extra_write=self.config.sandbox.allow_write,
            allow_localhost=self.config.sandbox.allow_localhost,
        )
        if shutil.which("bash") or shutil.which("sh"):
            self.tools.register(BashTool(self.sandbox, self.sandbox_policy))
        self.tools.register(ScreenshotTool(self.sandbox, self.sandbox_policy))
        for tool in self.options.tools:  # before the Agent tool, so subagents can use them too
            self.tools.register(tool)
        if self.options.expose_skill_tool and len(self.skills):
            self.tools.register(SkillTool(self.skills))
        if self.agents:
            self.tools.register(AgentTool(self.agents, self._spawn_subagent, self._agent_is_read_only))
            self.tools.register(TeamTool(self.agents, self._spawn_subagent, self._agent_is_read_only))

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

        self.models = ModelDirectory(home_dir() / "cache" / "models")
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
            instructions_now=world.text_hash(load_agent_md(self.workspace.root)),
        )
        report.unfinished = unfinished
        # A fresh system prompt: the model, scratch dir or AGENT.md may have changed.
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
            "world", git=world.git_state(root), agent_md=world.text_hash(load_agent_md(root))
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
        )
        self.llm.show_reasoning = cfg.show_thinking
        self.gate.classifier = AutoModeClassifier(self.llm, model=cfg.small_model)
        self.compactor = Compactor(
            self.llm, context_window=lambda: self.context_window, threshold=cfg.auto_compact_threshold
        )

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
        )
        return client, None

    def _agent_is_read_only(self, spec: AgentSpec) -> bool:
        """An agent that can't change anything, so several can safely run at once."""
        names = self.tools.names() if spec.tools is None else spec.tools
        return all(tool.read_only for name in names if (tool := self.tools.get(name)) is not None)

    # ------------------------------------------------------------------ settings

    def reload_keys(self) -> None:
        """Re-read auth.json (and .env files), in case a key was added while the session ran."""
        load_auth(self.config.providers)
        load_dotenv_files(*dotenv_paths(self.config.working_dir))

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
        return build_system_prompt(
            self.workspace.root,
            self.scratch.dir,
            model_name=self.model_label,
            model_id=self.config.model,
            provider=self.config.provider_info.label,
            extra=self.options.extra_system,
            todo_list=self.config.todo_list,
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
        """Reasoning effort for every model: low | medium | high | max, or None for the model's default."""
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
        """Start a new conversation, as Claude Code's /clear does: the old one stays saved
        as it was (`tinyharness -r` picks it), and this one gets its own id and log.
        Permissions and scratch space are kept, and so are the token and cost totals: they're
        what the session spent (see UsageMeter.since_clear)."""
        had_conversation = any(m.role != "system" for m in self.history.messages)
        self.history.reset()
        self.ctx.usage.mark_clear()
        self.resume_report, self._resume_note = None, ""
        if had_conversation:
            previous = self.id
            self.transcript.close()
            self.id = _new_id()
            self.transcript = self.store.writer(self.id)
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
        """Send a message and stream what happens. `images` are pictures for the model to
        look at with it (screenshots, say): see `tinyharness.tools.read.load_image`."""
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
        )
        try:
            async for event in run_agent_loop(prompt, self.history, deps, images):
                if isinstance(event, (ToolUse, ToolResult, Compacted, Done)):
                    self._sync()  # saved as it goes, so a crash loses at most one step
                if isinstance(event, Done):
                    self.transcript.write("turn_end", reason=event.reason or "done")
                yield event
        except (asyncio.CancelledError, KeyboardInterrupt, GeneratorExit):
            self.transcript.write("turn_end", reason="interrupted")
            raise
        except Exception as e:  # recorded, so a failed turn can be told apart from a stopped one
            self.transcript.write("turn_end", reason="error", error=f"{type(e).__name__}: {e}"[:1000])
            raise
        finally:
            # If the turn was interrupted mid tool call, keep the history valid.
            self.history.repair()
            self._sync()
            self._log_world()

    def steer(self, text: str) -> None:
        """Say something to the agent while a turn is running: it arrives as a user message
        before the agent's next step. If the turn ends first, `take_steering()` returns it."""
        self._steering.append(text)

    @property
    def pending_steering(self) -> tuple[str, ...]:
        """Messages from `steer` waiting for the agent's next step."""
        return tuple(self._steering)

    def take_steering(self) -> list[str]:
        """Messages from `steer` not delivered yet (and forget them)."""
        taken, self._steering = self._steering, []
        return taken

    async def run_skill(self, name: str, args: str = "") -> AsyncGenerator[Event, None]:
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(s.name for s in self.skills.list()) or "(none)"
            raise ValueError(f"unknown skill {name!r}; available: {available}")
        async for event in self.run(build_skill_prompt(skill, args)):
            yield event

    def close(self) -> None:
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
    """Saved conversations that can be resumed (`SessionOptions(resume=info.id)`), newest first."""
    options = options or SessionOptions()
    if options.store is not None:
        return options.store.sessions(limit)
    working_dir = Path(options.working_dir or load_config(options.config_path).working_dir)
    return FileSessionStore(working_dir.expanduser().resolve()).sessions(limit)


async def query(prompt: str, options: SessionOptions | None = None) -> AsyncGenerator[Event, None]:
    """Run a single prompt in a fresh session."""
    session = Session(options)
    try:
        async for event in session.run(prompt):
            yield event
    finally:
        session.close()


async def invoke_skill(
    name: str, args: str = "", options: SessionOptions | None = None
) -> AsyncGenerator[Event, None]:
    """Run a single skill in a fresh session."""
    session = Session(options)
    try:
        async for event in session.run_skill(name, args):
            yield event
    finally:
        session.close()
