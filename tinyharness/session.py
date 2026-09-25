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

import time
import uuid
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.prompts import build_system_prompt
from tinyharness.agent.transcript import Transcript
from tinyharness.config import Config, load_config
from tinyharness.events import Event
from tinyharness.hooks import Hook, HookRunner, load_hooks
from tinyharness.llm.client import LLMClient
from tinyharness.llm.providers import Provider
from tinyharness.permissions import PermissionCallback, PermissionGate, PermissionMode, PermissionRules
from tinyharness.permissions.classifier import AutoModeClassifier
from tinyharness.plugins import load_plugins
from tinyharness.skills import SkillRegistry, build_skill_prompt, load_skills
from tinyharness.subagents import AgentSpec, load_agents
from tinyharness.subagents.runner import run_subagent
from tinyharness.tools.agent import AgentTool
from tinyharness.tools.base import ToolContext
from tinyharness.tools.registry import default_registry
from tinyharness.tools.skill import SkillTool
from tinyharness.workspace import Scratch, Workspace


class MissingAPIKeyError(RuntimeError):
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        super().__init__(
            f"No API key for {provider.label}. Set ${provider.api_key_env} "
            "(in your shell or a .env file), or run `tinyharness` to set one up."
        )


@dataclass
class SessionOptions:
    """Per-session overrides on top of the loaded Config. `None` means "use the config"."""

    config_path: Path | None = None
    working_dir: Path | None = None
    provider: str | None = None
    model: str | None = None
    permission_mode: PermissionMode | str | None = None
    thinking_budget: int | None = None
    api_key: str | None = None

    permission_callback: PermissionCallback | None = None
    hooks: list[Hook] = field(default_factory=list)
    allowed_tools: list[str] | None = None
    extra_system: str = ""
    max_steps: int = 50

    enable_skills: bool = True
    enable_subagents: bool = True
    enable_plugins: bool = True
    expose_skill_tool: bool = False  # let the model invoke skills itself (costs tokens every turn)

    def resolve_config(self) -> Config:
        cfg = load_config(self.config_path)
        if self.working_dir is not None:
            cfg.working_dir = Path(self.working_dir)
            cfg.working_dir.mkdir(parents=True, exist_ok=True)
        if self.provider:
            cfg.use_provider(self.provider)
        if self.model:
            cfg.select(self.model)
        if self.permission_mode:
            cfg.permission_mode = PermissionMode.parse(self.permission_mode)
        if self.thinking_budget is not None:
            cfg.thinking_budget = self.thinking_budget
        if self.api_key:
            cfg.api_key = self.api_key
        return cfg


class Session:
    def __init__(self, options: SessionOptions | None = None, config: Config | None = None) -> None:
        self.options = options or SessionOptions()
        self.config = config or self.options.resolve_config()
        self.id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        root = self.config.working_dir

        self.workspace = Workspace(root)
        self.scratch = Scratch(root, self.id, max_age_hours=self.config.scratch_max_age_hours)
        self.transcript = Transcript(root, self.id)
        self.history = History()
        self.ctx = ToolContext(workspace=self.workspace, config=self.config, scratch=self.scratch)

        # Extensions: skills, agents, hooks (plugins contribute to all three).
        self.skills = SkillRegistry(load_skills(root) if self.options.enable_skills else [])
        self.agents: dict[str, AgentSpec] = load_agents(root) if self.options.enable_subagents else {}
        hooks = load_hooks(root)
        if self.options.enable_plugins:
            for plugin in load_plugins(root):
                for skill in plugin.skills:
                    self.skills.add(skill)
                self.agents.update({a.name: a for a in plugin.agents})
                hooks.extend(plugin.hooks)
        self.hooks = HookRunner([*hooks, *self.options.hooks], cwd=root)

        # Skills may ship reference files; let tools read them without prompting.
        for skill in self.skills.list():
            if skill.directory is not None:
                self.workspace.add_read_root(skill.directory)

        self.tools = default_registry()
        if self.options.expose_skill_tool and len(self.skills):
            self.tools.register(SkillTool(self.skills))
        if self.agents:
            self.tools.register(AgentTool(self.agents, self._spawn_subagent))

        # One gate for the whole session, so "don't ask again" answers and mode
        # changes (e.g. Shift+Tab in the CLI) carry over from prompt to prompt.
        cfg = self.config
        self.gate = PermissionGate(
            workspace=self.workspace,
            mode=cfg.permission_mode,
            rules=PermissionRules.parse(cfg.permissions.allow, cfg.permissions.deny, root),
            hooks=self.hooks,
            callback=self.options.permission_callback,
            scratch=self.scratch,
        )

        self._connect_llm()
        self._started = False
        self.transcript.write("session_start", model=cfg.model, mode=self.mode, working_dir=str(root))

    # ------------------------------------------------------------------ setup

    def _connect_llm(self) -> None:
        """(Re)build everything that depends on the provider/model."""
        cfg = self.config
        api_key = cfg.resolve_api_key()
        if not api_key:
            raise MissingAPIKeyError(cfg.provider_info)
        self.llm = LLMClient(
            cfg.base_url,
            api_key,
            cfg.model,
            extra_headers=cfg.extra_headers,
            timeout=cfg.request_timeout,
            prompt_caching=cfg.prompt_caching,
            thinking_budget=cfg.thinking_budget,
        )
        self.gate.classifier = AutoModeClassifier(self.llm, model=cfg.small_model)
        self.compactor = Compactor(
            self.llm, context_window=cfg.context_window, threshold=cfg.auto_compact_threshold
        )

    async def _spawn_subagent(self, spec: AgentSpec, task: str, ctx: ToolContext) -> str:
        # Same rules as the parent, but the subagent's own permission mode.
        gate = replace(self.gate, mode=spec.permission_mode)
        return await run_subagent(spec, task, ctx=ctx, llm=self.llm, tools=self.tools, gate=gate)

    # ------------------------------------------------------------------ settings

    def set_model(self, spec: str) -> None:
        """Switch model; `provider:model` switches provider too. The conversation is kept."""
        provider, _ = self.config.parse_model(spec)
        self._switch(lambda: self.config.select(spec), provider)

    def set_provider(self, name: str) -> None:
        """Switch provider, using its default model. The conversation is kept."""
        self._switch(lambda: self.config.use_provider(name), name)

    def _switch(self, apply: Callable[[], None], provider: str) -> None:
        if provider not in self.config.providers:
            raise ValueError(
                f"unknown provider {provider!r}; choose one of: {', '.join(self.config.providers)}"
            )
        if not self.config.has_key(provider):
            raise MissingAPIKeyError(self.config.providers[provider])
        apply()
        self._connect_llm()

    @property
    def mode(self) -> PermissionMode:
        return self.gate.mode

    def set_mode(self, mode: PermissionMode | str) -> None:
        self.gate.mode = PermissionMode.parse(mode)

    def set_thinking_budget(self, tokens: int) -> None:
        self.config.thinking_budget = max(0, tokens)
        self.llm.thinking_budget = self.config.thinking_budget

    def clear(self) -> None:
        """Forget the conversation (permissions and scratch space are kept)."""
        self.history.reset()

    # ------------------------------------------------------------------ running

    async def run(self, prompt: str) -> AsyncGenerator[Event, None]:
        if not self._started:
            await self.hooks.fire("SessionStart", {"session_id": self.id})
            self._started = True
        if not self.history.messages:
            system = build_system_prompt(self.workspace.root, self.scratch.dir, self.options.extra_system)
            self.history.append(Message(role="system", content=system))

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
        )
        try:
            async for event in run_agent_loop(prompt, self.history, deps):
                yield event
        finally:
            # If the turn was interrupted mid tool call, keep the history valid.
            self.history.repair()

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
