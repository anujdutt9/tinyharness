"""Extensions: one Python file with a `setup(th)` function that adds tools, prompt
sections, slash commands and event handlers. The built-in features (web, agents, todo,
...) are extensions too, in tinyharness/extensions/. See docs/extension-api.md.

    # ~/.tinyharness/extensions/hello.py
    from tinyharness.ext import Extension

    def setup(th: Extension) -> None:
        @th.tool(read_only=True)
        async def hello(name: str) -> str:
            \"\"\"Greet someone by name.\"\"\"
            return f"Hello, {name}!"

Turned on in config: `[extensions] enabled = ["hello"]`.

Everything is registered in `setup()`, once, when the session starts: the tools and the
system prompt then stay the same for the whole session, so the provider's prompt cache
keeps hitting.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import re
import sys
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import Field, create_model

from tinyharness.hooks import Hook
from tinyharness.paths import home_dir, project_dir
from tinyharness.tools.base import Tool, ToolContext, ToolInput, ToolOutput

if TYPE_CHECKING:
    from tinyharness.session import Session

log = logging.getLogger(__name__)

ENTRY_POINTS = "tinyharness.extensions"  # pip packages register extensions in this group

# Extension events and the hook events they run as (hooks/hooks.py): one event bus.
EVENTS = {
    "session_start": "SessionStart",
    "user_prompt": "UserPromptSubmit",
    "tool_call": "PreToolUse",
    "tool_result": "PostToolUse",
    "turn_end": "Stop",
}


@dataclass
class Block:
    """Returned by a `tool_call` handler to stop the call, or a `turn_end` one to keep going."""

    reason: str


@dataclass
class Note:
    """Returned by a `tool_result` handler (added to the result for the model) or a
    `user_prompt` one (added to the user's message as context)."""

    text: str


class Ctx:
    """What an event handler gets besides the payload."""

    @staticmethod
    def block(reason: str) -> Block:
        return Block(reason)

    @staticmethod
    def note(text: str) -> Note:
        return Note(text)


@dataclass
class Call:
    """A tool call, as a `tool_call` handler sees it."""

    tool: str
    input: dict[str, Any]


@dataclass
class Command:
    name: str  # without the slash
    help: str
    handler: Callable[[str, CommandContext], Awaitable[None] | None]
    usage: str = ""
    extension: str = ""


class CommandContext:
    """What a slash command's handler gets: a way to print and to talk to the model."""

    def __init__(self, say: Callable[[str], None], send: Callable[[str], Awaitable[None]], session: Session):
        self.say = say
        self.send = send
        self.session = session


@dataclass
class RunResult:
    """What `th.run` returns: the exit code (None if it timed out) and the combined output."""

    code: int | None
    output: str


@dataclass
class Extension:
    """The `th` object an extension's `setup()` receives. Everything it registers belongs
    to this extension, so it can be listed (with its token cost) and unloaded."""

    name: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    session: Session | None = None
    tools: list[Tool[Any]] = field(default_factory=list)
    sections: list[tuple[str, str]] = field(default_factory=list)
    hooks: list[Hook] = field(default_factory=list)
    commands: list[Command] = field(default_factory=list)
    needs: list[str] = field(default_factory=list)  # NEEDS, as declared
    replaces: list[str] = field(default_factory=list)  # tools it removes from the core
    fingerprint: str = ""  # of its code, so a changed extension asks for its needs again
    granted: bool = True  # its needs are in effect (no needs, given in code, or you approved)

    @property
    def hosts(self) -> list[str]:
        """Hosts its NEEDS ask to reach (`network:<host>`)."""
        return [n.split(":", 1)[1] for n in self.needs if n.startswith("network:")]

    @property
    def writes(self) -> list[Path]:
        """Folders its NEEDS ask to write (`write:<path>`)."""
        return [Path(n.split(":", 1)[1]).expanduser().resolve() for n in self.needs if n.startswith("write:")]

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(f"tinyharness.extensions.{self.name}")

    @property
    def state_dir(self) -> Path:
        path = home_dir() / "extensions-data" / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---- tools

    def tool(
        self,
        fn: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        read_only: bool = False,
        network: bool = False,
    ) -> Any:
        """Make a tool from an async function: its docstring is the description, its type
        hints and defaults the schema. A parameter named `ctx` gets the ToolContext."""

        def register(f: Callable[..., Any]) -> Callable[..., Any]:
            self.tools.append(function_tool(f, name=name, read_only=read_only, network=network))
            return f

        return register(fn) if fn is not None else register

    def add_tool(self, tool: Tool[Any]) -> None:
        """Add a Tool subclass instance (tools/base.py) as it is."""
        self.tools.append(tool)

    def remove_tool(self, name: str) -> None:
        """Take a core tool away (an extension that replaces how something is done)."""
        self.replaces.append(name)

    # ---- prompt

    def prompt_section(self, name: str, text: str) -> None:
        """Text added to the system prompt under `# <name>`, after the core prompt."""
        self.sections.append((name, text.strip()))

    # ---- events

    def on(self, event: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if event not in EVENTS:
            raise ValueError(f"unknown event {event!r}; one of: {', '.join(EVENTS)}")

        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.hooks.append(
                Hook(event=EVENTS[event], type="python", callable=_as_hook(event, handler), name=self.name)
            )
            return handler

        return register

    # ---- commands

    def command(self, name: str, *, help: str = "", usage: str = "") -> Callable[[Callable[..., Any]], Any]:
        def register(handler: Callable[..., Any]) -> Callable[..., Any]:
            self.commands.append(Command(name.lstrip("/"), help, handler, usage, self.name))
            return handler

        return register

    # ---- subagents

    async def run(self, command: str, *, timeout: float = 120) -> RunResult:
        """Run a shell command in the sandbox, with this extension's needs applied (its
        hosts reachable, its folders writable) once you've approved them."""
        if self.session is None:
            raise RuntimeError("run needs a session")
        return await self.session.run_for_extension(self, command, timeout=timeout)

    async def run_agent(
        self,
        task: str,
        *,
        agent: str = "general-purpose",
        model: str | None = None,
        label: str = "",
        extra_tools: list[Tool[Any]] | None = None,
        inbox: Callable[[], list[str]] | None = None,
        ctx: ToolContext | None = None,
    ) -> str:
        """Run a subagent to completion under this session's gate and sandbox; its answer."""
        if self.session is None:
            raise RuntimeError("run_agent needs a session")
        return await self.session.run_agent(
            agent, task, model=model, label=label, extra_tools=extra_tools or [], inbox=inbox, ctx=ctx
        )


def _as_hook(event: str, handler: Callable[..., Any]) -> Callable[[str, dict[str, Any]], Awaitable[Any]]:
    """An extension handler as a python hook: its Block/Note become hook decisions."""

    async def hook(_event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        arg: Any = payload
        if event == "tool_call":
            arg = Call(payload.get("tool", ""), dict(payload.get("args") or {}))
        result = handler(arg, Ctx())
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Block):
            return {"decision": "block" if event == "turn_end" else "deny", "reason": result.reason}
        if isinstance(result, Note):
            return {"reason": result.text} if event == "tool_result" else {"context": result.text}
        return None

    return hook


# ---------------------------------------------------------------------- function tools

_ARG_DOC = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


def function_tool(
    fn: Callable[..., Any], *, name: str | None = None, read_only: bool = False, network: bool = False
) -> Tool[Any]:
    """A Tool built from a function's signature and docstring."""
    if not inspect.iscoroutinefunction(fn):
        raise TypeError(f"{fn.__name__}: a tool must be an async function")
    doc = inspect.getdoc(fn) or ""
    summary, _, rest = doc.partition("\n\n")
    arg_docs = {m[1]: m[2].strip() for line in rest.splitlines() if (m := _ARG_DOC.match(line))}
    hints = get_type_hints(fn)
    fields: dict[str, Any] = {}
    wants_ctx = False
    for param in inspect.signature(fn).parameters.values():
        if param.name == "ctx":
            wants_ctx = True
            continue
        annotation = hints.get(param.name, str)
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param.name] = (annotation, Field(default, description=arg_docs.get(param.name)))
    tool_name = name or fn.__name__
    model = create_model(f"{tool_name}_input", __base__=ToolInput, **fields)

    async def run(self: Tool[Any], args: Any, ctx: ToolContext) -> str | ToolOutput:
        kwargs = {k: getattr(args, k) for k in fields}
        if wants_ctx:
            kwargs["ctx"] = ctx
        result = await fn(**kwargs)
        return result if isinstance(result, ToolOutput) else str(result)

    cls = type(
        f"{tool_name}_tool",
        (Tool,),
        {
            "name": tool_name,
            "description": " ".join(summary.split()) or tool_name,
            "Input": model,
            "read_only": read_only,
            "network": network,
            "run": run,
        },
    )
    return cls()


# ---------------------------------------------------------------------- loading


@dataclass
class LoadError:
    name: str
    error: str


def find_module(name: str, working_dir: Path, *, project: bool) -> ModuleType | None:
    """An extension's module: built-in, ~/.tinyharness/extensions, the project's (only if
    `project`, a trusted folder), or an installed package's entry point."""
    if re.fullmatch(r"[A-Za-z_][\w-]*", name) is None:
        raise ValueError(f"not an extension name: {name!r}")
    builtin = f"tinyharness.extensions.{name.replace('-', '_')}"
    if importlib.util.find_spec(builtin) is not None:
        return importlib.import_module(builtin)
    folders = [home_dir() / "extensions"] + ([project_dir(working_dir) / "extensions"] if project else [])
    for folder in folders:
        for path in (folder / f"{name}.py", folder / name / "__init__.py"):
            if path.is_file():
                return _load_file(name, path)
    for ep in importlib.metadata.entry_points(group=ENTRY_POINTS):
        if ep.name == name:
            loaded = ep.load()
            return loaded if isinstance(loaded, ModuleType) else sys.modules[loaded.__module__]
    return None


def _load_file(name: str, path: Path) -> ModuleType:
    module_name = f"tinyharness_ext_{name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"can't load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


Setup = Callable[[Extension], None]


def load_extensions(
    names: list[str],
    session: Session,
    settings: dict[str, Any],
    *,
    project: bool,
    given: Sequence[ModuleType | Setup] = (),
) -> tuple[list[Extension], list[LoadError]]:
    """Set up each named extension, in order, then the ones `given` in code (SDK:
    `SessionOptions(extensions=[...])`, a module or a setup function). One that fails is
    reported and skipped."""
    loaded: list[Extension] = []
    errors: list[LoadError] = []
    sources: list[tuple[str, ModuleType | Setup | None]] = []
    for name in dict.fromkeys(names):
        try:
            sources.append((name, find_module(name, session.workspace.root, project=project)))
        except Exception as e:
            errors.append(LoadError(name, f"{type(e).__name__}: {e}"))
    for item in given:  # named after the module, or the function (`def hello(th)` is "hello")
        name = item.__name__.rpartition(".")[2]
        if name not in (n for n, _ in sources):  # also turned on in config: load it once
            sources.append((name, item))
    for name, source in sources:
        try:
            if source is None:
                errors.append(LoadError(name, "not found"))
                continue
            setup = source if not isinstance(source, ModuleType) else getattr(source, "setup", None)
            if not callable(setup):
                errors.append(LoadError(name, "has no setup(th) function"))
                continue
            config = settings.get(name)
            ext = Extension(
                name=name,
                description=(inspect.getdoc(source) or "").split("\n\n")[0].strip(),
                config=dict(config) if isinstance(config, dict) else {},
                session=session,
                needs=check_needs(getattr(source, "NEEDS", []) or []),
                fingerprint=fingerprint(source),
            )
            # Needs take effect once approved; an extension you pass in code is your own.
            given_in_code = any(source is item for item in given)
            ext.granted = not ext.needs or given_in_code or NeedsConsent().granted(ext)
            setup(ext)
            loaded.append(ext)
        except Exception as e:  # one broken extension shouldn't stop the session
            log.warning("extension %s failed to load: %s", name, e)
            errors.append(LoadError(name, f"{type(e).__name__}: {e}"))
    return loaded, errors


def builtin_names() -> list[str]:
    from tinyharness import extensions

    return sorted(p.stem for p in Path(extensions.__file__).parent.glob("*.py") if not p.stem.startswith("_"))


# ---------------------------------------------------------------------- needs

_NEED = re.compile(r"^(network:[A-Za-z0-9*.-]+|write:.+)$")


def check_needs(needs: Iterable[Any]) -> list[str]:
    """NEEDS as declared, each `network:<host>` or `write:<path>`; anything else is an error."""
    out = []
    for need in needs:
        if not isinstance(need, str) or not _NEED.match(need):
            raise ValueError(f"NEEDS entries are 'network:<host>' or 'write:<path>', not {need!r}")
        out.append(need)
    return out


def fingerprint(source: ModuleType | Setup) -> str:
    """A hash of an extension's code: its file, or every .py file of a package."""
    path = getattr(source, "__file__", None)
    if not path:
        return "code"
    file = Path(path)
    files = sorted(file.parent.rglob("*.py")) if file.name == "__init__.py" else [file]
    digest = hashlib.sha256()
    for f in files:
        digest.update(str(f.relative_to(file.parent)).encode() + b"\0" + f.read_bytes())
    return digest.hexdigest()


class NeedsConsent:
    """The needs you've approved, per extension, with a fingerprint of its code then: a
    changed extension (or changed needs) asks again. Kept in ~/.tinyharness/, which
    nothing in a project and no sandboxed command can write."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or home_dir() / "extension-needs.json"

    def granted(self, ext: Extension) -> bool:
        entry = self._load().get(ext.name)
        if not isinstance(entry, dict):
            return False
        return entry.get("fingerprint") == ext.fingerprint and entry.get("needs") == sorted(ext.needs)

    def grant(self, ext: Extension) -> None:
        data = self._load()
        data[ext.name] = {"fingerprint": ext.fingerprint, "needs": sorted(ext.needs)}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        except OSError as e:
            log.warning("could not save extension consent to %s: %s", self.path, e)
        ext.granted = True

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}
