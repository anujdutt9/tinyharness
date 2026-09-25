"""Slash commands for the REPL.

Adding a command is one decorated function:

    @command("/hello", "say hello", group="Session")
    def hello(repl: Repl, arg: str) -> None:
        repl.console.print("hi")

Commands that take a choice (/model, /mode, /provider) open a picker when run
without an argument.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from tinyharness.cli import setup
from tinyharness.cli.theme import MODE_HINT, MODE_LABEL, MODE_STYLE
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import remember_model
from tinyharness.llm.providers import fetch_models
from tinyharness.llm.tokens import count_message_tokens
from tinyharness.permissions import PermissionMode
from tinyharness.security import redact
from tinyharness.session import MissingAPIKeyError

if TYPE_CHECKING:
    from tinyharness.cli.repl import Repl

Handler = Callable[["Repl", str], Awaitable[None] | None]


@dataclass(frozen=True)
class Command:
    name: str
    help: str
    handler: Handler
    usage: str = ""
    group: str = ""


COMMANDS: dict[str, Command] = {}


def command(
    name: str, help: str, *, group: str, usage: str = "", aliases: tuple[str, ...] = ()
) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        for n in (name, *aliases):
            COMMANDS[n] = Command(name, help, fn, usage, group)
        return fn

    return register


# --------------------------------------------------------------------- helpers


def _short(text: str, limit: int = 72) -> str:
    first = " ".join(text.split())
    return first if len(first) <= limit else first[: limit - 1] + "…"


def _grid(first_width: int | None = None) -> Table:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="muted", no_wrap=True, min_width=first_width)
    table.add_column()
    return table


def _ok(repl: Repl, message: str) -> None:
    """`message` may use rich markup."""
    repl.console.print(f"  [ok]✔[/] {message}")


def _fail(repl: Repl, message: str) -> None:
    repl.console.print(Text("  ✗ ", style="err").append(message))


# --------------------------------------------------------------------- session


@command("/help", "show commands and shortcuts", group="Session", aliases=("/?",))
def help_(repl: Repl, arg: str) -> None:
    groups: dict[str, list[Command]] = {}
    for cmd in dict.fromkeys(COMMANDS.values()):  # de-duplicate aliases, keep order
        groups.setdefault(cmd.group, []).append(cmd)
    if len(repl.session.skills):
        skill = Command("/<skill>", "run a skill (see /skills)", help_, "[args]", "Tools & skills")
        groups["Tools & skills"].append(skill)

    width = max(len(f"    {c.name} {c.usage}") for cmds in groups.values() for c in cmds)
    for group, cmds in groups.items():
        repl.console.print(Text(f"  {group}", style="brand"))
        table = _grid(width)
        for cmd in cmds:
            table.add_row(f"    [bold default]{cmd.name}[/] {cmd.usage}".rstrip(), cmd.help)
        repl.console.print(table)
        repl.console.print()

    repl.console.print(Text("  Shortcuts", style="brand"))
    table = _grid(width)
    for keys, what in [
        ("shift+tab", "switch permission mode"),
        ("alt+enter", "new line"),
        ("↑ / ↓", "previous messages"),
        ("ctrl+c", "stop the agent / clear the input"),
        ("ctrl+d", "exit"),
    ]:
        table.add_row(f"    [key]{keys}[/]", what)
    repl.console.print(table)


@command("/exit", "quit", group="Session", aliases=("/quit",))
def exit_(repl: Repl, arg: str) -> None:
    repl.running = False


@command("/clear", "start a fresh conversation (permissions are kept)", group="Session")
def clear(repl: Repl, arg: str) -> None:
    n = len(repl.session.history.messages)
    repl.session.clear()
    repl.context_used = 0.0
    repl.console.clear()
    _ok(repl, f"Cleared {n} message{'s' if n != 1 else ''}. Fresh start.")


@command("/context", "show how much of the context window is used", group="Session")
def context(repl: Repl, arg: str) -> None:
    used = count_message_tokens(repl.session.history.to_openai())
    window = repl.session.config.context_window
    fraction = min(used / window, 1.0)
    width = 30
    filled = round(fraction * width)
    color = "ok" if fraction < 0.6 else "warn" if fraction < 0.85 else "err"
    bar = Text("  ").append("█" * filled, style=color).append("░" * (width - filled), style="subtle")
    bar.append(f"  {fraction:.0%}", style="bold")
    repl.console.print(bar)
    n = len(repl.session.history.messages)
    repl.console.print(f"  [muted]{used:,} of {window:,} tokens · {n} messages[/]")


@command("/cwd", "show the working directory", group="Session")
def cwd(repl: Repl, arg: str) -> None:
    repl.console.print(f"  {repl.session.workspace.root}")


# --------------------------------------------------------------------- model & provider


_SEARCH = "__search__"
_ADD_PROVIDER = "__add_provider__"


@command("/model", "pick a model (any provider you have a key for)", group="Model", usage="[provider:]model")
async def model(repl: Repl, arg: str) -> None:
    cfg = repl.session.config
    current = f"{cfg.provider}:{cfg.model}"
    spec = arg or await pick("Choose a model", _model_options(repl), current)
    if spec == _SEARCH:
        spec = await _search_models(repl)
    elif spec == _ADD_PROVIDER:
        if (provider := await setup.connect(repl.console, cfg)) is None:
            return
        spec = f"{provider}:"
    if spec:
        _switch(repl, lambda: repl.session.set_model(spec))


def _model_options(repl: Repl) -> list[tuple[str, str]]:
    """Recommended models of every provider with a key, current provider first."""
    cfg = repl.session.config
    names = [cfg.provider, *(n for n in cfg.providers if n != cfg.provider and cfg.has_key(n))]
    rows = [(n, m) for n in names for m in cfg.providers[n].models]
    width = max((len(m.display) for _, m in rows), default=10)
    pwidth = max(len(cfg.providers[n].label) for n in names)
    options = [
        (f"{n}:{m.id}", f"{m.display:<{width}}  {cfg.providers[n].label:<{pwidth}}  {m.note}".rstrip())
        for n, m in rows
    ]
    options.append((_SEARCH, f"Search all {cfg.provider_info.label} models…"))
    options.append((_ADD_PROVIDER, "Connect another provider…"))
    return options


async def _search_models(repl: Repl) -> str:
    cfg = repl.session.config
    info = cfg.provider_info
    try:
        with repl.console.status(f"[muted]Fetching models from {info.label}…[/]", spinner="dots"):
            ids = await fetch_models(info, cfg.resolve_api_key())
    except Exception as e:  # any failure just means no suggestions
        ids = []
        repl.console.print(
            f"  [warn]Couldn't list {info.label} models ({type(e).__name__}); type an id instead.[/]"
        )
    hint = f"type to search {len(ids)} models" if ids else "e.g. " + info.default_model
    model_id = await ask("Model id", choices=ids, hint=hint)
    return f"{cfg.provider}:{model_id}" if model_id else ""


@command("/provider", "pick a provider (uses its default model)", group="Model", usage="[name]")
async def provider(repl: Repl, arg: str) -> None:
    cfg = repl.session.config
    name = arg or await pick("Choose a provider", setup.provider_options(cfg), cfg.provider)
    if not name:
        return
    if name in cfg.providers and not cfg.has_key(name) and not await setup.connect(repl.console, cfg, name):
        return
    _switch(repl, lambda: repl.session.set_provider(name))


@command("/login", "add or replace a provider's API key", group="Model", usage="[provider]")
async def login(repl: Repl, arg: str) -> None:
    cfg = repl.session.config
    if arg and arg not in cfg.providers:
        _fail(repl, f"unknown provider {arg!r}; choose one of: {', '.join(cfg.providers)}")
        return
    name = await setup.connect(repl.console, cfg, arg or None)
    if name and name != cfg.provider:
        repl.console.print(
            f"  [muted]Switch to it with[/] /provider {name} [muted]or pick one of its models in[/] /model"
        )


def _switch(repl: Repl, apply: Callable[[], None]) -> None:
    try:
        apply()
    except (ValueError, MissingAPIKeyError) as e:
        _fail(repl, str(e))
        return
    cfg = repl.session.config
    remember_model(cfg.provider, cfg.model)
    repl.context_used = 0.0
    window = f"{cfg.context_window // 1000:,}k context"
    _ok(
        repl,
        f"Using [bold]{escape(cfg.model_info.display)}[/] [muted]via {cfg.provider_info.label} · {window}[/]",
    )


@command("/whoami", "show provider, endpoint, model and API key source", group="Model")
def whoami(repl: Repl, arg: str) -> None:
    cfg = repl.session.config
    table = _grid()
    for label, value in [
        ("provider", f"{cfg.provider_info.label} ({cfg.provider})"),
        ("endpoint", cfg.base_url),
        ("model", cfg.model),
        ("context", f"{cfg.context_window:,} tokens"),
        ("key from", "api_key (config)" if cfg.api_key else f"${cfg.api_key_env}"),
        ("key", redact(cfg.resolve_api_key())),
    ]:
        table.add_row(f"  {label}", escape(str(value)))
    repl.console.print(table)


@command("/ping", "check that the API key works", group="Model")
async def ping(repl: Repl, arg: str) -> None:
    with repl.console.status("[muted]Pinging…[/]", spinner="dots"):
        ok, detail = await repl.session.llm.ping()
    (_ok if ok else _fail)(repl, escape(detail) if ok else detail)


_THINKING_PRESETS = {"off": 0, "low": 1024, "medium": 4096, "high": 16384}


@command("/think", "set the reasoning budget", group="Model", usage="[off|low|medium|high|<tokens>]")
async def think(repl: Repl, arg: str) -> None:
    arg = arg.lower()
    if not arg:
        current = repl.session.config.thinking_budget
        preset = next((k for k, v in _THINKING_PRESETS.items() if v == current), None)
        options = [(k, f"{k:<7} {f'{v:,} tokens' if v else ''}") for k, v in _THINKING_PRESETS.items()]
        arg = await pick("Reasoning budget", options, preset) or ""
        if not arg:
            return
    if arg in _THINKING_PRESETS:
        budget = _THINKING_PRESETS[arg]
    elif arg.isdigit():
        budget = int(arg)
    else:
        _fail(repl, "usage: /think off|low|medium|high|<tokens>")
        return
    repl.session.set_thinking_budget(budget)
    _ok(repl, f"Thinking {f'budget set to [bold]{budget:,}[/] tokens' if budget else 'turned off'}")


# --------------------------------------------------------------------- permissions, tools, skills


@command("/mode", "pick the permission mode", group="Tools & skills", usage="[mode]")
async def mode(repl: Repl, arg: str) -> None:
    if not arg:
        options = [(m, f"{MODE_LABEL[m]:<13} {MODE_HINT[m]}") for m in PermissionMode]
        choice = await pick("Permission mode", options, repl.session.mode)
        if choice is None:
            return
        arg = choice
    try:
        repl.session.set_mode(arg)
    except ValueError as e:
        _fail(repl, str(e))
        return
    m = repl.session.mode
    _ok(repl, f"Mode set to [bold {MODE_STYLE[m]}]{MODE_LABEL[m]}[/] [muted]· {MODE_HINT[m]}[/]")


@command("/tools", "list tools available to the model", group="Tools & skills")
def tools(repl: Repl, arg: str) -> None:
    table = _grid()
    for tool in repl.session.tools.available(repl.session.mode):
        badge = "[subtle]read-only[/]" if tool.read_only else "[warn]asks[/]" if tool.needs_permission else ""
        table.add_row(f"  [bold default]{tool.name}[/]", f"{_short(tool.description, 64)}  {badge}")
    repl.console.print(table)


@command("/skills", "list skills", group="Tools & skills")
def skills(repl: Repl, arg: str) -> None:
    found = repl.session.skills.list()
    if not found:
        repl.console.print(
            "  [muted]No skills yet. Add one at[/] ~/.tinyharness/skills/<name>/SKILL.md "
            "[muted]or run[/] tinyharness --import-skills"
        )
        return
    table = _grid()
    for skill in found:
        table.add_row(f"  [bold default]/{skill.name}[/]", _short(skill.description))
    repl.console.print(table)
