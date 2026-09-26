"""Slash commands for the REPL.

Adding a command is one decorated function:

    @command("/hello", "say hello", group="Session")
    def hello(repl: Repl, arg: str) -> None:
        repl.console.print("hi")

Commands that take a choice (/model, /mode, /provider) open a picker when run
without an argument.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from tinyharness.cli.theme import MODE_HINT, MODE_LABEL, MODE_STYLE, percent
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import RECENT_MODELS, forget_model, recent_models, remember_model
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
    """`message` may use rich markup. No automatic highlighting: it would color the
    digits in names like "Qwen3.8"."""
    repl.console.print(f"  [ok]✔[/] {message}", highlight=False)


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
        ("tab", "accept the gray suggestion or the highlighted command"),
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
    used = count_message_tokens(repl.session.history.to_dicts())
    window = repl.session.context_window
    fraction = min(used / window, 1.0)
    width = 30
    filled = round(fraction * width)
    color = "ok" if fraction < 0.6 else "warn" if fraction < 0.85 else "err"
    bar = Text("  ").append("█" * filled, style=color).append("░" * (width - filled), style="subtle")
    bar.append(f"  {percent(fraction)}", style="bold")
    repl.console.print(bar)
    n = len(repl.session.history.messages)
    repl.console.print(f"  [muted]{used:,} of {window:,} tokens · {n} messages[/]")


@command("/compact", "summarize the conversation so far to free up context", group="Session")
async def compact(repl: Repl, arg: str) -> None:
    session = repl.session
    before = count_message_tokens(session.history.to_dicts())
    try:
        with repl.console.status("[muted]Summarizing the conversation…[/]", spinner="dots"):
            done = await session.compact()
    except Exception as e:  # e.g. the model is rate-limited; the conversation is untouched
        _fail(repl, f"Couldn't compact: {e}")
        return
    if not done:
        repl.console.print("  [muted]Nothing to compact yet: the conversation is already short.[/]")
        return
    after = count_message_tokens(session.history.to_dicts())
    repl._measure_context()
    _ok(
        repl,
        f"Compacted: {before:,} → {after:,} tokens [muted]· now {percent(repl.context_used)} of context[/]",
    )


@command("/cwd", "show the working directory", group="Session")
def cwd(repl: Repl, arg: str) -> None:
    repl.console.print(f"  {repl.session.workspace.root}")


# --------------------------------------------------------------------- model & provider


_SEARCH = "__search__"


@command(
    "/model", "pick a model from any provider you have a key for", group="Model", usage="[provider:]model"
)
async def model(repl: Repl, arg: str) -> None:
    repl.session.reload_keys()
    cfg = repl.session.config
    current = f"{cfg.provider}:{cfg.model}"
    recent = [current, *(r for r in recent_models() if r != current and cfg.has_key(r.partition(":")[0]))]
    spec = arg
    if not spec:
        options = [(r, _model_label(repl, r)) for r in recent[:RECENT_MODELS]]
        options.append((_SEARCH, "Search all models…"))
        spec = await pick("Choose a model", options, current) or ""
    if spec == _SEARCH:
        spec = await _search_models(repl)
    if spec:
        await _switch(repl, spec)


def _model_label(repl: Repl, spec: str) -> str:
    provider, _, model_id = spec.partition(":")
    info = repl.session.client_for(provider).model_info(model_id)
    name = info.label if info and info.label else model_id
    return f"{name}  [{repl.session.config.providers[provider].label}]"


async def _search_models(repl: Repl) -> str:
    """Every model of every provider with a key, fetched live (cached for a day)."""
    cfg = repl.session.config
    keyed = [name for name in cfg.providers if cfg.has_key(name)]
    with repl.console.status("[muted]Loading model lists…[/]", spinner="dots"):
        results = await asyncio.gather(
            *(repl.session.client_for(name).list_models() for name in keyed), return_exceptions=True
        )
    specs: dict[str, str] = {}
    for name, result in zip(keyed, results, strict=True):
        label = cfg.providers[name].label
        if isinstance(result, BaseException):
            repl.console.print(f"  [warn]Couldn't list {label} models ({type(result).__name__}).[/]")
            continue
        for m in result:
            context = f" · {m.context // 1000:,}k context" if m.context else ""
            specs[f"{name}:{m.id}"] = f"{m.label or label}{context}"
    hint = (
        f"type to search {len(specs):,} models"
        if specs
        else "provider:model, e.g. " + f"{cfg.provider}:{cfg.model}"
    )
    return await ask("Model", choices=list(specs), meta=specs, hint=hint)


@command("/provider", "pick a provider (uses its default model)", group="Model", usage="[name]")
async def provider(repl: Repl, arg: str) -> None:
    repl.session.reload_keys()
    cfg = repl.session.config
    name = arg or await pick("Choose a provider", _provider_options(repl), cfg.provider)
    if name:
        await _switch(repl, f"{name}:")


def _provider_options(repl: Repl) -> list[tuple[str, str]]:
    cfg = repl.session.config
    width = max(len(p.label) for p in cfg.providers.values())
    return [
        (name, f"{p.label:<{width}}  " + ("✔ key set" if cfg.has_key(name) else f"needs {p.api_key_env}"))
        for name, p in cfg.providers.items()
    ]


async def _switch(repl: Repl, spec: str) -> None:
    """Check the model answers, then switch to it. `spec` is "provider:model" ("provider:" = its default)."""
    session = repl.session
    try:
        with repl.console.status("[muted]Checking the model…[/]", spinner="dots"):
            check = await session.check_model(spec)
        if check.model_unusable:
            forget_model(spec)
            _fail(repl, f"Can't use that model: {check.message}")
            repl.console.print(f"  [muted]Still using {escape(session.model_label)}.[/]")
            return
        session.set_model(spec)
    except (ValueError, MissingAPIKeyError) as e:
        _fail(repl, str(e))
        return
    remember_model(session.config.provider, session.config.model)
    repl._measure_context()  # the same conversation, measured against the new model's window
    window = f"{session.context_window // 1000:,}k context"
    label = escape(session.model_label)
    _ok(repl, f"Using [bold]{label}[/] [muted]via {session.config.provider_info.label} · {window}[/]")
    if not check.ok:  # inconclusive (timeout, server error, ...): switched, but say so
        repl.console.print(f"  [warn]Couldn't confirm it responds ({escape(check.message)}).[/]")


@command("/whoami", "show provider, endpoint, model and API key source", group="Model")
def whoami(repl: Repl, arg: str) -> None:
    session, cfg = repl.session, repl.session.config
    table = _grid()
    for label, value in [
        ("provider", f"{cfg.provider_info.label} ({cfg.provider})"),
        ("api", f"{cfg.provider_info.api} · {session.llm.endpoint}"),
        ("model", cfg.model),
        ("context", f"{session.context_window:,} tokens"),
        ("effort", cfg.effort or "model default"),
        ("key from", "api_key (config)" if cfg.api_key else f"${cfg.api_key_env}"),
        ("key", redact(cfg.resolve_api_key())),
    ]:
        table.add_row(f"  {label}", escape(str(value)))
    repl.console.print(table)


@command("/ping", "check that the API key and model work", group="Model")
async def ping(repl: Repl, arg: str) -> None:
    with repl.console.status("[muted]Pinging…[/]", spinner="dots"):
        check = await repl.session.llm.ping()
    if check.ok:
        _ok(repl, f"{escape(repl.session.model_label)} answered · {check.message}")
    else:
        _fail(repl, f"{check.status or 'no answer'} — {check.message}")


_EFFORT_HELP = {
    "default": "whatever the model does on its own",
    "low": "quick answers, least reasoning",
    "medium": "balanced",
    "high": "more reasoning for harder problems",
    "max": "the most the model offers",
}


@command(
    "/effort",
    "set how hard the model reasons",
    group="Model",
    usage="[default|low|medium|high|max]",
    aliases=("/think",),
)
async def effort(repl: Repl, arg: str) -> None:
    current = repl.session.config.effort or "default"
    choice = arg.lower() or await pick(
        "Reasoning effort", [(k, f"{k:<8} {v}") for k, v in _EFFORT_HELP.items()], current
    )
    if not choice:
        return
    if choice not in _EFFORT_HELP:
        _fail(repl, f"usage: /effort {'|'.join(_EFFORT_HELP)}")
        return
    repl.session.set_effort(None if choice == "default" else choice)  # type: ignore[arg-type]
    _ok(repl, f"Reasoning effort set to [bold]{choice}[/]")


# --------------------------------------------------------------------- permissions, tools, skills


@command("/mode", "pick the permission mode", group="Tools & skills", usage="[mode]")
async def mode(repl: Repl, arg: str) -> None:
    if not arg:
        options = [(m, f"{MODE_LABEL[m]:<13} {MODE_HINT[m]}") for m in PermissionMode]
        choice = await pick("Permission mode", options, repl.session.mode)
        if choice is None:
            return
        arg = choice
    if arg == PermissionMode.AUTO and not await repl.enter_auto_mode():
        repl.console.print(f"  [muted]Still in {MODE_LABEL[repl.session.mode]} mode.[/]")
        return
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
