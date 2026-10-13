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
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from tinyharness.cli.render import dollars, tokens
from tinyharness.cli.setup import add_key
from tinyharness.cli.theme import (
    LIGHT,
    MODE_HINT,
    MODE_LABEL,
    MODE_STYLE,
    THEME_SETTING,
    THEME_SOURCE,
    percent,
)
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import RECENT_MODELS, forget_model, recent_models, remember_model, save_theme
from tinyharness.paths import pretty_path
from tinyharness.permissions import PermissionMode
from tinyharness.sandbox import setup_hint
from tinyharness.security import redact
from tinyharness.session import MissingAPIKeyError

if TYPE_CHECKING:
    from tinyharness.cli.repl import Repl
    from tinyharness.llm.providers import Provider

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
    for ext in repl.session.extensions:
        for c in ext.commands:
            groups.setdefault("Extensions", []).append(
                Command(f"/{c.name}", f"{c.help} [subtle]({ext.name})[/]", help_, c.usage, "Extensions")
            )

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
    if n:
        repl.console.print("  [muted]The previous conversation is saved: tinyharness -r picks it.[/]")
    meter = repl.session.ctx.usage
    if meter.calls:  # the status bar's totals stay: they're the session's, not the conversation's
        spent = f"{tokens(meter.total.output)} out" + (
            f" · {dollars(meter.cost)}" if meter.cost is not None else ""
        )
        repl.console.print(f"  [muted]Session so far: {spent} (/cost for details)[/]")


@command("/context", "show how much of the context window is used", group="Session")
def context(repl: Repl, arg: str) -> None:
    used = repl.session.compactor.used(repl.session.history)
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


@command("/cost", "show tokens and cost for this session", group="Session")
def cost(repl: Repl, arg: str) -> None:
    meter = repl.session.ctx.usage
    if not meter.calls:
        repl.console.print("  [muted]No model calls yet.[/]")
        return
    total = meter.total
    sent = total.input + total.cached
    repl.console.print("  [bold]This session[/] [muted](since launch; /clear doesn't reset it)[/]")
    rows = _grid(first_width=16)
    rows.add_row("model calls", f"{meter.calls:,} [muted](subagents included)[/]")
    rows.add_row(
        "largest context", f"{meter.largest_context():,} tokens [muted](the most any one call sent)[/]"
    )
    rows.add_row("output", f"{total.output:,} tokens [muted](what the model wrote, reasoning included)[/]")
    cached = f" [muted]· {total.cached / sent:.0%} of it cache reads, billed at a fraction[/]" if sent else ""
    rows.add_row("input, all calls", f"{sent:,} tokens{cached}")
    if (last := meter.last) is not None and (last_in := last.input + last.cached):
        rows.add_row(
            "latest call", f"{last.cached / last_in:.0%} of its {last_in:,} input tokens read from the cache"
        )
    rows.add_row(
        "fixed prefix",
        f"{repl.session.prefix_tokens():,} tokens [muted](system prompt and tools, sent with every call)[/]",
    )
    if meter.cost is not None:
        note = (
            f" [muted]· {meter.unpriced} call(s) had no price and aren't included[/]"
            if meter.unpriced
            else ""
        )
        rows.add_row("cost", f"{dollars(meter.cost)}{note}")
    else:
        rows.add_row("cost", "[muted]unknown: no price for this model (set one under \\[pricing])[/]")
    if (since := meter.since_clear()) is not None:
        spent, calls = since
        line = f"{calls:,} call{'s' if calls != 1 else ''} · {tokens(spent.output)} out"
        if spent.cost is not None:
            line += f" · {dollars(spent.cost)}"
        rows.add_row("since /clear", line)
    repl.console.print(Padding(rows, (0, 0, 0, 2)))  # under the heading
    repl.console.print(
        "  [muted]Every call re-sends the whole conversation, so input adds up with each step; "
        "most of it is cached.[/]"
    )


@command(
    "/btw",
    "ask a side question; the answer isn't added to the conversation",
    group="Session",
    usage="<question>",
)
def btw(repl: Repl, arg: str) -> None:
    if not arg:
        repl.console.print(
            "  [muted]Usage: /btw <question>. It's answered from the conversation so far, without tools,\n"
            "  and works while the agent is busy; neither the question nor the answer is kept.[/]"
        )
        return
    repl.ask_aside(arg)


@command("/compact", "summarize the conversation so far to free up context", group="Session")
async def compact(repl: Repl, arg: str) -> None:
    session = repl.session
    before = session.compactor.used(session.history)
    try:
        with repl.console.status("[muted]Summarizing the conversation…[/]", spinner="dots"):
            done = await session.compact()
    except Exception as e:  # e.g. the model is rate-limited; the conversation is untouched
        _fail(repl, f"Couldn't compact: {e}")
        return
    if not done:
        repl.console.print("  [muted]Nothing to compact yet: the conversation is already short.[/]")
        return
    after = session.compactor.used(session.history)
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
        (name, f"{p.label:<{width}}  " + ("✔ key set" if cfg.has_key(name) else "add a key"))
        for name, p in cfg.providers.items()
    ]


async def _switch(repl: Repl, spec: str, *, asked_for_key: bool = False) -> None:
    """Check the model answers, then switch to it. `spec` is "provider:model" ("provider:" = its default).
    A provider without a key asks for one first, rather than sending you to a file."""
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
    except MissingAPIKeyError as e:
        if not asked_for_key and await _ask_for_key(repl, e.provider):
            await _switch(repl, spec, asked_for_key=True)
            return
        _fail(repl, str(e))
        return
    except ValueError as e:
        _fail(repl, str(e))
        return
    remember_model(session.config.provider, session.config.model)
    repl._measure_context()  # the same conversation, measured against the new model's window
    window = f"{session.context_window // 1000:,}k context"
    label = escape(session.model_label)
    _ok(repl, f"Using [bold]{label}[/] [muted]via {session.config.provider_info.label} · {window}[/]")
    if not check.ok:  # inconclusive (timeout, server error, ...): switched, but say so
        repl.console.print(f"  [warn]Couldn't confirm it responds ({escape(check.message)}).[/]")


async def _ask_for_key(repl: Repl, info: Provider) -> bool:
    """Offer to add a provider's key here. True once a key is saved (or set in the shell)."""
    repl.console.print(
        f"\n  {info.label} needs an API key. [muted]Paste it below, or press Enter to skip.[/]"
    )
    check = await add_key(repl.console, info, repl.session.config.providers)
    if check is None:
        return False
    repl.session.reload_keys()
    return True


@command("/key", "add or replace a provider key saved in auth.json", group="Model", usage="[provider]")
async def key(repl: Repl, arg: str) -> None:
    session = repl.session
    session.reload_keys()
    cfg = session.config
    name = arg or await pick("Which provider's key?", _provider_options(repl), cfg.provider)
    if not name:
        return
    if name not in cfg.providers:
        _fail(repl, f"Unknown provider {name!r}. Choose one of: {', '.join(cfg.providers)}")
        return
    info = cfg.providers[name]
    if cfg.has_key(name):
        if info.api_key_source in ("environment", ".env"):
            repl.console.print(
                f"  [muted]{info.label} uses a key from {info.api_key_source}; auth.json can "
                "store a fallback but cannot override it.[/]"
            )
        else:
            repl.console.print(f"  [muted]{info.label} has a saved key; a new one replaces it.[/]")
    check = await add_key(repl.console, info, cfg.providers)
    if check is None:
        repl.console.print("  [muted]No change.[/]")
        return
    session.reload_keys()
    if name == cfg.provider:  # the running client has the old key: reconnect (conversation kept)
        session.set_model(f"{cfg.provider}:{cfg.model}")
    if info.api_key_source != "auth.json":
        _ok(repl, f"Saved {info.label} fallback; still using {info.api_key_source}.")
    elif check.ok:
        _ok(repl, f"{info.label} is ready.")
    else:
        repl.console.print(f"  [warn]Saved, but couldn't confirm it works ({escape(check.message)}).[/]")


@command("/whoami", "show provider, endpoint, model and API key source", group="Model")
def whoami(repl: Repl, arg: str) -> None:
    session, cfg = repl.session, repl.session.config
    table = _grid()
    key_source = {
        "environment": f"environment (${cfg.api_key_env})",
        ".env": "development .env",
        "auth.json": "~/.tinyharness/auth.json",
        "SDK override": "SessionOptions(api_key=...)",
    }.get(cfg.api_key_source, cfg.api_key_source)
    for label, value in [
        ("provider", f"{cfg.provider_info.label} ({cfg.provider})"),
        ("api", f"{cfg.provider_info.api} · {session.llm.endpoint}"),
        ("model", cfg.model),
        ("context", f"{session.context_window:,} tokens"),
        ("effort", cfg.effort or "model default"),
        (
            "key from",
            key_source or ("(none needed)" if not cfg.api_key_env else "(not set)"),
        ),
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
    "default": "the model's own choice (medium where reasoning is opt-in)",
    "none": "no reasoning: fastest",
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


@command("/theme", "light, dark, or follow the terminal", group="Model", usage="[system|light|dark]")
async def theme(repl: Repl, arg: str) -> None:
    detected = "light" if LIGHT else "dark"
    labels = {
        "system": f"follow the terminal (now: {detected})",
        "light": "light background",
        "dark": "dark background",
    }
    choice = arg.lower() or await pick(
        "Theme", [(k, f"{k:<7} {v}") for k, v in labels.items()], THEME_SETTING
    )
    if not choice:
        return
    if choice not in labels:
        _fail(repl, "usage: /theme system|light|dark")
        return
    save_theme(choice)
    _ok(repl, f"Theme set to [bold]{choice}[/] [muted]· takes effect the next time you start tinyharness[/]")
    if THEME_SOURCE in ("TINYHARNESS_THEME", "config"):
        where = "TINYHARNESS_THEME" if THEME_SOURCE == "TINYHARNESS_THEME" else "theme in your config file"
        repl.console.print(f"  [warn]{where} is set to {THEME_SETTING}, which takes priority.[/]")


@command(
    "/output",
    "print the last tool call's full output, a running subagent's calls, or an open /btw answer",
    group="Session",
)
def output(repl: Repl, arg: str) -> None:
    if not repl.show_aside():
        repl.show_last_output()


@command("/thinking", "show or hide the model's reasoning (also ctrl+t)", group="Model", usage="[on|off]")
def thinking(repl: Repl, arg: str) -> None:
    choice = arg.lower()
    if choice not in ("", "on", "off"):
        _fail(repl, "usage: /thinking on|off")
        return
    repl.show_thinking(choice == "on" if choice else not repl.renderer.show_thinking)


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


@command("/sandbox", "what sandboxed commands can write, read and reach", group="Tools & skills")
def sandbox(repl: Repl, arg: str) -> None:
    session = repl.session
    if session.sandbox is None:
        repl.console.print("  [warn]No sandbox[/] [muted]· every command asks before it runs[/]")
        repl.console.print(f"  [muted]Setup: {setup_hint()}.[/]")
        return
    policy = session.sandbox_policy
    table = _grid(12)
    table.add_row("  [bold default]writes[/]", ", ".join(pretty_path(p) for p in policy.write_roots[:4]))
    table.add_row("  [bold default]reads[/]", "anything except credential files (.env, keys, auth.json)")
    network = ", ".join(policy.allowed_hosts) or "none"
    table.add_row("  [bold default]network[/]", network)
    repl.console.print(table)
    repl.console.print(
        "  [muted]Bash cannot disable containment. A blocked exact host can be approved for "
        "one command or the session; configure stable destinations with \\[sandbox] allow_hosts.[/]"
    )
    if session.ctx.outside:
        repl.console.print(
            f"  [warn]Outside content in this conversation:[/] {', '.join(session.ctx.outside[:5])} "
            "[muted]· calls that could send data out ask first, until /clear[/]"
        )
    if session.proxy is not None and session.proxy.denied:
        repl.console.print(
            f"  [muted]Blocked this session:[/] {', '.join(dict.fromkeys(session.proxy.denied))}"
        )


@command(
    "/extensions",
    "list extensions, or turn one on or off",
    group="Tools & skills",
    usage="[on|off|allow name]",
)
def extensions(repl: Repl, arg: str) -> None:
    from tinyharness.ext import builtin_names

    session = repl.session
    action, _, name = arg.partition(" ")
    name = name.strip()
    if action == "allow" and name:
        try:
            session.grant_needs(name)
        except ValueError as e:
            _fail(repl, str(e))
            return
        _ok(repl, f"{name} can now do what it needs [muted](remembered until it changes)[/]")
        return
    if action in ("on", "off") and name:
        names = [n for n in session.config.extensions.enabled if n != name]
        if action == "on":
            names.append(name)
        session.set_extensions(names)
        if action == "on" and name not in [e.name for e in session.extensions]:
            error = next((e.error for e in session.extension_errors if e.name == name), "not loaded")
            _fail(repl, f"{name}: {error}")
            return
        _ok(
            repl,
            f"{name} turned {action} for this session [muted]· the next request rebuilds the "
            f"prompt cache once. To keep it, list it under \\[extensions] enabled in config.toml[/]",
        )
        return
    if arg:
        _fail(repl, "usage: /extensions [on|off|allow name]")
        return
    table = _grid()
    for ext in session.extensions:
        cost = session.extension_tokens(ext)
        what = ", ".join(t.name for t in ext.tools) or "no tools"
        needs = ""
        if ext.needs:
            state = (
                "[subtle]approved[/]"
                if ext.granted
                else f"[warn]not approved: /extensions allow {ext.name}[/]"
            )
            needs = f"  [muted]needs {', '.join(ext.needs)}[/] · {state}"
        table.add_row(
            f"  [bold default]{ext.name}[/]",
            f"{_short(ext.description or what, 56)}  [subtle]+{cost:,} tokens per request[/]{needs}",
        )
    for err in session.extension_errors:
        table.add_row(f"  [bold default]{err.name}[/]", f"[error]not loaded: {err.error}[/]")
    if session.extensions or session.extension_errors:
        repl.console.print(table)
    off = [n for n in builtin_names() if n not in session.config.extensions.enabled]
    if off:
        repl.console.print(f"  [muted]Built in, off: {', '.join(off)} · /extensions on <name>[/]")
    elif not session.extensions:
        repl.console.print("  [muted]No extensions. Add one at[/] ~/.tinyharness/extensions/<name>.py")


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
