"""`tinyharness` command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tinyharness import __version__
from tinyharness.agent.history import Message
from tinyharness.cli.render import Renderer
from tinyharness.cli.repl import Repl, report_error
from tinyharness.cli.setup import print_key_guide, run_setup
from tinyharness.cli.theme import BRAND, LOGO, MODE_HINT, MODE_LABEL, MODE_STYLE, THEME, gradient
from tinyharness.cli.widgets import pick
from tinyharness.config import note_launch, remember_model
from tinyharness.paths import pretty_path, source_checkout
from tinyharness.permissions import PermissionMode
from tinyharness.session import MissingAPIKeyError, Session, SessionOptions, list_sessions
from tinyharness.skills import import_skills
from tinyharness.trust import TrustStore, scan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinyharness", description="A small, security-first coding agent.")
    p.add_argument("-p", "--prompt", help="run a single prompt and exit")
    p.add_argument("--model", help="model to use, optionally as provider:model (e.g. openai:gpt-6-sol)")
    p.add_argument("--provider", help="provider to use, e.g. anthropic, openai, openrouter")
    p.add_argument("--mode", choices=[m.value for m in PermissionMode], help="permission mode")
    p.add_argument(
        "-c",
        "--continue",
        dest="resume_latest",
        action="store_true",
        help="resume the latest conversation here",
    )
    p.add_argument(
        "-r", "--resume", nargs="?", const="", metavar="ID", help="resume a conversation (picker if no ID)"
    )
    p.add_argument("--cwd", type=Path, help="working directory (default: current directory)")
    p.add_argument("--config", type=Path, help="config file (default: ~/.tinyharness/config.toml)")
    p.add_argument("--setup", action="store_true", help="choose a provider and add or replace its API key")
    p.add_argument("--expose-skills", action="store_true", help="let the model invoke skills on its own")
    p.add_argument("--debug", action="store_true", help="verbose logging and full tracebacks")
    p.add_argument("--version", action="version", version=f"tinyharness {__version__}")

    group = p.add_argument_group("skill import")
    group.add_argument(
        "--import-skills",
        nargs="?",
        const="",
        metavar="PATH",
        help="copy skills from PATH (default ~/.claude/skills) into ~/.tinyharness/skills, then exit",
    )
    group.add_argument("--symlink", action="store_true", help="with --import-skills: symlink instead of copy")
    group.add_argument("--force", action="store_true", help="with --import-skills: overwrite existing skills")
    return p


def run_import(console: Console, source: str, symlink: bool, force: bool) -> None:
    results = import_skills(Path(source).expanduser() if source else None, symlink=symlink, force=force)
    if not results:
        console.print(f"No skills found at {source or '~/.claude/skills'}")
    icons = {"copied": "[ok]✔[/]", "linked": "[ok]↪[/]", "skipped": "[subtle]·[/]", "failed": "[err]✗[/]"}
    for r in results:
        detail = f"  [muted]({r.detail})[/]" if r.detail else ""
        console.print(f"  {icons[r.action]} {r.name:<30} [muted]{r.action}[/]{detail}")


KEYS = (
    "  [key]/[/] [muted]commands[/]  [subtle]·[/]  [key]shift+tab[/] [muted]mode[/]  [subtle]·[/]  "
    "[key]ctrl+o[/] [muted]last output[/]  [subtle]·[/]  [key]ctrl+c[/] [muted]interrupt[/]  [subtle]·[/]  "
    "[key]ctrl+d[/] [muted]exit[/]\n"
)


def print_banner(console: Console, session: Session, *, compact: bool = False) -> None:
    """The full card the first time; after that, one line (plus anything that needs attention)."""
    cfg = session.config
    mode = session.mode
    if compact:
        line = Text.assemble(
            (f"{LOGO} ", BRAND),
            gradient("tinyharness"),
            (f" v{__version__}", "subtle"),
            ("  ·  ", "subtle"),
            (session.model_label, "bold"),
            (f" via {cfg.provider_info.label}", "muted"),
            ("  ·  ", "subtle"),
            (pretty_path(cfg.working_dir, limit=40), "muted"),
            ("  ·  ", "subtle"),
            (MODE_LABEL[mode], f"bold {MODE_STYLE[mode]}"),
        )
        console.print()
        console.print(Panel(line, box=box.ROUNDED, border_style="subtle", padding=(0, 1), expand=False))
        for warning in _attention(session):
            console.print(Text("  ").append_text(warning))
        console.print(KEYS)
        return
    header = Text.assemble((f"{LOGO} ", BRAND), gradient("tinyharness"), (f"  v{__version__}", "subtle"))
    details = Table.grid(padding=(0, 2))
    details.add_column(style="muted")
    details.add_column()
    details.add_row(
        "model",
        Text.assemble((session.model_label, "bold"), (f"  via {cfg.provider_info.label}", "muted")),
    )
    details.add_row("project", pretty_path(cfg.working_dir, limit=60))
    details.add_row(
        "mode",
        Text.assemble((MODE_LABEL[mode], f"bold {MODE_STYLE[mode]}"), (f" · {MODE_HINT[mode]}", "muted")),
    )
    if session.sandbox is not None:
        net = "localhost only" if cfg.sandbox.allow_localhost else "no network"
        details.add_row(
            "sandbox",
            Text.assemble((session.sandbox.name, "ok"), (f" · project writes only · {net}", "muted")),
        )
    elif "Bash" in session.tools.names():
        why = "disabled in config" if not cfg.sandbox.enabled else "not available on this system"
        details.add_row(
            "sandbox", Text.assemble(("off", "warn"), (f" · {why} · every command asks first", "muted"))
        )
    if session.project_code.needs_trust and not session.project_trusted:
        n = len(session.project_code.hooks) + len(session.project_code.plugins)
        details.add_row(
            "project",
            Text.assemble(("hooks/plugins off", "warn"), (f" · {n} not loaded, folder not trusted", "muted")),
        )
    if n := len(session.skills):
        details.add_row("skills", f"{n} available [muted]· /skills[/]")
    body = Group(header, Text("a small, hackable coding agent", style="muted"), Text(), details)
    console.print()
    console.print(Panel(body, box=box.ROUNDED, border_style="subtle", padding=(1, 2), expand=False))
    console.print(KEYS)


def _attention(session: Session) -> list[Text]:
    """What the compact banner still has to say: things that are off."""
    cfg, out = session.config, []
    if session.sandbox is None and "Bash" in session.tools.names():
        why = "disabled in config" if not cfg.sandbox.enabled else "not available on this system"
        out.append(
            Text.assemble(("⚠ sandbox off", "warn"), (f" · {why} · every command asks first", "muted"))
        )
    if session.project_code.needs_trust and not session.project_trusted:
        out.append(
            Text.assemble(("⚠ this folder's hooks and plugins are off", "warn"), (" · not trusted", "muted"))
        )
    return out


async def run(args: argparse.Namespace, console: Console) -> None:
    renderer = Renderer(console)
    options = SessionOptions(
        config_path=args.config,
        working_dir=args.cwd,
        provider=args.provider,
        model=args.model,
        permission_mode=args.mode,
        permission_callback=renderer.ask_permission,
        on_event=renderer.handle,
        expose_skill_tool=args.expose_skills,
        use_last_model=True,
    )
    interactive = not args.prompt and sys.stdin.isatty()
    if interactive and not await first_run_setup(options, console, force=args.setup):
        return
    if interactive:
        options = await confirm_trust(options, console)
    if args.resume_latest or args.resume is not None:
        which = "latest" if args.resume_latest else args.resume or await pick_session(options, console)
        if which is None:
            return
        options = replace(options, resume=which)
    session = open_session(options, console, interactive=interactive)
    if session is None:
        return
    renderer.root = f"{session.workspace.root}/"
    renderer.meter = session.ctx.usage
    renderer.show_thinking = session.config.show_thinking
    try:
        repl = Repl(session, console, renderer)
        if args.prompt:
            if session.project_code.needs_trust and not session.project_trusted:
                why = session.project_code.too_large or (
                    "it isn't trusted yet. Run tinyharness here once to trust it"
                )
                console.print(Text(f"This folder's hooks and plugins weren't loaded: {why}.", style="warn"))
            await repl.run_turn(session.run(args.prompt))
            return
        print_banner(console, session, compact=note_launch() > 1)
        if session.resume_report is not None:
            print_resumed(console, session)
        await repl.loop()
    finally:
        session.close()


async def confirm_trust(options: SessionOptions, console: Console) -> SessionOptions:
    """Ask before loading a project's own hooks and plugins, which run code outside the
    sandbox. Only projects that have any ask, and only once (again if they change)."""
    code = scan(options.resolve_config().working_dir)
    store = TrustStore()
    if store.trusts(code):
        return options
    if code.too_large:
        console.print(
            Text.assemble(
                ("\n⚠ not loading this folder's hooks and plugins: ", "warn"),
                (
                    f"{code.too_large}. Past that, tinyharness can't tell if they change later, so it ",
                    "muted",
                ),
                ("won't offer to trust them. Shrink or remove the plugin to use it here.", "muted"),
            )
        )
        return replace(options, trust_project=False)
    why = "changed since you trusted it" if store.trusted_before(code) else "can run code on your machine"
    items = Text()
    for line in code.describe()[:8]:
        items.append(f"\n  • {line[:100]}", style="muted")
    if len(code.describe()) > 8:
        items.append(f"\n  • … {len(code.describe()) - 8} more", style="muted")
    console.print()
    console.print(
        Panel(
            Group(
                Text.assemble(("This folder's .tinyharness/ settings ", ""), (why, "bold"), (":", "")),
                items,
                Text("\nThey run outside the sandbox. Everything else works either way.", style="muted"),
            ),
            title=Text(pretty_path(code.root, limit=60), style="bold"),
            title_align="left",
            box=box.ROUNDED,
            border_style="warn",
            padding=(0, 1),
        )
    )
    choice = await pick(
        "Trust this folder?",
        [(True, "Yes, load them (remembered for this folder)"), (False, "No, continue without them")],
        mark_current=False,
    )
    if choice:
        store.trust(code)
    return replace(options, trust_project=bool(choice))


async def pick_session(options: SessionOptions, console: Console) -> str | None:
    """Choose a saved conversation from this project."""
    sessions = list_sessions(options)
    if not sessions:
        console.print("\n  [muted]No saved conversations in this project yet.[/]\n")
        return None
    labels = [(s.id, f"{s.title}  [{_ago(s.updated)} · {s.messages} messages]") for s in sessions]
    return await pick("Resume which conversation?", labels, mark_current=False)


def print_resumed(console: Console, session: Session) -> None:
    """Where the conversation left off, and what changed since."""
    report = session.resume_report
    assert report is not None
    last: dict[str, Message] = {
        m.role: m for m in session.history.messages if m.role in ("user", "assistant") and m.content
    }
    console.print(f"  [brand]↻[/] [bold]Resumed[/] [muted]· {report.summary()}[/]")
    for role, prefix in (("user", "❯"), ("assistant", "●")):
        if (m := last.get(role)) is not None:
            text = " ".join((m.content or "").rpartition("</session-resumed>")[2].split())
            console.print(f"  [subtle]{prefix}[/] [muted]{text[:160]}{'…' if len(text) > 160 else ''}[/]")
    if report.anything_changed:
        console.print("  [warn]The model will be told what changed before your next message.[/]")
    console.print()


def _ago(when: datetime) -> str:
    seconds = (datetime.now() - when).total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


async def first_run_setup(options: SessionOptions, console: Console, *, force: bool = False) -> bool:
    """For an installed copy, help a new user pick a provider and add its key, rather
    than starting with whichever key happens to be set. A git checkout (development)
    starts as before unless `force`. Returns False if the user backed out."""
    if not force and source_checkout() is not None:
        return True
    config = options.resolve_config()
    if not (force or config.auto_selected or not config.has_key(config.provider)):
        return True
    name = await run_setup(console, config)
    if name is None:
        console.print("\n  [muted]Setup cancelled. Run[/] tinyharness [muted]again any time.[/]\n")
        return False
    config.use_provider(name)
    remember_model(config.provider, config.model)  # next time, start with it
    return True


def open_session(options: SessionOptions, console: Console, *, interactive: bool) -> Session | None:
    """Start a session. With no API key at all, explain how to add one instead."""
    try:
        return Session(options)
    except MissingAPIKeyError:
        config = options.resolve_config()
        if not interactive or any(config.has_key(name) for name in config.providers):
            raise  # a key exists, just not for the chosen provider: the error says which
        print_key_guide(console, config)
        return None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    console = Console(theme=THEME)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.import_skills is not None:
        run_import(console, args.import_skills, args.symlink, args.force)
        return
    try:
        asyncio.run(run(args, console))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as e:  # startup errors (missing API key, bad config)
        report_error(console, e)
        raise SystemExit(1) from e
