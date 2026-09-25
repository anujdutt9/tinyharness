"""`tinyharness` command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tinyharness import __version__
from tinyharness.cli import setup
from tinyharness.cli.render import Renderer
from tinyharness.cli.repl import Repl, report_error
from tinyharness.cli.theme import BRAND, LOGO, MODE_HINT, MODE_LABEL, MODE_STYLE, THEME, gradient
from tinyharness.permissions import PermissionMode
from tinyharness.session import MissingAPIKeyError, Session, SessionOptions
from tinyharness.skills import import_skills


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinyharness", description="A small, security-first coding agent.")
    p.add_argument("-p", "--prompt", help="run a single prompt and exit")
    p.add_argument("--model", help="model to use, optionally as provider:model (e.g. openai:gpt-6-sol)")
    p.add_argument("--provider", help="provider to use, e.g. anthropic, openai, openrouter")
    p.add_argument("--mode", choices=[m.value for m in PermissionMode], help="permission mode")
    p.add_argument("--cwd", type=Path, help="working directory (default: current directory)")
    p.add_argument("--config", type=Path, help="config file (default: ~/.config/tinyharness/config.toml)")
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


def print_banner(console: Console, session: Session) -> None:
    cfg = session.config
    mode = session.mode
    header = Text.assemble((f"{LOGO} ", BRAND), gradient("tinyharness"), (f"  v{__version__}", "subtle"))
    details = Table.grid(padding=(0, 2))
    details.add_column(style="muted")
    details.add_column()
    details.add_row(
        "model",
        Text.assemble((cfg.model_info.display, "bold"), (f"  via {cfg.provider_info.label}", "muted")),
    )
    details.add_row("project", _home_relative(cfg.working_dir))
    details.add_row(
        "mode",
        Text.assemble((MODE_LABEL[mode], f"bold {MODE_STYLE[mode]}"), (f" · {MODE_HINT[mode]}", "muted")),
    )
    if n := len(session.skills):
        details.add_row("skills", f"{n} available [muted]· /skills[/]")
    body = Group(header, Text("a small, hackable coding agent", style="muted"), Text(), details)
    console.print()
    console.print(Panel(body, box=box.ROUNDED, border_style="subtle", padding=(1, 2), expand=False))
    console.print(
        "  [key]/[/] [muted]commands[/]  [subtle]·[/]  [key]shift+tab[/] [muted]mode[/]  [subtle]·[/]  "
        "[key]ctrl+c[/] [muted]interrupt[/]  [subtle]·[/]  [key]ctrl+d[/] [muted]exit[/]\n"
    )


def _home_relative(path: Path | str, limit: int = 60) -> str:
    """~/code/app, or …/last/parts when the path is too long for the banner."""
    path = Path(path)
    try:
        text = f"~/{path.relative_to(Path.home())}"
    except ValueError:
        text = str(path)
    if len(text) <= limit:
        return text
    parts = path.parts
    for i in range(1, len(parts)):
        short = "…/" + "/".join(parts[i:])
        if len(short) <= limit:
            return short
    return "…" + text[-(limit - 1) :]


async def run(args: argparse.Namespace, console: Console) -> None:
    renderer = Renderer(console)
    options = SessionOptions(
        config_path=args.config,
        working_dir=args.cwd,
        provider=args.provider,
        model=args.model,
        permission_mode=args.mode,
        permission_callback=renderer.ask_permission,
        expose_skill_tool=args.expose_skills,
    )
    session = await open_session(options, console, interactive=not args.prompt and sys.stdin.isatty())
    if session is None:
        return
    renderer.root = f"{session.workspace.root}/"
    try:
        repl = Repl(session, console, renderer)
        if args.prompt:
            await repl.run_turn(session.run(args.prompt))
            return
        print_banner(console, session)
        await repl.loop()
    finally:
        session.close()


async def open_session(options: SessionOptions, console: Console, *, interactive: bool) -> Session | None:
    """Start a session; with no API key, walk the user through connecting a provider."""
    try:
        return Session(options)
    except MissingAPIKeyError:
        if not interactive:
            raise
    provider = await setup.first_run(console, options.resolve_config())
    if provider is None:
        console.print("  [muted]No provider connected. Run[/] tinyharness [muted]again when you're ready.[/]")
        return None
    options.provider = provider
    if options.model and ":" not in options.model:
        options.model = None  # a bare model id belonged to the provider without a key
    return Session(options)


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
