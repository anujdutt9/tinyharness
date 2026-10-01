"""What a new user sees when no API key is set: where to get one and where it goes.

Keys are pasted here, checked with the provider and kept in ~/.tinyharness/auth.json
(or set in the shell). An installed copy (from PyPI) walks a new user through
`run_setup` on first run; `/key`, `/provider` and `/model` use `add_key` later;
`print_key_guide` is the fallback when nothing can be asked.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from tinyharness.cli.theme import LOGO, gradient
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import Config, save_key
from tinyharness.llm.client import LLMClient, Ping
from tinyharness.llm.providers import Provider
from tinyharness.paths import pretty_path

MAX_KEY_ATTEMPTS = 3


async def run_setup(console: Console, config: Config) -> str | None:
    """Ask which provider to use and make sure it has a working key, saving a pasted
    one to the key file. Returns the provider's name, or None if the user backed out."""
    console.print()
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), gradient("Welcome to tinyharness")))
    console.print("\n  Pick the model provider to use. [muted]You can change it later with /model.[/]\n")

    width = max(len(p.label) for p in config.providers.values())
    options = [
        (name, f"{p.label:<{width}}  " + ("✔ ready" if p.has_key else "add a key"))
        for name, p in config.providers.items()
    ]
    picked = await pick("Which provider?", options, config.provider, mark_current=False)
    if picked is None:
        return None
    info = config.providers[picked]
    check = await add_key(console, info, config.providers, info.api_key or "")
    if check is None:
        return None
    if check.ok:
        console.print(f"  [ok]✔[/] {info.label} is ready.\n")
    else:  # not a key problem (network, a busy server, ...): carry on, but say so
        console.print(f"  [warn]Couldn't confirm the key works ({escape(check.message)}).[/]\n")
    return picked


async def add_key(
    console: Console, info: Provider, providers: dict[str, Provider], key: str = ""
) -> Ping | None:
    """Make sure `info` has a working key: ask for one (unless `key` is given), check it
    with the provider, and save a pasted one to the key file, where it's used right away.
    Returns the check (which may be inconclusive, e.g. offline), or None if the user
    backed out or the provider rejected every key tried."""
    pasted = False
    if not info.needs_key:  # a local server: just check it answers
        with console.status(f"[muted]Checking {info.label}…[/]", spinner="dots"):
            return await LLMClient(info, "", info.default_model).ping()
    for _ in range(MAX_KEY_ATTEMPTS):
        if not key:
            if info.key_url:
                console.print(f"\n  Get your {info.label} key at [path]{info.key_url}[/]")
            key = await ask(f"Paste your {info.api_key_env}:", secret=True, hint="hidden as you type")
            if not key:
                return None
            pasted = True
        with console.status(f"[muted]Checking the key with {info.label}…[/]", spinner="dots"):
            check = await LLMClient(info, key, info.default_model).ping()
        if not check.bad_key:
            break
        which = "that key" if pasted else f"the {info.api_key_env} that is set"
        console.print(f"  [err]✗[/] {info.label} rejected {which}: [muted]{escape(check.message)}[/]")
        key = ""
    else:
        return None
    if pasted:
        path = save_key(info.name, key.strip(), providers)
        console.print(f"  [ok]✔[/] Saved to [path]{pretty_path(path)}[/] [muted](only you can read it)[/]")
    return check


def print_key_guide(console: Console, config: Config) -> None:
    first = next(iter(config.providers.values()))

    console.print()
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), gradient("Welcome to tinyharness")))
    console.print("\n  To get started, you need an API key for [bold]one[/] of these providers:\n")

    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column(style="muted")
    for p in config.providers.values():
        table.add_row(f"    {p.label}", p.key_url)
    console.print(table)

    console.print(
        "\n  Then run [bold]tinyharness --setup[/] and paste it when asked, or set it in your shell:\n"
    )
    console.print(f"    export {first.api_key_env}=<your key>\n")
    console.print("  [muted]Tip: OpenRouter gives you models from every provider with a single key.[/]\n")
