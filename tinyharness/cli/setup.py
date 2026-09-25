"""What a new user sees when no API key is set: where to get one and where it goes.

tinyharness never asks for a key itself; keys live in ~/.config/tinyharness/.env
(created here with a line per provider) or in the shell environment.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.text import Text

from tinyharness.cli.theme import LOGO, gradient
from tinyharness.config import Config, write_env_template
from tinyharness.paths import pretty_path


def print_key_guide(console: Console, config: Config) -> None:
    env_file = write_env_template(config.providers)
    first = next(iter(config.providers.values()))

    console.print()
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), gradient("Welcome to tinyharness")))
    console.print("\n  To get started, add an API key for [bold]one[/] of these providers:\n")

    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column(style="key")
    table.add_column(style="muted")
    for p in config.providers.values():
        table.add_row(f"    {p.label}", p.api_key_env, p.key_url)
    console.print(table)

    console.print("\n  Then either put it in your key file (already created for you):\n")
    console.print(f"    [path]{pretty_path(env_file)}[/]")
    console.print(f"    [muted]uncomment a line, e.g.[/] {first.api_key_env}=<your key>\n")
    console.print("  or export it in your shell:\n")
    console.print(f"    export {first.api_key_env}=<your key>\n")
    console.print(
        "  and run [bold]tinyharness[/] again. [muted]Tip: OpenRouter gives you models from every "
        "provider with a single key.[/]\n"
    )
