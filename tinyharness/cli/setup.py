"""Connecting a provider: pick one, paste its API key, done.

Runs automatically when no key is found at startup, and on demand with /login.
Keys are saved to ~/.config/tinyharness/.env (readable only by you), never to the
config file.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

from tinyharness.cli.theme import LOGO, gradient
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import Config, save_api_key


def provider_options(config: Config) -> list[tuple[str, str]]:
    width = max(len(p.label) for p in config.providers.values())
    return [
        (name, f"{p.label:<{width}}  " + ("✔ key set" if config.has_key(name) else f"needs ${p.api_key_env}"))
        for name, p in config.providers.items()
    ]


async def connect(console: Console, config: Config, provider: str | None = None) -> str | None:
    """Ask for a provider's API key and save it. Returns the provider name, or None if cancelled."""
    if provider is None:
        provider = await pick("Which provider do you want to use?", provider_options(config), config.provider)
        if provider is None:
            return None
    info = config.providers[provider]

    console.print()
    if info.key_url:
        console.print(f"  Get an API key for {info.label} at [link={info.key_url}]{info.key_url}[/link]")
    key = await ask(f"Paste your {info.label} API key", secret=True, hint="input is hidden")
    if not key:
        console.print("  [muted]Cancelled.[/]")
        return None
    path = save_api_key(info.api_key_env, key)
    console.print(f"  [ok]✔[/] Saved as [bold]{info.api_key_env}[/] in [muted]{path}[/]")
    return provider


async def first_run(console: Console, config: Config) -> str | None:
    """Welcome screen for someone with no API key yet."""
    console.print()
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), gradient("Welcome to tinyharness")))
    console.print(
        "\n  tinyharness works with any of the providers below. Pick one and paste an API key;\n"
        "  it's saved on this machine only. You can add more later with [key]/login[/].\n"
    )
    return await connect(console, config)
