"""Connect a model provider during first run or from an active session.

Each provider advertises the methods it supports. API keys remain the universal
fallback; supported providers can add a browser or device flow without changing
the setup UI. Secrets are kept in ~/.tinyharness/auth.json (or set in the shell).
"""

from __future__ import annotations

import webbrowser

import httpx
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from tinyharness.cli.theme import LOGO, wordmark
from tinyharness.cli.widgets import ask, pick
from tinyharness.config import (
    Config,
    chatgpt_profiles,
    ensure_chatgpt_host_id,
    save_chatgpt_profile,
    save_chatgpt_registration,
    save_key,
    save_provider_connection,
)
from tinyharness.llm.auth import (
    AuthenticationError,
    DeviceAuthorization,
    chatgpt_browser_login,
    exchange_openrouter_code,
    github_copilot_login,
    openrouter_authorization,
    openrouter_browser_login,
)
from tinyharness.llm.client import LLMClient, Ping
from tinyharness.llm.providers import Provider
from tinyharness.paths import pretty_path

MAX_KEY_ATTEMPTS = 3


async def run_setup(console: Console, config: Config) -> str | None:
    """Ask which provider to use and connect it with one of its supported methods."""
    console.print()
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), wordmark("Welcome to tinyharness")))
    console.print("\n  Choose where your models come from. [muted]You can change this later.[/]\n")

    width = max(len(p.label) for p in config.providers.values())
    options = [
        (name, f"{p.label:<{width}}  " + _connection_status(p)) for name, p in config.providers.items()
    ]
    picked = await pick("Which provider?", options, config.provider, mark_current=False)
    if picked is None:
        return None
    info = config.providers[picked]
    check = await connect_provider(console, info, config.providers, info.api_key or "")
    if check is None:
        return None
    if check.ok:
        console.print(f"  [ok]✔[/] {info.label} is ready.\n")
    else:  # not a key problem (network, a busy server, ...): carry on, but say so
        console.print(f"  [warn]Couldn't confirm the key works ({escape(check.message)}).[/]\n")
    return picked


async def connect_provider(
    console: Console, info: Provider, providers: dict[str, Provider], credential: str = ""
) -> Ping | None:
    """Connect one provider using an existing credential or a method the user selects."""
    if info.local:
        return await _connect_local(console, info)
    if info.endpoint_setup:
        return await _connect_endpoint(console, info)
    if info.auth_refresh == "chatgpt":
        return await _connect_chatgpt(console, info)
    if credential or not info.needs_key:
        return await add_key(console, info, providers, credential)
    method = info.auth_methods[0]
    if len(info.auth_methods) > 1:
        labels = {
            "openrouter": "Sign in with OpenRouter in your browser",
            "github_copilot": "Sign in with GitHub and use your Copilot subscription",
            "chatgpt": "Continue with ChatGPT",
            "api_key": "Paste an API key",
        }
        picked = await pick(
            f"Connect {info.label}",
            [(name, labels[name]) for name in info.auth_methods],
            mark_current=False,
        )
        if picked is None:
            return None
        method = picked
    if method == "api_key":
        return await add_key(console, info, providers)
    if method == "openrouter":
        return await _connect_openrouter(console, info, providers)
    if method == "github_copilot":
        return await _connect_github_copilot(console, info)
    if method == "chatgpt":
        return await _connect_chatgpt(console, info)
    raise ValueError(f"unknown authentication method {method!r}")


async def _connect_local(console: Console, info: Provider) -> Ping | None:
    return await _connect_endpoint(console, info)


async def _connect_endpoint(console: Console, info: Provider) -> Ping | None:
    current = "saved" if info.default_model else "standard"
    mode = await pick(
        f"Connect {info.label}",
        [
            ("current", f"Use {current} endpoint  · {info.base_url}"),
            ("custom", "Enter another endpoint"),
            ("credential", "Add, replace or remove the server key"),
        ],
        "current",
        mark_current=False,
    )
    if mode is None:
        return None
    key = info.api_key
    if mode == "custom":
        connection = await _ask_endpoint(console, info)
        if connection is None:
            return None
        info.base_url, key = connection
    elif mode == "credential":
        changed, key = await _ask_server_key(info)
        if not changed:
            return None
    console.print(f"\n  Looking for {info.label} at [path]{info.base_url}[/]…")
    try:
        models = await LLMClient(info, key, info.default_model).fetch_models()
    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
        console.print(f"  [warn]Couldn't connect there: [muted]{escape(str(error))}[/]")
        if mode == "custom":
            return None
        connection = await _ask_endpoint(console, info)
        if connection is None:
            return None
        info.base_url, key = connection
        try:
            models = await LLMClient(info, key, info.default_model).fetch_models()
        except (httpx.HTTPError, OSError, ValueError, KeyError) as retry_error:
            console.print(
                f"  [err]✗[/] Couldn't connect to {info.label}: [muted]{escape(str(retry_error))}[/]"
            )
            return None
    if not models:
        console.print(f"  [err]✗[/] {info.label} is running but reports no models.")
        return None
    model = models[0].id
    if len(models) > 1:
        picked = await pick(
            f"Which {info.label} model?",
            [(item.id, item.label or item.id) for item in models],
            mark_current=False,
        )
        if picked is None:
            return None
        model = picked
    info.default_model = model
    info.small_model = model
    client = LLMClient(info, key, model)
    with console.status(f"[muted]Checking {model}…[/]", spinner="dots"):
        check = await client.ping(model)
    if check.ok:
        path = save_provider_connection(info, key)
        console.print(f"  [ok]✔[/] Connected to {model} and saved to [path]{pretty_path(path)}[/].")
        return check
    console.print(f"  [err]✗[/] {info.label} couldn't run {model}: [muted]{escape(check.message)}[/]")
    return None


async def _ask_endpoint(console: Console, info: Provider) -> tuple[str, str] | None:
    endpoint = await ask(
        f"{info.label} endpoint:",
        hint=f"for example {info.base_url}",
    )
    if not endpoint:
        return None
    if not endpoint.startswith(("http://", "https://")):
        console.print("  [err]✗[/] The endpoint must start with http:// or https://")
        return None
    changed, key = await _ask_server_key(info)
    return (endpoint.rstrip("/"), key) if changed else None


async def _ask_server_key(info: Provider) -> tuple[bool, str]:
    if info.api_key:
        credential = await pick(
            "Server credential",
            [
                ("keep", "Keep the saved server key"),
                ("replace", "Replace the server key"),
                ("none", "Connect without a server key"),
            ],
            "keep",
            mark_current=False,
        )
        if credential is None:
            return False, ""
        if credential == "keep":
            key = info.api_key
        elif credential == "replace":
            key = await ask("New server key:", secret=True, hint="hidden as you type")
            if not key:
                return False, ""
        else:
            key = ""
    else:
        key = await ask("Server key (optional):", secret=True, hint="press Enter when none is required")
    return True, key


async def _connect_github_copilot(console: Console, info: Provider) -> Ping | None:
    def show_device(device: DeviceAuthorization) -> None:
        console.print(f"\n  Enter [bold]{device.user_code}[/] at [path]{device.verification_url}[/]")
        webbrowser.open(device.verification_url)

    try:
        with console.status("[muted]Waiting for GitHub authorization…[/]", spinner="dots"):
            token = await github_copilot_login(notify=show_device)
        client = LLMClient(info, token, info.default_model)
        with console.status("[muted]Loading your Copilot models…[/]", spinner="dots"):
            models = await client.fetch_models()
        if client.copilot_plan:
            console.print(f"  Plan: [bold]{client.copilot_plan}[/]")
        if not models:
            console.print("  [err]✗[/] GitHub Copilot reports no tool-capable models for this account.")
            return None
        # The setup client has no disk-backed model directory. Keep the live metadata on
        # this provider so its check uses the endpoint GitHub advertised for the model.
        info.models = models
        remaining = list(models)
        while remaining:
            model = remaining[0].id
            if len(remaining) > 1:
                picked = await pick(
                    "Which Copilot model?",
                    [(item.id, item.label or item.id) for item in remaining],
                    mark_current=False,
                )
                if picked is None:
                    return None
                model = picked
            info.default_model = model
            info.small_model = model
            client.model = model
            with console.status(f"[muted]Checking {model}…[/]", spinner="dots"):
                check = await client.ping(model)
            if check.ok or not check.model_unusable:
                break
            console.print(
                f"  [warn]{escape(model)} is listed by GitHub but is not available to this account. "
                "Choose another model.[/]"
            )
            remaining = [item for item in remaining if item.id != model]
    except (AuthenticationError, httpx.HTTPError) as error:
        console.print(f"  [err]✗[/] {escape(str(error))}")
        return None
    if not remaining or not check.ok:
        console.print(f"  [err]✗[/] GitHub Copilot couldn't run {model}: [muted]{escape(check.message)}[/]")
        return None
    path = save_provider_connection(info, token)
    console.print(f"  [ok]✔[/] Connected to {model} and saved to [path]{pretty_path(path)}[/].")
    return check


async def _connect_chatgpt(console: Console, info: Provider) -> Ping | None:
    profiles = chatgpt_profiles()
    existing = None
    if profiles:
        options = [
            (
                str(profile.get("client_id") or ""),
                f"Continue with {profile.get('email') or 'saved ChatGPT account'}",
            )
            for profile in profiles
            if profile.get("client_id")
        ]
        options.append(("new", "Use another ChatGPT account or workspace"))
        picked = await pick("ChatGPT account", options, options[0][0], mark_current=False)
        if picked is None:
            return None
        existing = next((profile for profile in profiles if profile.get("client_id") == picked), None)

    host_id = ensure_chatgpt_host_id()

    def remember_registration(client_id: str) -> None:
        save_chatgpt_registration(host_id, client_id)

    console.print("\n  Continue with ChatGPT in the browser that opens…")
    try:
        with console.status("[muted]Waiting for ChatGPT authorization…[/]", spinner="dots"):
            profile = await chatgpt_browser_login(
                host_id,
                existing=existing,
                on_registration=remember_registration,
            )
        # Keep the validated registration even if model discovery is temporarily down,
        # but do not auto-select a connection until its first inference succeeds.
        save_chatgpt_profile(profile, active=False)
        info.set_auth_data(profile)
        info.set_api_key(str(profile["access_token"]), "auth.json")
        client = LLMClient(info, info.api_key, info.default_model)
        with console.status("[muted]Loading models for your ChatGPT plan…[/]", spinner="dots"):
            models = await client.fetch_models()
        if not models:
            console.print("  [err]✗[/] This ChatGPT account reports no available models.")
            return None
        info.models = models
        model = models[0].id
        if len(models) > 1:
            picked = await pick(
                "Which ChatGPT model?",
                [(item.id, item.label or item.id) for item in models],
                mark_current=False,
            )
            if picked is None:
                return None
            model = picked
        info.default_model = model
        info.small_model = model
        client.model = model
        with console.status(f"[muted]Checking {model}…[/]", spinner="dots"):
            check = await client.ping(model)
    except (AuthenticationError, httpx.HTTPError) as error:
        console.print(f"  [err]✗[/] {escape(str(error))}")
        return None
    if not check.ok:
        console.print(f"  [err]✗[/] ChatGPT couldn't run {model}: [muted]{escape(check.message)}[/]")
        return None
    save_chatgpt_profile(profile)
    path = save_provider_connection(info)
    account = profile.get("email") or "ChatGPT account"
    console.print(
        f"  [ok]✔[/] Connected {escape(str(account))} to {model} and saved to [path]{pretty_path(path)}[/]."
    )
    return check


def _connection_status(info: Provider) -> str:
    if info.local or info.endpoint_setup:
        return "model endpoint"
    return "✔ connected" if info.has_key else "connect"


async def _connect_openrouter(
    console: Console, info: Provider, providers: dict[str, Provider]
) -> Ping | None:
    mode = await pick(
        "Where is tinyharness running?",
        [
            ("browser", "This computer  · open a browser and return automatically"),
            ("headless", "Remote or headless  · paste a one-time code"),
        ],
        mark_current=False,
    )
    if mode is None:
        return None
    try:
        if mode == "browser":
            console.print("\n  Opening OpenRouter in your browser…")
            with console.status("[muted]Waiting for authorization…[/]", spinner="dots"):
                key = await openrouter_browser_login()
        else:
            authorization = openrouter_authorization(None)
            console.print(f"\n  Open this URL in a browser:\n\n  [path]{authorization.url}[/]\n")
            code = await ask("Paste the one-time code:", hint="expires in 10 minutes")
            if not code:
                return None
            with console.status("[muted]Finishing the connection…[/]", spinner="dots"):
                key = await exchange_openrouter_code(code.strip(), authorization.verifier)
    except AuthenticationError as error:
        console.print(f"  [err]✗[/] {escape(str(error))}")
        return None
    with console.status(f"[muted]Checking the connection with {info.label}…[/]", spinner="dots"):
        check = await LLMClient(info, key, info.default_model).ping()
    if check.bad_key:
        console.print(f"  [err]✗[/] {info.label} rejected the connection: [muted]{escape(check.message)}[/]")
        return None
    path = save_key(info.name, key, providers)
    console.print(f"  [ok]✔[/] Connected and saved to [path]{pretty_path(path)}[/]")
    return check


async def add_key(
    console: Console, info: Provider, providers: dict[str, Provider], key: str = ""
) -> Ping | None:
    """Make sure `info` has a working key: ask for one (unless `key` is given), check it
    with the provider, and save a pasted one to the key file. A process environment or
    development ``.env`` key remains active because it has higher precedence.
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
    console.print(Text.assemble(("  " + LOGO + " ", "brand"), wordmark("Welcome to tinyharness")))
    console.print("\n  Connect one model provider to get started:\n")

    table = Table.grid(padding=(0, 3))
    table.add_column(style="bold")
    table.add_column(style="muted")
    for p in config.providers.values():
        if p.local:
            methods = f"local model server · {p.base_url}"
        elif p.auth_refresh == "chatgpt":
            methods = "Continue with ChatGPT"
        elif p.auth_refresh == "github_copilot":
            methods = "GitHub subscription"
        else:
            methods = "browser sign-in or API key" if len(p.auth_methods) > 1 else p.key_url
        table.add_row(f"    {p.label}", methods)
    console.print(table)

    console.print("\n  Run [bold]tinyharness --setup[/] for guided setup, or provide a key in your shell:\n")
    console.print(f"    export {first.api_key_env}=<your key>\n")
