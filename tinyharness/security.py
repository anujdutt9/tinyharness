"""Security primitives shared across tools, hooks and the permission layer."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

# Paths that look like credentials. Tools refuse to read these even if the
# permission gate would allow it, so a prompt-injected "read .env and summarize"
# can't leak secrets. Set TINYHARNESS_ALLOW_SECRET_READS=1 to opt out.
_SECRET_PATH_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"(^|/)\.env($|\.|/)"), (".env", ".env.*", ".env/**")),
    (
        re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.|$)"),
        tuple(
            value for kind in ("rsa", "ed25519", "ecdsa", "dsa") for value in (f"id_{kind}", f"id_{kind}.*")
        ),
    ),
    (re.compile(r"(^|/)\.ssh/"), (".ssh/**",)),
    (re.compile(r"(^|/)\.aws/"), (".aws/**",)),
    (re.compile(r"(^|/)\.gnupg/"), (".gnupg/**",)),
    (re.compile(r"(^|/)\.netrc$"), (".netrc",)),
    (re.compile(r"(^|/)\.npmrc$"), (".npmrc",)),
    (re.compile(r"(^|/)\.pypirc$"), (".pypirc",)),
    (re.compile(r"(^|/)\.docker/config\.json$"), (".docker/config.json",)),
    (re.compile(r"(^|/)\.kube/config$"), (".kube/config",)),
    (re.compile(r"(^|/)\.config/gh/hosts\.yml$"), (".config/gh/hosts.yml",)),
    (re.compile(r"(^|/)\.config/gcloud/"), (".config/gcloud/**",)),
    (re.compile(r"(^|/)\.azure/"), (".azure/**",)),
    (
        re.compile(r"(^|/)\.terraform\.d/credentials\.tfrc\.json$"),
        (".terraform.d/credentials.tfrc.json",),
    ),
    (re.compile(r"(^|/)auth\.json$"), ("auth.json",)),
)
SECRET_PATH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(rule[0] for rule in _SECRET_PATH_RULES)

# Committed templates of the files above: they hold placeholders, not secrets.
_PUBLIC_PATH_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(r"(^|/)\.env\.(example|sample|template|dist)$"),
        tuple(f".env.{suffix}" for suffix in ("example", "sample", "template", "dist")),
    ),
)
PUBLIC_PATH_PATTERNS: tuple[re.Pattern[str], ...] = tuple(rule[0] for rule in _PUBLIC_PATH_RULES)

# Environment variables that must never reach a subprocess we spawn (hooks).
_SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASSPHRASE")
_SECRET_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    }
)


def _is_secret_env_name(name: str) -> bool:
    return name.endswith(_SECRET_ENV_SUFFIXES) or name in _SECRET_ENV_NAMES


def is_secret_path(path: str | Path) -> bool:
    if os.environ.get("TINYHARNESS_ALLOW_SECRET_READS"):
        return False
    # These checks run in-process, so they must understand paths from every supported
    # platform even when a PurePath or serialized path was produced elsewhere. Windows
    # paths are case-insensitive and use backslashes; the policy patterns use `/`.
    s = str(path).replace("\\", "/").casefold()
    return any(p.search(s) for p in SECRET_PATH_PATTERNS) and not any(
        p.search(s) for p in PUBLIC_PATH_PATTERNS
    )


def secret_path_regexes() -> tuple[str, ...]:
    """SECRET_PATH_PATTERNS for matching absolute paths (the OS sandbox's read denylist)."""
    if os.environ.get("TINYHARNESS_ALLOW_SECRET_READS"):
        return ()
    return _absolute(SECRET_PATH_PATTERNS)


def public_path_regexes() -> tuple[str, ...]:
    """PUBLIC_PATH_PATTERNS for matching absolute paths (exceptions to the denylist)."""
    return _absolute(PUBLIC_PATH_PATTERNS)


def sandbox_path_globs(regexes: Iterable[str], roots: Iterable[Path]) -> tuple[str, ...]:
    """Translate shared credential regexes to Sandbox Runtime path globs.

    Sandbox Runtime accepts filesystem globs rather than regular expressions. Refuse
    unknown expressions instead of silently weakening a custom policy.
    """
    rules = {
        pattern.pattern.replace("(^|/)", "/"): suffixes
        for pattern, suffixes in (*_SECRET_PATH_RULES, *_PUBLIC_PATH_RULES)
    }
    suffixes: list[str] = []
    direct: list[str] = []
    fixture_suffixes = ("/pytest-of-[^/]+/", r"/pytest\-of\-[^/]+/")
    for regex in regexes:
        if regex in rules:
            suffixes.extend(rules[regex])
            continue
        fixture_suffix = next((suffix for suffix in fixture_suffixes if regex.endswith(suffix)), None)
        if regex.startswith("^") and fixture_suffix is not None:
            escaped_root = regex[1 : -len(fixture_suffix)]
            root = re.sub(r"\\(.)", r"\1", escaped_root)
            direct.append(f"{root}/pytest-of-*/**")
            continue
        raise ValueError(f"Sandbox Runtime cannot translate read policy regex: {regex}")
    return tuple(
        dict.fromkeys(
            [
                *direct,
                *(
                    f"{Path(root).expanduser().resolve()}/**/{suffix}"
                    for root in roots
                    for suffix in suffixes
                ),
            ]
        )
    )


def _absolute(patterns: tuple[re.Pattern[str], ...]) -> tuple[str, ...]:
    return tuple(p.pattern.replace("(^|/)", "/") for p in patterns)


def sanitized_env() -> dict[str, str]:
    """The current environment minus anything that looks like a credential."""
    return {k: v for k, v in os.environ.items() if not _is_secret_env_name(k)}


def redact(secret: str) -> str:
    """Show just enough of a key to identify it."""
    if not secret:
        return "(missing)"
    if len(secret) <= 16:
        return "(set)"
    return f"{secret[:8]}...{secret[-4:]}"


# -- keeping keys out of the model's context --------------------------------------
#
# Two layers: the CLI refuses to send a message that looks like it contains a key
# (`find_api_key`), and every tool result has the values of the keys in this
# environment masked before the model sees it (`mask_secrets`).

_MIN_SECRET_LENGTH = 12  # shorter values are too likely to appear by coincidence

# Key formats that are recognizable on sight. Keys without a prefix (e.g. Mistral's)
# are still caught by exact value once they're configured.
API_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])("
    r"sk-ant-[A-Za-z0-9_-]{20,}"  # Anthropic
    r"|sk-or-v1-[A-Za-z0-9]{20,}"  # OpenRouter
    r"|sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}"  # OpenAI, DeepSeek
    r"|AIza[0-9A-Za-z_-]{35}"  # Google
    r"|gsk_[A-Za-z0-9]{20,}"  # Groq
    r"|xai-[A-Za-z0-9]{20,}"  # xAI
    r")"
)


def known_secrets(extra: dict[str, str] | None = None) -> dict[str, str]:
    """{name: value} for credentials in the environment (plus `extra`), longest first."""
    found = {k: v for k, v in os.environ.items() if _is_secret_env_name(k)}
    found.update(extra or {})
    usable = {k: v for k, v in found.items() if len(v) >= _MIN_SECRET_LENGTH}
    return dict(sorted(usable.items(), key=lambda kv: -len(kv[1])))


def mask_secrets(text: str, extra: dict[str, str] | None = None) -> str:
    """Replace every known credential value in `text` with `[REDACTED NAME]`."""
    for name, value in known_secrets(extra).items():
        if value in text:
            text = text.replace(value, f"[REDACTED {name}]")
    return text


def redact_secrets(text: str, extra: dict[str, str] | None = None) -> str:
    """`mask_secrets`, plus anything shaped like an API key even if it isn't configured
    here (a key the user pasted, one inside a URL). For text that leaves the session:
    hook payloads and session logs."""
    return API_KEY_PATTERN.sub("[REDACTED API KEY]", mask_secrets(text, extra))


def find_api_key(text: str, extra: dict[str, str] | None = None) -> str | None:
    """Return a redacted preview of the first API key in ``text``, if present."""
    for value in known_secrets(extra).values():
        if value in text:
            return redact(value)
    match = API_KEY_PATTERN.search(text)
    return redact(match.group(1)) if match else None
