"""Security primitives shared across tools, hooks and the permission layer."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Paths that look like credentials. Tools refuse to read these even if the
# permission gate would allow it, so a prompt-injected "read .env and summarize"
# can't leak secrets. Set TINYHARNESS_ALLOW_SECRET_READS=1 to opt out.
SECRET_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.env($|\.|/)"),
    re.compile(r"(^|/)id_(rsa|ed25519|ecdsa|dsa)(\.|$)"),
    re.compile(r"(^|/)\.ssh/"),
    re.compile(r"(^|/)\.aws/credentials"),
    re.compile(r"(^|/)\.gnupg/"),
    re.compile(r"(^|/)\.netrc$"),
    re.compile(r"(^|/)auth\.json$"),  # tinyharness's key file, and other tools' by that name
)

# Committed templates of the files above: they hold placeholders, not secrets.
PUBLIC_PATH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)\.env\.(example|sample|template|dist)$"),
)

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
    s = str(path)
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
