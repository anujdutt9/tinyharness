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


def is_secret_path(path: str | Path) -> bool:
    if os.environ.get("TINYHARNESS_ALLOW_SECRET_READS"):
        return False
    s = str(path)
    return any(p.search(s) for p in SECRET_PATH_PATTERNS)


def sanitized_env() -> dict[str, str]:
    """The current environment minus anything that looks like a credential."""
    return {
        k: v
        for k, v in os.environ.items()
        if not k.endswith(_SECRET_ENV_SUFFIXES) and k not in _SECRET_ENV_NAMES
    }


def redact(secret: str) -> str:
    """Show just enough of a key to identify it."""
    if not secret:
        return "(missing)"
    if len(secret) <= 16:
        return "(set)"
    return f"{secret[:8]}...{secret[-4:]}"
