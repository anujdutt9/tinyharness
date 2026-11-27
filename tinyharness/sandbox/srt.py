"""Optional cross-platform backend for Anthropic's Sandbox Runtime CLI.

Sandbox Runtime uses Seatbelt on macOS, Bubblewrap on Linux, and restricted accounts
plus Windows Filtering Platform on Windows. tinyharness only detects an installed
``srt`` executable, so this integration adds no Python or runtime dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tinyharness.paths import home_dir
from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.policy import SandboxPolicy
from tinyharness.security import sandbox_path_globs

# Variables only the wrapped command gets, never SRT itself (see `environment`).
_INNER_ONLY = frozenset(
    {
        "TMPDIR",
        "TMP",
        "TEMP",
        "HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_CACHE_HOME",
        "PYTHONUSERBASE",
        "PATH",
    }
)


class SandboxRuntime(Sandbox):
    """Translate :class:`SandboxPolicy` into Sandbox Runtime settings."""

    name = "sandbox-runtime"
    manages_network = True

    def __init__(self, settings_dir: Path | None = None) -> None:
        self.settings_dir = settings_dir or home_dir() / "sandbox-settings"

    @classmethod
    def available(cls) -> bool:
        """Return whether SRT and its platform helpers are installed."""
        dependencies = {
            "darwin": ("srt", "rg"),
            "linux": ("srt", "bwrap", "socat", "rg"),
            "win32": ("srt",),
        }.get(sys.platform)
        if dependencies is None or not all(shutil.which(name) for name in dependencies):
            return False
        # tinyharness commands use a POSIX shell. SRT supplies the Windows isolation
        # helper, while Git Bash (or another bash/sh) supplies that shell.
        return sys.platform != "win32" or bool(shutil.which("bash") or shutil.which("sh"))

    def wrap(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        """Wrap ``argv`` with an immutable, private settings file."""
        executable = shutil.which("srt")
        if executable is None:
            raise RuntimeError("Sandbox Runtime (srt) is no longer available")
        encoded = json.dumps(self._settings(policy), sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(encoded).hexdigest()[:20]
        self.settings_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = self.settings_dir / f"{digest}.json"
        if not path.exists():
            fd, temporary = tempfile.mkstemp(prefix=f".{digest}-", dir=self.settings_dir)
            try:
                os.chmod(temporary, 0o600)
                with os.fdopen(fd, "wb") as file:
                    file.write(encoded)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        path.chmod(0o600)
        private_tmp = {name: policy.env[name] for name in sorted(_INNER_ONLY) if name in policy.env}
        if private_tmp:
            env = shutil.which("env")
            if env is None:
                raise RuntimeError("Sandbox Runtime needs the env command to set its private temp directory")
            argv = [env, *(f"{name}={value}" for name, value in private_tmp.items()), *argv]
        return [executable, "--settings", str(path), "--", *argv]

    def environment(self, policy: SandboxPolicy) -> dict[str, str]:
        """Keep SRT control sockets out of a long per-session temporary path, and SRT's own
        home-relative protections on the real home."""
        # Linux Unix-domain socket paths are limited to 108 bytes. SRT creates its
        # proxy bridge before it starts the wrapped command, so only that command
        # receives the private session temp variables added in wrap(). The private $HOME
        # goes the same way: SRT resolves paths like ~/.ssh itself, from its own $HOME.
        return {key: value for key, value in policy.env.items() if key not in _INNER_ONLY}

    def _settings(self, policy: SandboxPolicy) -> dict[str, object]:
        """Build SRT settings without exposing regex-based native policy details."""
        home = Path.home().expanduser().resolve()
        deny_read: list[Path | str] = [
            home / ".ssh",
            home / ".aws",
            home / ".gnupg",
            home / ".netrc",
            home / ".npmrc",
            home / ".pypirc",
            home / ".docker/config.json",
            home / ".kube/config",
            home / ".config/gh/hosts.yml",
            home / ".config/gcloud",
            home / ".azure",
            home_dir() / "auth.json",
        ]
        allow_read: list[str] = []
        for root in policy.project_roots:
            prefix = root.as_posix()
            deny_read.extend(
                [
                    f"{prefix}/**/.env",
                    f"{prefix}/**/.env.*",
                    f"{prefix}/**/.npmrc",
                    f"{prefix}/**/.pypirc",
                    f"{prefix}/**/.netrc",
                    f"{prefix}/**/auth.json",
                    f"{prefix}/**/.docker/config.json",
                    f"{prefix}/**/.kube/config",
                ]
            )
            allow_read.extend(
                f"{prefix}/**/.env.{suffix}" for suffix in ("example", "sample", "template", "dist")
            )
        # SRT accepts globs rather than Seatbelt's regular expressions. Translate the
        # shared policy for the project and its sibling repositories in addition to the
        # longstanding home-directory protections above. Unknown custom regexes fail
        # closed in the translator instead of being silently omitted.
        read_roots = tuple(
            dict.fromkeys(
                [
                    *(root.resolve().parent for root in policy.project_roots),
                    *(root.resolve() for root in policy.project_roots),
                ]
            )
        )
        deny_read.extend(sandbox_path_globs(policy.deny_read, read_roots))
        allow_read.extend(sandbox_path_globs(policy.allow_read, read_roots))
        return {
            "network": {
                "allowedDomains": list(policy.allowed_hosts),
                "deniedDomains": [],
                "allowUnixSockets": [],
                "allowLocalBinding": policy.allow_localhost,
            },
            "filesystem": {
                "denyRead": [str(path) for path in deny_read],
                "allowRead": allow_read,
                "allowWrite": [str(path) for path in policy.write_roots],
                "denyWrite": [str(path) for path in _deny_write(policy)],
            },
            "enableWeakerNestedSandbox": False,
            "enableWeakerNetworkIsolation": False,
            "allowAppleEvents": False,
        }


def _deny_write(policy: SandboxPolicy) -> list[Path]:
    """`policy.protected` as SRT denies, with write roots inside them kept open.

    SRT's denyWrite beats its allowWrite, unlike Seatbelt's later rule (`reopened`). So a
    protected folder holding a write root (~/.tinyharness, with the session's scratch, when
    the project is your home) is denied entry by entry around that root instead. Entries
    created there later aren't covered on this backend.
    """
    denied: list[Path] = []

    def around(path: Path, inside: list[Path]) -> None:
        if not inside:
            denied.append(path)
            return
        if path in inside:
            return
        try:
            children = list(path.iterdir())
        except OSError:
            return
        for child in children:
            around(child, [root for root in inside if root.is_relative_to(child)])

    for path in policy.protected:
        around(path, [root for root in policy.write_roots if root.is_relative_to(path)])
    return denied


__all__ = ["SandboxRuntime"]
