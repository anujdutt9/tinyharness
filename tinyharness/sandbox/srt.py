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
        return [executable, "--settings", str(path), "--", *argv]

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
            prefix = str(root)
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
                "denyWrite": [str(path) for path in policy.protected],
            },
            "enableWeakerNestedSandbox": False,
            "enableWeakerNetworkIsolation": False,
            "allowAppleEvents": False,
        }


__all__ = ["SandboxRuntime"]
