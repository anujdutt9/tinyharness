"""Optional cross-platform backend for Anthropic's Sandbox Runtime CLI.

Sandbox Runtime uses Seatbelt on macOS, Bubblewrap on Linux, and restricted accounts
plus Windows Filtering Platform on Windows. tinyharness only detects an installed
``srt`` executable, so this integration adds no Python or runtime dependency.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tinyharness.paths import home_dir
from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.policy import SandboxPolicy
from tinyharness.security import sandbox_path_globs, sanitized_env

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
        """Return whether SRT is installed and can run a command on this machine."""
        return cls.installed() and cannot_run() is None

    @classmethod
    def installed(cls) -> bool:
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

    def validate_policy(self, policy: SandboxPolicy) -> None:
        """Reject writable roots that would reopen a protected directory."""
        # SRT's denies win over its allows. Enumerating existing siblings leaves any
        # future entries unprotected, so refuse a policy that needs this exception.
        if reopened := policy.reopened():
            raise ValueError(
                f"Sandbox Runtime cannot reopen a protected path for writing: {reopened[0]}. "
                "Use a narrower project or allow_write root outside protected paths."
            )

    def _settings(self, policy: SandboxPolicy) -> dict[str, object]:
        """Build SRT settings without exposing regex-based native policy details."""
        self.validate_policy(policy)
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
                "denyWrite": [str(path) for path in policy.protected],
            },
            "enableWeakerNestedSandbox": False,
            "enableWeakerNetworkIsolation": False,
            "allowAppleEvents": False,
        }


# Ubuntu's AppArmor `bwrap-userns-restrict` profile, in SRT's words.
_NESTED_USERNS = "nested userns is capability-restricted"
SECCOMP_HELPER_FAILED = re.compile(rf"^apply-seccomp: [^\n]*{_NESTED_USERNS}[^\n]*$", re.M)


@functools.cache
def cannot_run() -> str | None:
    """Why SRT can't run even `true` on this machine, or None when it can.

    Being installed isn't enough. Seen in CI: on Ubuntu 26.04, AppArmor denies CAP_SYS_ADMIN
    to bubblewrap's children, so SRT's seccomp helper aborted every command before it
    started, and each Bash call failed with a note blaming the command. Checked once per
    process, like Seatbelt's probe; only called once SRT is installed.
    """
    shell = shutil.which("sh") or shutil.which("bash")
    if shell is None:
        return "no POSIX shell (sh) to run commands with"
    try:
        with tempfile.TemporaryDirectory(prefix="th-srt-") as scratch:
            root = Path(scratch).resolve()
            argv = SandboxRuntime(root / "settings").wrap(
                [shell, "-c", "true"], SandboxPolicy(write_roots=(root,))
            )
            probe = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                stdin=subprocess.DEVNULL,
                cwd=root,
                env=sanitized_env(),
            )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        return f"srt didn't start ({error})"
    if probe.returncode == 0:
        return None
    output = f"{probe.stderr}\n{probe.stdout}"
    if helper := SECCOMP_HELPER_FAILED.search(output):
        return helper[0][:300]
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return (lines[0][:300] if lines else "") or f"srt exited with code {probe.returncode}"


def unusable_hint() -> str | None:
    """What to tell the user when SRT is installed but can't run here, or None."""
    if not SandboxRuntime.installed() or (reason := cannot_run()) is None:
        return None
    hint = f"Sandbox Runtime is installed but can't run commands here: {reason}"
    if _NESTED_USERNS in reason:
        hint += (
            ". Ubuntu's AppArmor profile for bubblewrap blocks its seccomp helper (Ubuntu 26.04 and "
            "some 24.04 hosts), an open Sandbox Runtime issue: "
            "https://github.com/anthropics/sandbox-runtime/issues/429"
        )
    return hint


__all__ = ["SandboxRuntime"]
