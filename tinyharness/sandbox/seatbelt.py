"""macOS backend: Seatbelt, via /usr/bin/sandbox-exec.

Apple marks sandbox-exec deprecated, but it ships on every Mac and is what Codex,
Claude Code and Cursor use. The profile starts from `(allow default)` so ordinary
tools work, then takes away writes, secret reads, the network, and the ways a
process can get something *else* to act for it outside the sandbox:

    open / LaunchServices   `open evil.command` would run it in Terminal, unsandboxed
    Apple Events            `osascript` telling another app to run a command

In Seatbelt profiles the last matching rule wins, so order matters below.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.policy import SandboxPolicy

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Device files ordinary programs write to (terminals, /dev/null, dtrace probes).
_DEVICES = r'(regex #"^/dev/(null|zero|tty.*|ptmx|fd/.*|stdout|stderr|dtracehelper)$")'

_NO_ESCAPE = """\
; nothing may be handed to a process outside the sandbox
(deny appleevent-send)
(deny mach-lookup
  (global-name "com.apple.coreservices.launchservicesd")
  (global-name "com.apple.lsd.mapdb")
  (global-name "com.apple.lsd.modifydb")
  (global-name "com.apple.lsd.open"))"""


class Seatbelt(Sandbox):
    name = "seatbelt"

    @classmethod
    def available(cls) -> bool:
        return sys.platform == "darwin" and _works()

    def wrap(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        return [SANDBOX_EXEC, "-p", profile(policy), *argv]


def profile(policy: SandboxPolicy) -> str:
    lines = ["(version 1)", "(allow default)", "", "; network: none, or loopback only", "(deny network*)"]
    if policy.proxy_port and not policy.allow_localhost:
        lines.append(f'(allow network-outbound (remote ip "localhost:{policy.proxy_port}"))')
    if policy.allow_localhost:
        lines += [
            '(allow network-bind network-inbound (local ip "localhost:*"))',
            '(allow network-outbound (remote ip "localhost:*"))',
        ]
    lines += ["", "; writes: only inside the write roots", "(deny file-write*)"]
    lines.append(_rule("allow file-write*", [*map(_subpath, policy.write_roots), _DEVICES]))
    if policy.protected:
        lines.append(_rule("deny file-write*", [_subpath(p) for p in policy.protected]))
    if reopened := policy.reopened():
        lines.append(_rule("allow file-write*", [_subpath(p) for p in reopened]))
    if policy.deny_read:
        # Contents only: tools like pytest stat every file, and existence tells nothing.
        lines += ["", "; reads: anything but the contents of credential-shaped files"]
        lines.append(_rule("deny file-read-data", [_regex(r) for r in policy.deny_read]))
        if policy.allow_read:
            lines.append(_rule("allow file-read-data", [_regex(r) for r in policy.allow_read]))
    lines += ["", _NO_ESCAPE]
    return "\n".join(lines) + "\n"


def _rule(action: str, filters: list[str]) -> str:
    return f"({action}\n  " + "\n  ".join(filters) + ")"


def _subpath(path: Path) -> str:
    return f'(subpath "{_quote(str(path))}")'


def _regex(pattern: str) -> str:
    """`#"..."` is a raw string in Seatbelt profiles: the regex goes in as is."""
    if '"' in pattern:
        raise ValueError(f"sandbox regex can't contain a double quote: {pattern}")
    return f'(regex #"{pattern}")'


def _quote(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


@functools.cache
def _works() -> bool:
    """sandbox-exec exists and may be used here. It can't be nested, so this is False
    when tinyharness itself runs inside another sandbox."""
    if not Path(SANDBOX_EXEC).exists():
        return False
    try:
        probe = subprocess.run(
            [SANDBOX_EXEC, "-p", "(version 1)(allow default)", "/usr/bin/true"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0
