"""OS-level sandboxing for the commands the agent runs.

One policy (policy.py), several backends. `detect_sandbox()` picks the strongest one
this machine supports; None means commands run unsandboxed, and the permission
gate then asks before every one of them.
"""

import shutil
import sys

from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.policy import SandboxPolicy
from tinyharness.sandbox.seatbelt import Seatbelt
from tinyharness.sandbox.srt import SandboxRuntime

# Prefer the small native macOS profile we control. Sandbox Runtime supplies Linux and
# Windows support and is a fallback when the native backend cannot run.
BACKENDS: tuple[type[Sandbox], ...] = (Seatbelt, SandboxRuntime)


def detect_sandbox() -> Sandbox | None:
    for backend in BACKENDS:
        if backend.available():
            return backend()
    return None


_LINUX_INSTALL = {
    "apt-get": "sudo apt-get install bubblewrap socat ripgrep",
    "dnf": "sudo dnf install bubblewrap socat ripgrep",
    "pacman": "sudo pacman -S bubblewrap socat ripgrep",
    "zypper": "sudo zypper install bubblewrap socat ripgrep",
}


def setup_hint() -> str:
    """Return platform-specific setup help when no backend is usable."""
    if sys.platform == "linux":
        # The exact command for the package manager that's here, so it can be pasted as is.
        manager = next((name for name in _LINUX_INSTALL if shutil.which(name)), None)
        install = _LINUX_INSTALL[manager] if manager else "install bubblewrap, socat and ripgrep"
        return f"{install}, then npm install -g @anthropic-ai/sandbox-runtime"
    if sys.platform == "win32":
        return (
            "install Git Bash and @anthropic-ai/sandbox-runtime, then run its "
            "windows-install command from an elevated prompt"
        )
    if sys.platform == "darwin":
        return (
            "the native sandbox may be unavailable inside another sandbox; as a fallback, "
            "install ripgrep and @anthropic-ai/sandbox-runtime"
        )
    return "this operating system has no supported sandbox backend"


__all__ = [
    "BACKENDS",
    "Sandbox",
    "SandboxPolicy",
    "SandboxRuntime",
    "Seatbelt",
    "detect_sandbox",
    "setup_hint",
]
