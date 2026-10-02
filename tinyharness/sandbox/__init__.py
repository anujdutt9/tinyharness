"""OS-level sandboxing for the commands the agent runs.

One policy (policy.py), several backends. `detect_sandbox()` picks the strongest one
this machine supports; None means commands run unsandboxed, and the permission
gate then asks before every one of them.
"""

from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.policy import SandboxPolicy
from tinyharness.sandbox.seatbelt import Seatbelt

BACKENDS: tuple[type[Sandbox], ...] = (Seatbelt,)


def detect_sandbox() -> Sandbox | None:
    for backend in BACKENDS:
        if backend.available():
            return backend()
    return None


__all__ = ["BACKENDS", "Sandbox", "SandboxPolicy", "Seatbelt", "detect_sandbox"]
