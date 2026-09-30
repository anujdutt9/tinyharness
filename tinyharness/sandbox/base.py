from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from tinyharness.sandbox.policy import SandboxPolicy


class Sandbox(ABC):
    """An OS mechanism that confines a command to a SandboxPolicy."""

    name: ClassVar[str]

    @classmethod
    @abstractmethod
    def available(cls) -> bool:
        """Whether this backend works on this machine, right now."""

    @abstractmethod
    def wrap(self, argv: list[str], policy: SandboxPolicy) -> list[str]:
        """The command line that runs `argv` under `policy`."""
