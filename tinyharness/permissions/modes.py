from __future__ import annotations

from enum import StrEnum


class PermissionMode(StrEnum):
    """How much the agent may do without asking.

    PLAN       read-only tools only; mutating tools are hidden and blocked
    ASK        reads run freely; every edit and command asks
    DEFAULT    edits inside the project and sandboxed commands run; anything that leaves
               the boundary (other folders, sandbox=false, network tools) asks
    AUTO       as DEFAULT, and an LLM classifier decides the unclear cases
    BYPASS     never prompt (secret-path denylist still applies)
    """

    PLAN = "plan"
    ASK = "ask"
    DEFAULT = "default"
    AUTO = "auto"
    BYPASS = "bypassPermissions"

    def next(self) -> PermissionMode:
        """The mode Shift+Tab switches to: default → auto → plan → ask → default."""
        cycle = [PermissionMode.DEFAULT, PermissionMode.AUTO, PermissionMode.PLAN, PermissionMode.ASK]
        return cycle[(cycle.index(self) + 1) % len(cycle)] if self in cycle else PermissionMode.DEFAULT

    @classmethod
    def parse(cls, value: str | PermissionMode) -> PermissionMode:
        if value == "acceptEdits":  # the earlier name for what default does now
            return cls.DEFAULT
        try:
            return cls(value)
        except ValueError:
            choices = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown permission mode {value!r}; choose one of: {choices}") from None
