from __future__ import annotations

from enum import StrEnum


class PermissionMode(StrEnum):
    """How much the agent may do without asking.

    PLAN            read-only tools only; mutating tools are hidden and blocked
    DEFAULT         reads run freely; edits and outside paths ask
    ACCEPT_EDITS    auto-allow file tools inside the working directory
    AUTO            routine work runs; risky actions ask; unclear ones go to an LLM classifier
    BYPASS          never prompt (secret-path denylist still applies)
    """

    PLAN = "plan"
    DEFAULT = "default"
    ACCEPT_EDITS = "acceptEdits"
    AUTO = "auto"
    BYPASS = "bypassPermissions"

    def next(self) -> PermissionMode:
        """The mode Shift+Tab switches to: default → acceptEdits → auto → plan → default."""
        cycle = [
            PermissionMode.DEFAULT,
            PermissionMode.ACCEPT_EDITS,
            PermissionMode.AUTO,
            PermissionMode.PLAN,
        ]
        return cycle[(cycle.index(self) + 1) % len(cycle)] if self in cycle else PermissionMode.DEFAULT

    @classmethod
    def parse(cls, value: str | PermissionMode) -> PermissionMode:
        try:
            return cls(value)
        except ValueError:
            choices = ", ".join(m.value for m in cls)
            raise ValueError(f"unknown permission mode {value!r}; choose one of: {choices}") from None
