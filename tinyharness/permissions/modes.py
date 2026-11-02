from __future__ import annotations

from enum import StrEnum


class PermissionMode(StrEnum):
    """How much the agent may do without asking.

    PLAN       read-only tools only; mutating tools are hidden and blocked
    ASK        reads run freely; every edit and command asks
    DEFAULT    edits inside the project and sandboxed commands run; other folders and
               network tools ask
    AUTO       as DEFAULT; an LLM classifier decides other unclear cases
    BYPASS     skip routine prompts inside containment; hard boundaries still ask
    """

    PLAN = "plan"
    ASK = "ask"
    DEFAULT = "default"
    AUTO = "auto"
    BYPASS = "bypassPermissions"

    def next(self) -> PermissionMode:
        """The mode Shift+Tab switches to: default → auto → plan → default.

        Ask isn't in the cycle (`/mode ask`): people approve prompts out of habit, so a
        prompt for every edit protects less than it seems, and a fourth stop slows the rest.
        """
        cycle = [PermissionMode.DEFAULT, PermissionMode.AUTO, PermissionMode.PLAN]
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
