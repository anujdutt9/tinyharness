"""Allow/deny rules: `Tool` or `Tool(glob)`.

    Edit                  any Edit call
    Edit(src/*)           Edit on any path under <working-dir>/src (`*` also matches `/`)
    Read(~/notes/*)       Read under ~/notes
    *(/etc/*)             any tool touching /etc
    WebFetch(domain:docs.python.org)    WebFetch on one site
    WebFetch(domain:*.github.com)       ...or on its subdomains

Rules come from the config file (`[permissions] allow/deny`) and from
"don't ask again" answers during a session. Deny always beats allow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal

DOMAIN = "domain:"  # prefix of a site pattern, as opposed to a path

_RULE = re.compile(r"^\s*([\w*-]+)\s*(?:\((.*)\))?\s*$")


@dataclass(frozen=True)
class Rule:
    tool: str  # tool name, or "*" for any tool
    pattern: str | None = None  # absolute glob; None matches any target

    @classmethod
    def parse(cls, text: str, root: Path) -> Rule:
        match = _RULE.match(text)
        if not match:
            raise ValueError(f"invalid permission rule {text!r}; expected Tool or Tool(pattern)")
        tool, pattern = match.groups()
        if not pattern:
            return cls(tool)
        if pattern.startswith(DOMAIN):
            return cls(tool, DOMAIN + pattern.removeprefix(DOMAIN).strip().lower())
        path = Path(pattern).expanduser()
        return cls(tool, str(path if path.is_absolute() else root / path))

    def matches(self, tool: str, path: str | None) -> bool:
        if self.tool not in ("*", tool):
            return False
        if self.pattern is None:
            return True
        return path is not None and fnmatchcase(path, self.pattern)

    def __str__(self) -> str:
        return f"{self.tool}({self.pattern})" if self.pattern else self.tool


@dataclass
class PermissionRules:
    allow: list[Rule] = field(default_factory=list)
    deny: list[Rule] = field(default_factory=list)

    @classmethod
    def parse(cls, allow: list[str], deny: list[str], root: Path) -> PermissionRules:
        return cls([Rule.parse(r, root) for r in allow], [Rule.parse(r, root) for r in deny])

    def check(self, tool: str, path: str | None) -> Literal["allow", "deny"] | None:
        if any(rule.matches(tool, path) for rule in self.deny):
            return "deny"
        if any(rule.matches(tool, path) for rule in self.allow):
            return "allow"
        return None

    def add_allow(self, rule: Rule) -> None:
        if rule not in self.allow:
            self.allow.append(rule)

    def remove_allow(self, rule: Rule) -> None:
        if rule in self.allow:
            self.allow.remove(rule)
