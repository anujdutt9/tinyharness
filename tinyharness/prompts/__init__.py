"""Prompt templates live next to this file as Markdown, so they can be read and
diffed on their own.

    load_prompt("system", working_dir="/repo", scratch_dir="~/.tinyharness/projects/-repo/tmp/x")

Placeholders are `{name}` (lowercase identifiers only), so literal braces such as
JSON examples need no escaping. A missing variable is an error, which keeps an
unfilled `{placeholder}` from silently reaching the model.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
AGENTS_DIR = PROMPTS_DIR / "agents"

_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


@cache
def _read(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8").strip()


def placeholders(name: str) -> set[str]:
    return set(_PLACEHOLDER.findall(_read(name)))


def load_prompt(name: str, **variables: object) -> str:
    """Read `prompts/<name>.md` and fill in its `{placeholders}`."""
    template = _read(name)
    missing = placeholders(name) - variables.keys()
    if missing:
        raise KeyError(f"prompt {name!r} needs values for: {', '.join(sorted(missing))}")
    return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), template)
