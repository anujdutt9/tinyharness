"""Parse Markdown files with optional YAML frontmatter (SKILL.md, agent .md files)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split `---\\n<yaml>\\n---\\n<body>` into (metadata, body).

    Malformed YAML degrades to empty metadata rather than failing, so one bad
    file never prevents the rest from loading.
    """
    if not text.startswith("---"):
        return {}, text.strip()
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text.strip()
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, parts[2].strip()


def read_frontmatter_file(path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        return parse_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return None
