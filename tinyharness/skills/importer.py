"""Import skills from another tool's directory (default ~/.claude/skills/) into
tinyharness's home (~/.tinyharness/skills/).

Default mode: copy. --symlink links instead of copying (live-tracked but breaks
if the source is moved). Existing destination dirs are skipped unless --force is
set.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from tinyharness.paths import home_dir
from tinyharness.skills.skill import parse_skill_md


@dataclass
class ImportResult:
    name: str
    src: Path
    dst: Path
    action: Literal["copied", "linked", "skipped", "failed"]
    detail: str = ""


def import_skills(
    source: Path | None = None,
    *,
    symlink: bool = False,
    force: bool = False,
) -> list[ImportResult]:
    src_root = (source or Path.home() / ".claude" / "skills").expanduser()
    dst_root = home_dir() / "skills"
    dst_root.mkdir(parents=True, exist_ok=True)

    results: list[ImportResult] = []
    if not src_root.exists() or not src_root.is_dir():
        return results

    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / "SKILL.md"
        if not skill_md.exists():
            continue
        name = entry.name
        dst = dst_root / name

        if parse_skill_md(skill_md) is None:
            results.append(ImportResult(name, entry, dst, "failed", "could not parse SKILL.md"))
            continue

        if dst.exists():
            if not force:
                results.append(
                    ImportResult(name, entry, dst, "skipped", "destination exists; use --force to overwrite")
                )
                continue
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

        try:
            if symlink:
                dst.symlink_to(entry.resolve())
            else:
                shutil.copytree(entry, dst)
        except OSError as e:
            results.append(ImportResult(name, entry, dst, "failed", str(e)))
            continue

        results.append(ImportResult(name, entry, dst, "linked" if symlink else "copied"))

    return results
