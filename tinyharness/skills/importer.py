"""Import skills from another tool's directory (default ~/.claude/skills/) into
tinyharness's home (~/.tinyharness/skills/).

Default mode: copy. --symlink links instead of copying (live-tracked but breaks
if the source is moved). Existing destination dirs are skipped unless --force is
set. A forced import replaces the old skill only once the new one is fully in place,
so a failed copy leaves the old one as it was.
"""

from __future__ import annotations

import contextlib
import os
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
    # Built beside skills/, not in it: anything in skills/ would load as a skill.
    staging = home_dir() / ".skill-import"

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

        replacing = dst.exists() or dst.is_symlink()
        if replacing and not force:
            results.append(
                ImportResult(name, entry, dst, "skipped", "destination exists; use --force to overwrite")
            )
            continue

        staged = staging / name
        try:
            staging.mkdir(exist_ok=True)
            _remove(staged)  # left over from an import that was interrupted
            if symlink:
                staged.symlink_to(entry.resolve())
            else:
                shutil.copytree(entry, staged)
            _swap_in(staged, dst, staging / f"{name}.old")
        except OSError as e:
            with contextlib.suppress(OSError):  # tidying up mustn't hide the real error
                _remove(staged)
            detail = f"could not {'link' if symlink else 'copy'} {entry}: {e.strerror or e}"
            if replacing:
                detail += "; the existing skill was left in place"
            results.append(ImportResult(name, entry, dst, "failed", detail))
            continue

        results.append(ImportResult(name, entry, dst, "linked" if symlink else "copied"))

    return results


def _swap_in(staged: Path, dst: Path, backup: Path) -> None:
    """Move `staged` to `dst`. What was at `dst` is moved aside first and deleted only
    once the new one is in place; if that move fails, it goes back."""
    if not (dst.exists() or dst.is_symlink()):
        os.rename(staged, dst)
        return
    _remove(backup)
    os.rename(dst, backup)
    try:
        os.rename(staged, dst)
    except OSError:
        os.rename(backup, dst)
        raise
    _remove(backup)


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)
