"""Importing skills from another tool's directory, and replacing them with --force."""

import errno
import os
import shutil

import pytest

from tinyharness.skills import import_skills, load_skills


@pytest.fixture
def source(tmp_path, isolated_home):
    src = tmp_path / "claude-skills"
    write_skill(src, "fmt", "new")
    return src


def write_skill(root, name, body):
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\n{body}")


def installed(isolated_home, name="fmt"):
    return (isolated_home / "skills" / name / "SKILL.md").read_text()


def test_force_replaces_an_existing_skill(source, isolated_home, tmp_path):
    write_skill(isolated_home / "skills", "fmt", "old")
    [result] = import_skills(source, force=True)
    assert result.action == "copied"
    assert installed(isolated_home).endswith("new")
    assert [s.name for s in load_skills(tmp_path)] == ["fmt"]  # no staging leftovers loaded as skills


def test_a_failed_copy_keeps_the_old_skill(source, isolated_home, monkeypatch):
    write_skill(isolated_home / "skills", "fmt", "old")

    def disk_full(src, dst, **kw):
        os.makedirs(dst)
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(shutil, "copytree", disk_full)
    [result] = import_skills(source, force=True)
    assert result.action == "failed"
    assert result.detail == (
        f"could not copy {source / 'fmt'}: No space left on device; the existing skill was left in place"
    )
    assert installed(isolated_home).endswith("old")
    assert not (isolated_home / ".skill-import" / "fmt").exists()


def test_a_failed_swap_puts_the_old_skill_back(source, isolated_home, monkeypatch):
    write_skill(isolated_home / "skills", "fmt", "old")
    real_rename = os.rename

    def rename(src, dst):
        if str(src).endswith(os.path.join(".skill-import", "fmt")):  # moving the new one in
            raise OSError(errno.EACCES, "Permission denied")
        real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename)
    [result] = import_skills(source, force=True)
    assert result.action == "failed" and "left in place" in result.detail
    assert installed(isolated_home).endswith("old")


def test_an_interrupted_import_is_cleaned_up_next_time(source, isolated_home):
    write_skill(isolated_home / ".skill-import", "fmt", "half-copied")
    [result] = import_skills(source)
    assert result.action == "copied" and installed(isolated_home).endswith("new")


def test_symlink_import_replaces_too(source, isolated_home):
    write_skill(isolated_home / "skills", "fmt", "old")
    [result] = import_skills(source, symlink=True, force=True)
    assert result.action == "linked"
    assert (isolated_home / "skills" / "fmt").is_symlink()
    assert installed(isolated_home).endswith("new")


def test_existing_skills_are_skipped_without_force(source, isolated_home):
    write_skill(isolated_home / "skills", "fmt", "old")
    [result] = import_skills(source)
    assert result.action == "skipped" and installed(isolated_home).endswith("old")
