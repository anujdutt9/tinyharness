"""Workspace only resolves paths and says whether they're inside the working
directory. Policy (allow/deny) is the permission gate's job."""

import os

from tinyharness.workspace import Workspace


def test_relative_paths_resolve_under_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.resolve("a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
    assert ws.contains("a/b.txt")


def test_outside_paths_resolve_but_are_not_contained(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.resolve("/etc/passwd") is not None
    assert not ws.contains("/etc/passwd")
    assert not ws.contains("../../../etc/passwd")


def test_symlinks_are_followed_before_judging(tmp_path):
    ws = Workspace(tmp_path)
    os.symlink("/etc/passwd", tmp_path / "leak")
    resolved = ws.resolve("leak")
    assert resolved is not None and "passwd" in str(resolved)
    assert not ws.contains("leak")


def test_read_roots_are_trusted(tmp_path_factory):
    root, other = tmp_path_factory.mktemp("root"), tmp_path_factory.mktemp("skill")
    ws = Workspace(root, read_roots=[other])
    assert ws.is_trusted(other / "x.md")
    assert ws.in_read_root(other / "x.md")
    assert ws.is_trusted(root / "x.md")
    assert not ws.is_trusted("/etc/passwd")


def test_config_location_follows_xdg_config_home(tmp_path, monkeypatch):
    from tinyharness.paths import default_config_path

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert default_config_path() == tmp_path / "xdg" / "tinyharness" / "config.toml"
    monkeypatch.delenv("XDG_CONFIG_HOME")
    assert default_config_path().parts[-3:] == (".config", "tinyharness", "config.toml")
