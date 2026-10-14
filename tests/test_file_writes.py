"""Read, Pdf, Write and Edit act on exactly the path that was approved; writes are all or nothing."""

import errno
import os
import stat

import pytest

from conftest import call, make_ctx
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.tools.base import ToolError
from tinyharness.tools.edit import EditTool
from tinyharness.tools.pdf import PdfTool
from tinyharness.tools.read import ReadTool
from tinyharness.tools.write import WriteTool
from tinyharness.workspace import Workspace, files


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "f.txt").write_text("original\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "f.txt").write_text("not yours\n")
    return root.resolve(), outside.resolve()


async def approve(root, tool, raw):
    """What the loop does: ask the gate, then pin the approved path onto the arguments."""
    gate = PermissionGate(workspace=Workspace(root), mode=PermissionMode.DEFAULT)
    args = tool.parse(raw)
    decision = await gate.check(tool, raw, tool.target_path(args))
    assert decision.allow and decision.path == root / raw["path"]
    return tool.pin_path(args, decision.path)


def swap_for_link(path, target):
    """What a background job could do between the permission check and the write."""
    os.rename(path, path.with_name(path.name + ".moved"))
    path.symlink_to(target)


# ---- the approved path, not a fresh resolution of it ----


@pytest.mark.parametrize("tool", [WriteTool(), EditTool()])
async def test_a_directory_swapped_for_a_symlink_after_approval_is_refused(project, tool):
    root, outside = project
    raw = {"path": "sub/f.txt", "content": "pwned\n", "old_string": "original", "new_string": "pwned"}
    ctx = make_ctx(root)
    ctx.read_paths.add(str(root / "sub" / "f.txt"))
    args = await approve(root, tool, raw)
    swap_for_link(root / "sub", outside)
    with pytest.raises(ToolError, match=r"sub in .* is now a symlink or not a directory"):
        await tool.run(args, ctx)
    assert (outside / "f.txt").read_text() == "not yours\n"


@pytest.mark.parametrize("tool", [WriteTool(), EditTool()])
async def test_windows_fallback_refuses_a_directory_swapped_after_approval(project, tool, monkeypatch):
    """Seen live: the no-dir-fd fallback followed a swapped parent outside the project."""
    root, outside = project
    raw = {"path": "sub/f.txt", "content": "pwned\n", "old_string": "original", "new_string": "pwned"}
    ctx = make_ctx(root)
    ctx.read_paths.add(str(root / "sub" / "f.txt"))
    args = await approve(root, tool, raw)
    swap_for_link(root / "sub", outside)
    monkeypatch.setattr(files, "_SUPPORTED", False)
    with pytest.raises(ToolError, match=r"sub in .* is now a symlink or not a directory"):
        await tool.run(args, ctx)
    assert (outside / "f.txt").read_text() == "not yours\n"


async def test_a_file_swapped_for_a_symlink_is_replaced_not_followed(project):
    root, outside = project
    ctx = make_ctx(root)
    ctx.read_paths.add(str(root / "sub" / "f.txt"))
    tool = WriteTool()
    args = await approve(root, tool, {"path": "sub/f.txt", "content": "new\n"})
    swap_for_link(root / "sub" / "f.txt", outside / "f.txt")
    await tool.run(args, ctx)
    assert (outside / "f.txt").read_text() == "not yours\n"
    assert not (root / "sub" / "f.txt").is_symlink() and (root / "sub" / "f.txt").read_text() == "new\n"


async def test_without_a_gate_paths_are_resolved_as_before(project, tmp_path):
    """Called directly (SDK), a path through a symlink still works: /tmp is one on macOS."""
    root, _ = project
    link = tmp_path / "link"
    link.symlink_to(root / "sub")
    out = await call(WriteTool(), {"path": str(link / "new.txt"), "content": "hi"}, make_ctx(root))
    assert "wrote" in out and (root / "sub" / "new.txt").read_text() == "hi"


@pytest.mark.parametrize("swap", ["directory", "file"])
@pytest.mark.parametrize(("tool", "name"), [(ReadTool(), "f.txt"), (PdfTool(), "doc.pdf")])
async def test_reads_refuse_a_path_that_became_a_symlink(project, tool, name, swap):
    """A read approved inside the project can't be pointed at a file outside it."""
    root, outside = project
    (root / "sub" / name).write_text("project content\n")
    (outside / name).write_text("outside content\n")
    args = await approve(root, tool, {"path": f"sub/{name}"})
    if swap == "directory":
        swap_for_link(root / "sub", outside)
    else:
        swap_for_link(root / "sub" / name, outside / name)
    with pytest.raises(ToolError, match=r"is now a (?:symlink|reparse point)"):
        await tool.run(args, make_ctx(root))


async def test_windows_fallback_read_refuses_a_swapped_parent(project, monkeypatch):
    """The Windows path fallback must not read through a planted directory reparse point."""
    root, outside = project
    args = await approve(root, ReadTool(), {"path": "sub/f.txt"})
    swap_for_link(root / "sub", outside)
    monkeypatch.setattr(files, "_SUPPORTED", False)
    with pytest.raises(ToolError, match=r"is now a (?:symlink|reparse point)"):
        await ReadTool().run(args, make_ctx(root))


async def test_windows_fallback_read_refuses_a_swapped_file(project, monkeypatch):
    root, outside = project
    args = await approve(root, ReadTool(), {"path": "sub/f.txt"})
    swap_for_link(root / "sub" / "f.txt", outside / "f.txt")
    monkeypatch.setattr(files, "_SUPPORTED", False)
    with pytest.raises(ToolError, match=r"is now a (?:symlink|reparse point)"):
        await ReadTool().run(args, make_ctx(root))


async def test_read_still_reads_an_approved_file(project):
    root, _ = project
    args = await approve(root, ReadTool(), {"path": "sub/f.txt"})
    assert await ReadTool().run(args, make_ctx(root)) == "     1\toriginal"


# ---- all or nothing ----


@pytest.mark.parametrize("tool", [WriteTool(), EditTool()])
async def test_a_failed_write_leaves_the_file_as_it_was(project, tool, monkeypatch):
    root, _ = project
    ctx = make_ctx(root)
    ctx.read_paths.add(str(root / "sub" / "f.txt"))

    def disk_full(fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(files.os, "fsync", disk_full)
    raw = {"path": "sub/f.txt", "content": "new\n", "old_string": "original", "new_string": "new"}
    with pytest.raises(ToolError, match="write failed: No space left on device"):
        await call(tool, raw, ctx)
    assert (root / "sub" / "f.txt").read_text() == "original\n"
    assert os.listdir(root / "sub") == ["f.txt"]  # no temporary file left behind


async def test_directories_made_for_a_failed_write_are_removed(project, monkeypatch):
    root, _ = project
    monkeypatch.setattr(files.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError(errno.EIO, "I/O error")))
    with pytest.raises(ToolError):
        await call(WriteTool(), {"path": "a/b/c/new.txt", "content": "x"}, make_ctx(root))
    assert not (root / "a").exists()


async def test_edit_keeps_the_files_permissions(project):
    root, _ = project
    script = root / "run.sh"
    script.write_text("echo old\n")
    script.chmod(0o750)
    ctx = make_ctx(root)
    ctx.read_paths.add(str(script))
    await call(EditTool(), {"path": "run.sh", "old_string": "old", "new_string": "new"}, ctx)
    assert script.read_text() == "echo new\n"
    if os.name != "nt":
        assert stat.S_IMODE(script.stat().st_mode) == 0o750
