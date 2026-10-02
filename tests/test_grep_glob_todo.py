import pydantic
import pytest

from conftest import call
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.tools.base import ToolError
from tinyharness.tools.glob import GlobTool
from tinyharness.tools.grep import GrepTool
from tinyharness.tools.registry import default_registry
from tinyharness.tools.todo import TodoWriteTool, format_todos
from tinyharness.workspace import Workspace

# ---- Glob ----


async def test_glob_finds_files(ctx, tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("a")
    (tmp_path / "src" / "b.py").write_text("b")
    (tmp_path / "README.md").write_text("hi")
    out = await call(GlobTool(), {"pattern": "**/*.py"}, ctx)  # defaults to the working dir
    assert "a.py" in out and "b.py" in out
    assert "README.md" not in out


async def test_glob_skips_hidden_and_secrets(ctx, tmp_path):
    (tmp_path / ".env").write_text("X=1")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: x")
    (tmp_path / "code.py").write_text("y")
    out = await call(GlobTool(), {"pattern": "**/*", "path": str(tmp_path)}, ctx)
    assert "code.py" in out
    assert ".env" not in out and ".git" not in out


async def test_glob_outside_path(ctx, tmp_path_factory):
    """Tools accept outside paths; the permission gate decides whether that's allowed."""
    other = tmp_path_factory.mktemp("other")
    (other / "a.txt").write_text("hi")
    assert "a.txt" in await call(GlobTool(), {"pattern": "**/*", "path": str(other)}, ctx)


# ---- Grep ----


async def test_grep_files_with_matches(ctx, tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    pass\n")
    (tmp_path / "b.py").write_text("def bar():\n    pass\n")
    out = await call(GrepTool(), {"pattern": "foo", "include": "*.py"}, ctx)
    assert "a.py" in out and "b.py" not in out


async def test_grep_content_mode(ctx, tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbeta foo bar\ngamma")
    out = await call(GrepTool(), {"pattern": "foo", "output_mode": "content"}, ctx)
    assert "a.py:2:beta foo bar" in out


async def test_grep_count_mode(ctx, tmp_path):
    (tmp_path / "a.py").write_text("foo\nfoo bar\nfoo baz")
    out = await call(GrepTool(), {"pattern": "foo", "output_mode": "count"}, ctx)
    assert "3 total match" in out


async def test_grep_skips_binary(ctx, tmp_path):
    (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02foo\x00")
    (tmp_path / "src.py").write_text("foo")
    out = await call(GrepTool(), {"pattern": "foo"}, ctx)
    assert "src.py" in out and "img.bin" not in out


async def test_grep_invalid_regex(ctx):
    with pytest.raises(ToolError, match="invalid regex"):
        await call(GrepTool(), {"pattern": "[unclosed"}, ctx)


async def test_grep_skips_dependency_and_hidden_dirs(ctx, tmp_path):
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.js").write_text("foo")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "x.txt").write_text("foo")
    (tmp_path / "src.py").write_text("foo")
    out = await call(GrepTool(), {"pattern": "foo"}, ctx)
    assert "src.py" in out
    assert "x.js" not in out and "x.txt" not in out


async def test_grep_on_a_named_file_still_refuses_secrets(ctx, tmp_path):
    (tmp_path / ".env").write_text("API_KEY=hunter2")
    with pytest.raises(ToolError, match="secret-shaped"):
        await call(GrepTool(), {"pattern": ".*", "path": ".env", "output_mode": "content"}, ctx)


async def test_grep_on_a_named_file_still_skips_binaries(ctx, tmp_path):
    (tmp_path / "img.bin").write_bytes(b"\x00foo")
    with pytest.raises(ToolError, match="not a text file"):
        await call(GrepTool(), {"pattern": "foo", "path": "img.bin"}, ctx)


async def test_grep_on_a_named_file(ctx, tmp_path):
    (tmp_path / "a.py").write_text("foo")
    assert "a.py" in await call(GrepTool(), {"pattern": "foo", "path": "a.py"}, ctx)


@pytest.fixture
def credentials(tmp_path_factory):
    """A credentials file outside the project, as ~/.aws/credentials would be."""
    home = tmp_path_factory.mktemp("home")
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("aws_secret_access_key = SECRET123\n")
    return home / ".aws"


@pytest.mark.parametrize("mode", ["content", "files_with_matches", "count"])
async def test_grep_does_not_follow_a_symlink_to_a_secret(ctx, tmp_path, credentials, mode):
    """The secret check sees the link's own name ("leak"), so following it would leak."""
    (tmp_path / "leak").symlink_to(credentials / "credentials")
    (tmp_path / "app.py").write_text("x = 1\n")
    out = await call(GrepTool(), {"pattern": ".", "output_mode": mode}, ctx)
    assert "SECRET123" not in out and "leak" not in out and "app.py" in out


async def test_grep_does_not_walk_into_a_symlinked_directory(ctx, tmp_path, credentials):
    (tmp_path / "config").symlink_to(credentials)
    (tmp_path / "app.py").write_text("key = 1\n")
    out = await call(GrepTool(), {"pattern": "SECRET|key", "output_mode": "content"}, ctx)
    assert "SECRET123" not in out and "app.py" in out


def test_a_file_swapped_for_a_symlink_after_listing_is_skipped(tmp_path, credentials):
    from tinyharness.tools.grep import _iter_searchable, _load

    (tmp_path / "notes.txt").write_text("fine\n")
    [listed] = list(_iter_searchable(tmp_path.resolve(), None))
    listed.unlink()
    listed.symlink_to(credentials / "credentials")  # between the walk and the read
    assert _load(listed) is None


# ---- TodoWrite ----

TODOS = [
    {"content": "Read input file", "activeForm": "Reading input file", "status": "completed"},
    {"content": "Write the parser", "activeForm": "Writing the parser", "status": "in_progress"},
    {"content": "Add tests", "activeForm": "Adding tests", "status": "pending"},
]


async def test_todo_write_stores_and_formats(ctx):
    out = await call(TodoWriteTool(), {"todos": TODOS}, ctx)
    assert ctx.todos == TODOS
    assert "1/3 done" in out
    assert "Writing the parser" in out
    assert "✔" in out and "◐" in out and "○" in out


def test_todo_rejects_bad_status():
    with pytest.raises(pydantic.ValidationError):
        TodoWriteTool().parse({"todos": [{"content": "x", "activeForm": "y", "status": "blocked"}]})


def test_todo_rejects_empty_content():
    with pytest.raises(pydantic.ValidationError):
        TodoWriteTool().parse({"todos": [{"content": "", "activeForm": "y", "status": "pending"}]})


def test_format_todos_empty():
    assert "cleared" in format_todos([])


# ---- Registry / gate integration ----


def test_the_core_registry_is_the_file_tools():
    """Everything else (web, pdf, open, todo, agents) is an extension you turn on."""
    assert default_registry().names() == ["Read", "Write", "Edit", "Glob", "Grep"]


def test_plan_mode_offers_only_read_only_tools():
    names = {t.name for t in default_registry().available(PermissionMode.PLAN)}
    assert names == {"Read", "Grep", "Glob"}


async def test_todowrite_skips_the_gate(tmp_path):
    gate = PermissionGate(workspace=Workspace(tmp_path))
    d = await gate.check(TodoWriteTool(), {"todos": []}, target_path=None)
    assert d.allow and d.source == "tool"
