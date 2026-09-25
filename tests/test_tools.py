import pytest

from conftest import call
from tinyharness.tools.base import ToolError
from tinyharness.tools.edit import EditTool
from tinyharness.tools.read import ReadTool
from tinyharness.tools.write import WriteTool


async def test_write_then_read(ctx, tmp_path):
    await call(WriteTool(), {"path": "a.txt", "content": "hello\nworld\n"}, ctx)
    out = await call(ReadTool(), {"path": "a.txt"}, ctx)
    assert "hello" in out and "world" in out


async def test_read_numbers_lines_and_pages(ctx, tmp_path):
    (tmp_path / "f.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
    out = await call(ReadTool(), {"path": "f.txt", "offset": 2, "limit": 3}, ctx)
    assert out.splitlines()[0] == "     3\tline3"
    assert "(5 more lines)" in out


async def test_overwrite_requires_read(ctx, tmp_path):
    (tmp_path / "x.txt").write_text("orig")
    with pytest.raises(ToolError, match="Read it first"):
        await call(WriteTool(), {"path": "x.txt", "content": "new"}, ctx)
    await call(ReadTool(), {"path": "x.txt"}, ctx)
    assert "wrote" in await call(WriteTool(), {"path": "x.txt", "content": "new"}, ctx)


async def test_edit_requires_unique_match(ctx, tmp_path):
    await call(WriteTool(), {"path": "a.txt", "content": "foo bar foo"}, ctx)  # Write counts as read
    with pytest.raises(ToolError, match="not unique"):
        await call(EditTool(), {"path": "a.txt", "old_string": "foo", "new_string": "baz"}, ctx)
    msg = await call(
        EditTool(), {"path": "a.txt", "old_string": "foo", "new_string": "baz", "replace_all": True}, ctx
    )
    assert "2 replacements" in msg
    assert (tmp_path / "a.txt").read_text() == "baz bar baz"


async def test_edit_requires_read_first(ctx, tmp_path):
    (tmp_path / "a.txt").write_text("foo")
    with pytest.raises(ToolError, match="before editing"):
        await call(EditTool(), {"path": "a.txt", "old_string": "foo", "new_string": "bar"}, ctx)


async def test_edit_missing_file(ctx):
    with pytest.raises(ToolError, match="file not found"):
        await call(EditTool(), {"path": "/nonexistent.txt", "old_string": "a", "new_string": "b"}, ctx)


def test_schema_is_inlined_and_untitled():
    from tinyharness.tools.todo import TodoWriteTool

    schema = TodoWriteTool().input_schema()
    text = str(schema)
    assert "$ref" not in text and "$defs" not in text
    assert "title" not in schema["properties"]["todos"]["items"]
    assert "activeForm" in schema["properties"]["todos"]["items"]["properties"]
