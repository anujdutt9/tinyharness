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


# ---- Agent ----


def agent_tool(**agents):
    from tinyharness.subagents import AgentSpec
    from tinyharness.tools.agent import AgentTool

    ran = []

    async def spawner(spec, prompt, ctx, *, label=""):
        ran.append(spec.name)
        return "done"

    specs = {name: AgentSpec(name, "d") for name in agents or ("explore", "general-purpose")}
    return AgentTool(specs, spawner), ran


async def test_agent_without_subagent_type_runs_general_purpose(tmp_path):
    """Models often leave it out (other harnesses default it); that used to fail the call."""
    from conftest import make_ctx

    tool, ran = agent_tool()
    args = tool.parse({"prompt": "Verify the deck", "description": "Verify deck"})
    assert tool.describe(args) == "Agent(general-purpose: Verify deck)"
    assert await tool.run(args, make_ctx(tmp_path)) == "done" and ran == ["general-purpose"]


async def test_agent_without_general_purpose_falls_back_to_the_first(tmp_path):
    from conftest import make_ctx

    tool, ran = agent_tool(researcher=1, explore=1)
    await tool.run(tool.parse({"prompt": "x"}), make_ctx(tmp_path))
    assert ran == ["researcher"]


from conftest import png  # noqa: E402

PNG = png(40, 20)


async def test_read_refuses_binary_files(tmp_path):
    """An archive's bytes as "text" would be thousands of tokens of noise for the model."""
    from conftest import make_ctx

    (tmp_path / "bundle.zip").write_bytes(b"PK\x03\x04\x00\x00" + bytes(64))
    with pytest.raises(ToolError, match=r"can.t read bundle\.zip: it.s a binary file"):
        await call(ReadTool(), {"path": "bundle.zip"}, make_ctx(tmp_path))


@pytest.mark.parametrize("name", ["shot.png", "shot.txt"])  # the bytes decide, not the name
async def test_read_shows_an_image_to_the_model(tmp_path, name):
    import base64

    from conftest import make_ctx
    from tinyharness.tools.base import ToolOutput

    (tmp_path / name).write_bytes(PNG)
    ctx = make_ctx(tmp_path)
    out = await call(ReadTool(), {"path": name}, ctx)
    assert isinstance(out, ToolOutput) and out.text.startswith(f"{name}: image/png image")
    [image] = out.images
    assert (image.media_type, base64.b64decode(image.data), image.name) == ("image/png", PNG, name)
    assert str((tmp_path / name).resolve()) in ctx.read_paths


async def test_read_shrinks_a_retina_screenshot(tmp_path):
    """3024 px wide is over what providers take once a request has several images."""
    import base64
    import io

    from PIL import Image as PILImage

    from conftest import make_ctx

    (tmp_path / "retina.png").write_bytes(png(3024, 1964))
    out = await call(ReadTool(), {"path": "retina.png"}, make_ctx(tmp_path))
    [image] = out.images
    assert PILImage.open(io.BytesIO(base64.b64decode(image.data))).size == (2000, 1299)
    assert "shown to you at most 2000 px a side" in out.text


async def test_read_refuses_a_broken_image(tmp_path):
    from conftest import make_ctx

    (tmp_path / "broken.png").write_bytes(b"\x89PNG\r\n\x1a\n" + bytes(64))
    with pytest.raises(ToolError, match=r"can't view broken\.png: not a readable image"):
        await call(ReadTool(), {"path": "broken.png"}, make_ctx(tmp_path))


def test_images_survive_the_session_log():
    from tinyharness.agent.history import Image, Message

    m = Message(role="user", content="slide 1", images=[Image("image/png", "iVBO", "shot.png")])
    assert Message.from_record(m.to_record()).images == m.images


async def test_edit_tolerates_trailing_whitespace_only(ctx, tmp_path):
    # Seen live: a blank line written as four spaces failed 11 edits in a row.
    code = "def f():\n    x = 1\n\n    return x\n"
    await call(WriteTool(), {"path": "a.py", "content": code}, ctx)
    old = "    x = 1\n    \n    return x"  # the blank line has spaces; the file's doesn't
    msg = await call(EditTool(), {"path": "a.py", "old_string": old, "new_string": "    return 1"}, ctx)
    assert "matched ignoring trailing whitespace" in msg
    assert (tmp_path / "a.py").read_text() == "def f():\n    return 1\n"


async def test_edit_does_not_ignore_indentation(ctx, tmp_path):
    await call(WriteTool(), {"path": "a.py", "content": "if x:\n    y()\n"}, ctx)
    with pytest.raises(ToolError, match="not found"):
        await call(EditTool(), {"path": "a.py", "old_string": "if x:\n  y()", "new_string": "z"}, ctx)


async def test_a_failed_edit_says_where_it_stopped_matching(ctx, tmp_path):
    lines = [f"line {i}" for i in range(1, 9)]
    await call(WriteTool(), {"path": "a.txt", "content": "\n".join(lines)}, ctx)
    old = "line 3\nline 4\nline five\nline 6"
    with pytest.raises(ToolError) as e:
        await call(EditTool(), {"path": "a.txt", "old_string": old, "new_string": "x"}, ctx)
    message = str(e.value)
    assert "closest match starts at line 3" in message and "lines 3-4 match" in message
    assert "'line five'" in message and "'line 5'" in message


async def test_a_failed_edit_whose_first_line_is_nowhere_says_so(ctx, tmp_path):
    await call(WriteTool(), {"path": "a.txt", "content": "alpha\nbeta\n"}, ctx)
    with pytest.raises(ToolError, match="first line isn't in the file"):
        await call(EditTool(), {"path": "a.txt", "old_string": "gamma\nbeta", "new_string": "x"}, ctx)


async def test_edit_refuses_a_file_that_changed_since_it_was_read(ctx, tmp_path):
    await call(WriteTool(), {"path": "a.txt", "content": "one\ntwo\n"}, ctx)
    (tmp_path / "a.txt").write_text("one\nTWO\n")  # a command, a teammate or the user
    with pytest.raises(ToolError, match="changed since you last read it"):
        await call(EditTool(), {"path": "a.txt", "old_string": "one", "new_string": "1"}, ctx)
    await call(ReadTool(), {"path": "a.txt"}, ctx)
    await call(EditTool(), {"path": "a.txt", "old_string": "one", "new_string": "1"}, ctx)
    # its own edit doesn't make the file "changed since read"
    await call(EditTool(), {"path": "a.txt", "old_string": "TWO", "new_string": "2"}, ctx)
    assert (tmp_path / "a.txt").read_text() == "1\n2\n"


async def test_an_edit_shows_the_changed_lines_as_they_are_now(ctx, tmp_path):
    content = "\n".join(f"line {i}" for i in range(1, 21))
    await call(WriteTool(), {"path": "a.txt", "content": content}, ctx)
    msg = await call(EditTool(), {"path": "a.txt", "old_string": "line 10", "new_string": "ten\nTEN"}, ctx)
    shown = msg.split("as they are now:\n", 1)[1].splitlines()
    assert shown[0] == "     7\tline 7" and "    10\tten" in shown and "    11\tTEN" in shown
    assert shown[-1] == "    14\tline 13"  # three lines after the change


def _agent_ctx(root):
    from conftest import make_ctx

    return make_ctx(root)


async def test_another_agents_change_counts_as_unseen(tmp_path):
    # Two agents (a lead and a teammate) each with their own view of the file.
    from tinyharness.workspace import Workspace

    workspace = Workspace(tmp_path)
    lead, teammate = _agent_ctx(tmp_path), _agent_ctx(tmp_path)
    lead.workspace = teammate.workspace = workspace
    (tmp_path / "a.txt").write_text("alpha\nbeta\n")
    await call(ReadTool(), {"path": "a.txt"}, lead)
    await call(ReadTool(), {"path": "a.txt"}, teammate)
    await call(EditTool(), {"path": "a.txt", "old_string": "alpha", "new_string": "ALPHA"}, teammate)
    with pytest.raises(ToolError, match="changed since you last read it"):
        await call(EditTool(), {"path": "a.txt", "old_string": "beta", "new_string": "BETA"}, lead)
    with pytest.raises(ToolError, match="changed since you last read it"):  # no lost update either
        await call(WriteTool(), {"path": "a.txt", "content": "alpha\nBETA\n"}, lead)
    await call(ReadTool(), {"path": "a.txt"}, lead)
    await call(EditTool(), {"path": "a.txt", "old_string": "beta", "new_string": "BETA"}, lead)
    assert (tmp_path / "a.txt").read_text() == "ALPHA\nBETA\n"
