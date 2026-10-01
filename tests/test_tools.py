from pathlib import Path

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
    assert out.splitlines()[-1] == "... (showing lines 3-5 of 10; continue with offset=5)"


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


# ---- found by the robustness audit (each was reproduced against the old code) ----


async def _read_then_edit(ctx, tmp_path, name, content, old, new, **extra):
    (tmp_path / name).write_bytes(content)
    await call(ReadTool(), {"path": name}, ctx)
    msg = await call(EditTool(), {"path": name, "old_string": old, "new_string": new, **extra}, ctx)
    return (tmp_path / name).read_bytes(), msg


async def test_a_loose_match_adds_no_blank_line(ctx, tmp_path):
    # old_string has a trailing space the file doesn't, and ends with a newline like new_string
    data, msg = await _read_then_edit(
        ctx,
        tmp_path,
        "f.py",
        b"def f():\n    return 1\nprint(f())\n",
        "def f(): \n    return 1\n",
        "def f():\n    return 2\n",
    )
    assert data == b"def f():\n    return 2\nprint(f())\n" and "ignoring trailing whitespace" in msg


async def test_a_crlf_file_keeps_its_line_endings(ctx, tmp_path):
    data, _ = await _read_then_edit(
        ctx, tmp_path, "w.txt", b"one\r\ntwo\r\nthree\r\n", "one\ntwo\n", "ONE\nTWO\n"
    )
    assert data == b"ONE\r\nTWO\r\nthree\r\n"


async def test_an_empty_old_string_is_refused(ctx, tmp_path):
    (tmp_path / "e.txt").write_text("abc\n")
    await call(ReadTool(), {"path": "e.txt"}, ctx)
    with pytest.raises(ToolError, match="old_string is empty"):
        await call(
            EditTool(), {"path": "e.txt", "old_string": "", "new_string": "X", "replace_all": True}, ctx
        )
    assert (tmp_path / "e.txt").read_text() == "abc\n"  # used to become "XaXbXcX\nX"


async def test_an_empty_file_can_be_filled(ctx, tmp_path):
    data, _ = await _read_then_edit(ctx, tmp_path, "new.txt", b"", "", "hello\n")
    assert data == b"hello\n"


async def test_reads_line_numbers_in_old_string_are_left_out(ctx, tmp_path):
    data, msg = await _read_then_edit(
        ctx,
        tmp_path,
        "n.py",
        b"x = 1\ny = 2\n",
        "     1\tx = 1\n     2\ty = 2",
        "     1\tx = 10\n     2\ty = 20",
    )
    assert data == b"x = 10\ny = 20\n" and "line numbers were left out" in msg


async def test_a_file_that_isnt_utf8_gets_a_clear_error(ctx, tmp_path):
    (tmp_path / "l.txt").write_bytes("café\n".encode("latin-1"))
    await call(ReadTool(), {"path": "l.txt"}, ctx)
    with pytest.raises(ToolError, match="isn't UTF-8 text"):
        await call(EditTool(), {"path": "l.txt", "old_string": "caf", "new_string": "CAF"}, ctx)


async def test_grep_anchors_work_in_every_mode(ctx, tmp_path):
    from tinyharness.tools.grep import GrepTool

    (tmp_path / "m.py").write_text("import os\nimport sys\n\ndef main():\n    pass\n")
    found = await call(GrepTool(), {"pattern": "^def main"}, ctx)
    counted = await call(GrepTool(), {"pattern": "^import", "output_mode": "count"}, ctx)
    assert "1 file(s) matched" in found  # was "no files matched"
    assert counted.startswith("2 total")  # was 1


async def test_grep_says_when_it_hit_the_limit_and_caps_long_lines(ctx, tmp_path):
    from tinyharness.tools.grep import GrepTool

    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("needle")
    (tmp_path / "min.js").write_text("needle" + "x" * 5000)
    listed = await call(GrepTool(), {"pattern": "needle", "head_limit": 2}, ctx)
    assert "head_limit reached" in listed
    content = await call(GrepTool(), {"pattern": "needle", "include": "*.js", "output_mode": "content"}, ctx)
    assert len(content) < 600 and "more chars" in content


async def test_glob_skips_dependency_directories_and_explains_absolute_patterns(ctx, tmp_path):
    from tinyharness.tools.glob import GlobTool

    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "a.py").write_text("")
    (tmp_path / "src.py").write_text("")
    out = await call(GlobTool(), {"pattern": "**/*.py"}, ctx)
    assert "src.py" in out and "node_modules" not in out
    with pytest.raises(ToolError, match="must be relative"):
        await call(GlobTool(), {"pattern": str(tmp_path / "*.py")}, ctx)


async def test_read_pages_large_files_by_size_and_says_how_to_continue(ctx, tmp_path):
    """2000 lines of code is ~26k tokens; the old Read sent it all, and compaction then
    cut it silently in the middle."""
    (tmp_path / "big.py").write_text("".join(f"value_{i} = {'x' * 60}\n" for i in range(2000)))
    out = await call(ReadTool(), {"path": "big.py"}, ctx)
    assert len(out) <= 52_000
    last = out.splitlines()[-1]
    assert last.startswith("... (showing lines 1-") and "of 2000; continue with offset=" in last
    offset = int(last.rsplit("=", 1)[1].rstrip(")"))
    nxt = await call(ReadTool(), {"path": "big.py", "offset": offset}, ctx)
    assert nxt.splitlines()[0].startswith(f"{offset + 1:>6}\t")


async def test_read_explains_an_offset_past_the_end_and_splits_lines_like_edit(ctx, tmp_path):
    (tmp_path / "s.txt").write_text("a\fb\nc\n")
    out = await call(ReadTool(), {"path": "s.txt"}, ctx)
    assert out == "     1\ta\fb\n     2\tc"  # a form feed doesn't start a line
    assert "past the end" in await call(ReadTool(), {"path": "s.txt", "offset": 5}, ctx)


async def test_bash_timeout_covers_a_command_that_closed_its_output(ctx, tmp_path):
    import time

    from tinyharness.sandbox.policy import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    start = time.monotonic()
    out = await call(
        BashTool(None, SandboxPolicy(write_roots=(tmp_path,))),
        {"command": "exec >out.txt 2>&1; sleep 8", "timeout": 1},
        ctx,
    )
    assert time.monotonic() - start < 4 and "timed out" in out


async def test_bash_is_non_interactive_and_keeps_omitted_output(ctx, tmp_path):
    from tinyharness.sandbox.policy import SandboxPolicy
    from tinyharness.tools.bash import BashTool
    from tinyharness.workspace.scratch import Scratch

    ctx.scratch = Scratch(tmp_path, "s1")
    bash = BashTool(None, SandboxPolicy(write_roots=(tmp_path,)))
    env = await call(bash, {"command": "echo $PAGER $GIT_PAGER $GIT_TERMINAL_PROMPT"}, ctx)
    assert env.splitlines()[0] == "cat cat 0"
    out = await call(bash, {"command": "seq 1 20000"}, ctx)
    path = out.split("all of it is in ", 1)[1].split(":", 1)[0]
    assert Path(path).read_text().splitlines()[9999] == "10000"  # the omitted middle is there


async def test_a_tsv_row_starting_with_a_number_isnt_taken_for_a_line_number(ctx, tmp_path):
    (tmp_path / "t.tsv").write_bytes(b"1\tapple\n2\tbanana\n")
    await call(ReadTool(), {"path": "t.tsv"}, ctx)
    with pytest.raises(ToolError, match="not found"):
        await call(EditTool(), {"path": "t.tsv", "old_string": "3\tbanana", "new_string": "3\tcherry"}, ctx)
    assert (tmp_path / "t.tsv").read_bytes() == b"1\tapple\n2\tbanana\n"


async def test_line_numbers_point_at_the_line_they_name(ctx, tmp_path):
    (tmp_path / "d.py").write_text("x = 1\nx = 1\n")
    await call(ReadTool(), {"path": "d.py"}, ctx)
    await call(EditTool(), {"path": "d.py", "old_string": "     2\tx = 1\n", "new_string": "x = 2\n"}, ctx)
    assert (tmp_path / "d.py").read_text() == "x = 1\nx = 2\n"


async def test_a_list_for_a_pattern_is_an_error_not_a_regex(tmp_path):
    from pydantic import ValidationError

    from tinyharness.tools.grep import GrepTool

    with pytest.raises(ValidationError):
        GrepTool().parse({"pattern": ["foo", "bar"]})


async def test_read_caps_a_page_by_bytes(ctx, tmp_path):
    (tmp_path / "u.txt").write_text(("é" * 100 + "\n") * 2000)
    out = await call(ReadTool(), {"path": "u.txt"}, ctx)
    assert len(out.encode()) <= 52_000


async def test_a_page_that_links_chart_js_gets_it_embedded_on_write(ctx, tmp_path):
    """Seen live: a model wrote the <script src="chart.umd.min.js"> tag, skipped the step
    that inlines it, and every chart in the report was empty."""
    page = '<html><head><script src="chart.umd.min.js"></script></head><body><canvas id="c"></canvas></body></html>'
    out = await call(WriteTool(), {"path": "report.html", "content": page}, ctx)
    html = (tmp_path / "report.html").read_text()
    assert "Chart.js embedded" in out
    assert 'src="chart.umd.min.js"' not in html and "Chart" in html and len(html) > 100_000


async def test_a_server_in_the_sandbox_points_to_screenshot(ctx, tmp_path):
    from tinyharness.sandbox.policy import SandboxPolicy
    from tinyharness.tools.bash import BashInput, BashTool

    tool = BashTool(object(), SandboxPolicy(write_roots=(tmp_path,)))  # a sandbox is present
    text = tool._report(
        "Couldn't connect to server",
        7,
        False,
        BashInput(command="curl -I http://localhost:8000/r.html"),
        True,
    )
    assert "use the Screenshot tool" in text and text.endswith("(exit code 7)")
