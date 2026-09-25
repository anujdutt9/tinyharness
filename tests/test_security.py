import pytest

from conftest import call, make_ctx
from tinyharness.security import redact, sanitized_env
from tinyharness.tools.base import ToolError
from tinyharness.tools.read import ReadTool
from tinyharness.workspace import Workspace


def test_sanitized_env_strips_credentials(monkeypatch):
    secrets = {
        "OPENROUTER_API_KEY": "sk-a",
        "OPENAI_API_KEY": "sk-b",
        "GH_TOKEN": "ghp_c",
        "AWS_SECRET_ACCESS_KEY": "wj/d",
        "AWS_SESSION_TOKEN": "FQoG_e",
        "MY_PASSWORD": "pw",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("PATH", "/usr/bin")
    env = sanitized_env()
    assert not set(secrets) & set(env)
    assert env["PATH"] == "/usr/bin"


@pytest.mark.parametrize("path", [".env", ".env.local", ".ssh/id_rsa", ".aws/credentials", ".netrc"])
async def test_read_refuses_secret_shaped_paths(ctx, tmp_path, path):
    target = tmp_path / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("secret")
    with pytest.raises(ToolError, match="secret-shaped"):
        await call(ReadTool(), {"path": path}, ctx)


async def test_secret_reads_can_be_opted_into(ctx, tmp_path, monkeypatch):
    monkeypatch.setenv("TINYHARNESS_ALLOW_SECRET_READS", "1")
    (tmp_path / ".env").write_text("FOO=bar")
    assert "FOO=bar" in await call(ReadTool(), {"path": ".env"}, ctx)


async def test_read_follows_registered_read_roots(tmp_path):
    project, skill = tmp_path / "proj", tmp_path / "skills" / "demo"
    project.mkdir()
    skill.mkdir(parents=True)
    (skill / "helper.py").write_text("print('hi')\n")
    ctx = make_ctx(project, Workspace(project, read_roots=[skill]))
    assert "print('hi')" in await call(ReadTool(), {"path": str(skill / "helper.py")}, ctx)


def test_redact():
    assert redact("") == "(missing)"
    assert redact("short") == "(set)"
    assert redact("sk-or-v1-1234567890abcdef") == "sk-or-v1...cdef"
