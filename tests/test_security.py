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


# ---- keeping keys out of the model's context ----

from tinyharness.security import find_api_key, mask_secrets  # noqa: E402


def test_mask_secrets_replaces_configured_key_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-1234567890abcdef")
    text = "config: key = 'sk-live-1234567890abcdef' # oops"
    assert mask_secrets(text) == "config: key = '[REDACTED OPENAI_API_KEY]' # oops"


def test_mask_secrets_ignores_short_values(monkeypatch):
    monkeypatch.setenv("SOME_TOKEN", "abc")
    assert mask_secrets("abc abc") == "abc abc"


def test_mask_secrets_includes_keys_passed_directly():
    assert mask_secrets("x sdk-key-000000000000 y", {"api_key": "sdk-key-000000000000"}) == (
        "x [REDACTED api_key] y"
    )


@pytest.mark.parametrize(
    "text",
    [
        "my key is sk-ant-api03-" + "a" * 40,
        "sk-or-v1-" + "0123abcd" * 8,
        "OPENAI_API_KEY=sk-proj-" + "Ab1_" * 10,
        "AIza" + "B" * 35,
        "gsk_" + "c" * 30,
        "xai-" + "d" * 30,
    ],
)
def test_find_api_key_recognizes_common_formats(text):
    preview = find_api_key(text)
    assert preview is not None and "..." in preview


@pytest.mark.parametrize(
    "text",
    ["what does the sk- prefix mean?", "use task-scheduler for this", "AIza is Google's prefix"],
)
def test_find_api_key_ignores_ordinary_text(text):
    assert find_api_key(text) is None


def test_find_api_key_catches_a_configured_key_without_a_prefix(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "Zm9vYmFyYmF6cXV4MTIzNDU2")
    assert find_api_key("here: Zm9vYmFyYmF6cXV4MTIzNDU2") is not None
