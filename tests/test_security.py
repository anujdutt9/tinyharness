import pytest

from conftest import call, make_ctx
from tinyharness.security import is_secret_path, redact, sanitized_env
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


@pytest.mark.parametrize("name", [".env.example", ".env.sample", "app/.env.template", ".env.dist"])
def test_env_templates_are_not_secrets(name):
    assert not is_secret_path(name)


@pytest.mark.parametrize("name", [".env", ".env.local", "app/.env.production", ".env.example.bak"])
def test_real_env_files_are_secrets(name):
    assert is_secret_path(name)


# ---- session logs ----


def test_session_logs_mask_credentials_but_keep_reasoning_intact(tmp_path, monkeypatch):
    import json
    import stat

    from tinyharness.agent.transcript import Transcript
    from tinyharness.paths import state_dir

    monkeypatch.setenv("OPENAI_API_KEY", "sk-configured0123456789abcdef")
    pasted = "sk-or-v1-" + "a1b2c3d4" * 4
    blob = "gAAA_sk-" + "x" * 30  # a signed reasoning blob that happens to look key-shaped
    with Transcript(tmp_path, "s1") as log:
        log.write("user", text=f"use {pasted} please")
        log.write("tool_use", tool="Bash", args={"command": "echo sk-configured0123456789abcdef"})
        log.write("msg", message={"content": "hi", "state": {"owner": "o", "data": {"sig": blob}}})
    text = log.path.read_text()
    assert pasted not in text and "sk-configured0123456789abcdef" not in text
    assert "[REDACTED API KEY]" in text and "[REDACTED OPENAI_API_KEY]" in text
    last = json.loads(text.splitlines()[-1])
    assert last["schema"] == 1 and last["session"] == "s1"
    assert last["harness_version"]
    assert {"turn", "step", "agent", "model_call", "routing_decision", "tool_call"} <= last.keys()
    assert last["message"]["state"]["data"]["sig"] == blob  # resuming must send it back unchanged
    assert stat.S_IMODE(log.path.stat().st_mode) == 0o600
    assert stat.S_IMODE((state_dir(tmp_path) / "sessions").stat().st_mode) == 0o700


def test_existing_log_directories_are_made_private(tmp_path):
    import stat

    from tinyharness.agent.transcript import Transcript
    from tinyharness.paths import state_dir

    old = state_dir(tmp_path) / "sessions"
    old.mkdir(parents=True)
    old.chmod(0o755)  # as earlier versions left it
    Transcript(tmp_path, "s2").close()
    assert stat.S_IMODE(old.stat().st_mode) == 0o700


@pytest.mark.parametrize("path", ["/home/me/.tinyharness/auth.json", "project/auth.json"])
def test_key_files_named_auth_json_are_secret(path):
    assert is_secret_path(path)
