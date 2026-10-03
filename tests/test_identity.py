"""Who-is-calling headers (llm/identity.py): sent only to providers that opt in, one
identity per session, a child identity per subagent, never mixed across parallel ones."""

import asyncio

import pytest

from tinyharness import Session, SessionOptions
from tinyharness.llm import identity
from tinyharness.llm.client import LLMClient
from tinyharness.llm.providers import Provider


def client(send_ids: bool) -> LLMClient:
    return LLMClient(Provider(name="gw", base_url="http://127.0.0.1:8080/v1", send_ids=send_ids), "", "m")


def test_headers_go_only_to_providers_that_ask():
    token = identity.CURRENT.set(identity.CallIdentity(session="s1", agent="s1-lead"))
    try:
        assert client(send_ids=True)._headers()["X-Session-ID"] == "s1"
        assert client(send_ids=True)._headers()["X-Agent-Role"] == "lead"
        assert "X-Session-ID" not in client(send_ids=False)._headers()
    finally:
        identity.CURRENT.reset(token)


def test_a_subagent_is_a_child_of_its_parent(monkeypatch):
    monkeypatch.setenv("TH_TASK", "sql-review")
    lead = identity.for_session("s1")
    sub = identity.child(lead, "sql-analyst")
    assert (sub.session, sub.parent, sub.role, sub.name, sub.task) == (
        "s1",
        "s1-lead",
        "subagent",
        "sql-analyst",
        "sql-review",
    )
    assert sub.headers()["X-Parent-Agent-ID"] == "s1-lead" and sub.headers()["X-Task"] == "sql-review"
    assert "X-Parent-Agent-ID" not in lead.headers()


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.toml"
    config.write_text('[extensions]\nenabled = ["agents"]\n')
    s = Session(SessionOptions(config_path=config, working_dir=tmp_path / "work"))
    yield s
    s.close()


async def test_parallel_subagents_each_call_as_themselves(session, monkeypatch):
    seen = []

    async def fake_run(spec, task, **kwargs):
        await asyncio.sleep(0.01)  # overlap with the other one
        seen.append(identity.CURRENT.get())
        return "ok"

    monkeypatch.setattr("tinyharness.session.run_subagent", fake_run)
    lead = session._identify()
    spec = session.agents["general-purpose"]
    await asyncio.gather(*(session._spawn_subagent(spec, "x", session.ctx) for _ in range(2)))
    assert {who.parent for who in seen} == {lead.agent}
    assert len({who.agent for who in seen}) == 2  # two teammates, two ids
    assert identity.CURRENT.get() == lead  # back to the lead afterwards
