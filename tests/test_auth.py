"""Provider browser and device authentication."""

import json
import time
import urllib.request
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from tinyharness.config import read_chatgpt_profile, save_chatgpt_profile
from tinyharness.llm.auth import (
    OPENAI_ISSUER,
    AuthenticationError,
    _callback_code,
    _chatgpt_callback,
    chatgpt_browser_login,
    exchange_github_copilot_token,
    exchange_openrouter_code,
    finish_github_copilot_login,
    openrouter_authorization,
    openrouter_browser_login,
    start_github_copilot_login,
)
from tinyharness.llm.client import LLMClient
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.providers import load_catalog


def test_openrouter_authorization_uses_pkce_and_supports_headless_mode():
    browser = openrouter_authorization("http://127.0.0.1:4321/callback")
    query = parse_qs(urlsplit(browser.url).query)
    assert query["callback_url"] == ["http://127.0.0.1:4321/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"] == [browser.state]
    assert query["key_label"] == ["tinyharness"]
    assert query["code_challenge"][0] != browser.verifier

    headless = openrouter_authorization(None)
    assert "callback_url" not in parse_qs(urlsplit(headless.url).query)


def test_callback_rejects_a_forged_state_and_provider_error():
    assert _callback_code("/callback?code=one&state=expected", "expected") == "one"
    with pytest.raises(AuthenticationError, match="invalid state"):
        _callback_code("/callback?code=one&state=forged", "expected")
    with pytest.raises(AuthenticationError, match="declined"):
        _callback_code("/callback?error=access_denied&state=expected", "expected")


def test_chatgpt_callback_requires_the_exact_path_and_state():
    assert _chatgpt_callback("/auth/callback?code=one&client_id=oaiapp_one&state=expected", "expected") == {
        "code": "one",
        "client_id": "oaiapp_one",
    }
    with pytest.raises(AuthenticationError, match="callback path"):
        _chatgpt_callback("/callback?code=one&state=expected", "expected")
    with pytest.raises(AuthenticationError, match="invalid state"):
        _chatgpt_callback("/auth/callback?code=one&state=forged", "expected")


async def test_chatgpt_login_registers_validates_and_returns_a_renewable_profile():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk.update(kid="key-1", use="sig", alg="RS256")
    authorization: dict[str, str] = {}
    registrations = []

    def opener(url: str) -> bool:
        query = parse_qs(urlsplit(url).query)
        authorization.update({key: values[0] for key, values in query.items()})
        assert authorization["client_id"] == "dynamic_agent_client"
        assert authorization["ext_agent_host_id"] == "urn:uuid:host"
        assert authorization["agent_name_hint"] == "tinyharness"
        assert authorization["scope"].split() == [
            "openid",
            "profile",
            "email",
            "offline_access",
            "resource.invoke",
            "chatgpt.tokens.use.direct",
        ]
        callback = (
            authorization["redirect_uri"]
            + "?"
            + urlencode({"code": "one-use", "state": authorization["state"], "client_id": "oaiapp_tiny"})
        )
        with urllib.request.urlopen(callback) as response:
            assert response.status == 200
        return True

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/accounts/oauth/token":
            form = parse_qs(request.content.decode())
            assert form["client_id"] == ["oaiapp_tiny"]
            assert form["code"] == ["one-use"]
            assert form["redirect_uri"] == [authorization["redirect_uri"]]
            token = jwt.encode(
                {
                    "iss": OPENAI_ISSUER,
                    "aud": "oaiapp_tiny",
                    "sub": "account-1",
                    "email": "person@example.com",
                    "nonce": authorization["nonce"],
                    "exp": int(time.time()) + 3600,
                },
                private_key,
                algorithm="RS256",
                headers={"kid": "key-1"},
            )
            return httpx.Response(
                200,
                json={
                    "access_token": "access-1",
                    "refresh_token": "refresh-1",
                    "id_token": token,
                    "expires_in": 3600,
                    "scope": "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct",
                },
            )
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": OPENAI_ISSUER,
                    "jwks_uri": "https://auth.openai.com/.well-known/jwks.json",
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
            )
        return httpx.Response(200, json={"keys": [public_jwk]})

    profile = await chatgpt_browser_login(
        "urn:uuid:host",
        opener=opener,
        on_registration=registrations.append,
        transport=httpx.MockTransport(handler),
    )

    assert registrations == ["oaiapp_tiny"]
    assert profile["client_id"] == "oaiapp_tiny"
    assert profile["subject"] == "account-1" and profile["email"] == "person@example.com"
    assert profile["access_token"] == "access-1" and profile["refresh_token"] == "refresh-1"
    assert profile["expires_at"] > time.time()


async def test_chatgpt_client_rotates_and_atomically_saves_the_refresh_token(tmp_path):
    """A rotating refresh token must be reloaded under the cross-process lock and saved
    together with its replacement before the request uses the new access token."""
    profile = {
        "client_id": "oaiapp_tiny",
        "ext_agent_host_id": "urn:uuid:host",
        "issuer": OPENAI_ISSUER,
        "subject": "account-1",
        "access_token": "expired-access",
        "refresh_token": "refresh-1",
        "id_token": "retained-id-token",
        "expires_at": 1,
        "scopes": ["chatgpt.tokens.use.direct"],
    }
    save_chatgpt_profile(profile)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/accounts/oauth/token":
            form = parse_qs(request.content.decode())
            assert form == {
                "grant_type": ["refresh_token"],
                "client_id": ["oaiapp_tiny"],
                "refresh_token": ["refresh-1"],
                "resource": ["https://api.openai.com/v1"],
            }
            return httpx.Response(
                200,
                json={
                    "access_token": "access-2",
                    "refresh_token": "refresh-2",
                    "expires_in": 3600,
                    "scope": "chatgpt.tokens.use.direct",
                },
            )
        seen.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "gpt-live", "display_name": "GPT Live", "visibility": "list"},
                    {"slug": "hidden", "display_name": "Hidden", "visibility": "hidden"},
                ]
            },
        )

    provider = load_catalog()["chatgpt"]
    provider.set_auth_data(profile)
    provider.set_api_key("expired-access", "auth.json")
    client = LLMClient(
        provider,
        provider.api_key,
        "gpt-live",
        models=ModelDirectory(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    models = await client.fetch_models()

    assert [(model.id, model.display) for model in models] == [("gpt-live", "GPT Live")]
    assert seen == ["Bearer access-2"]
    saved = read_chatgpt_profile("oaiapp_tiny")
    assert saved["access_token"] == "access-2" and saved["refresh_token"] == "refresh-2"


async def test_a_revoked_chatgpt_refresh_keeps_registration_but_stops_retrying(tmp_path):
    profile = {
        "client_id": "oaiapp_tiny",
        "ext_agent_host_id": "urn:uuid:host",
        "issuer": OPENAI_ISSUER,
        "subject": "account-1",
        "email": "person@example.com",
        "access_token": "expired-access",
        "refresh_token": "revoked-refresh",
        "id_token": "retained-id-token",
        "expires_at": 1,
        "scopes": ["chatgpt.tokens.use.direct"],
    }
    save_chatgpt_profile(profile)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_grant"})

    provider = load_catalog()["chatgpt"]
    provider.set_auth_data(profile)
    provider.set_api_key("expired-access", "auth.json")
    client = LLMClient(provider, provider.api_key, "gpt-live", transport=httpx.MockTransport(handler))

    with pytest.raises(AuthenticationError, match="/connect chatgpt"):
        await client.fetch_models()
    saved = read_chatgpt_profile("oaiapp_tiny")
    assert saved["client_id"] == "oaiapp_tiny" and saved["email"] == "person@example.com"
    assert "access_token" not in saved and "refresh_token" not in saved
    assert not provider.has_key


async def test_openrouter_code_is_exchanged_for_a_key():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/auth/keys"
        assert b'"code":"one-use"' in request.content
        assert b'"code_verifier":"secret"' in request.content
        return httpx.Response(200, json={"key": "sk-or-v1-new"})

    key = await exchange_openrouter_code("one-use", "secret", transport=httpx.MockTransport(handler))
    assert key == "sk-or-v1-new"


async def test_openrouter_exchange_explains_provider_errors():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": {"message": "code expired"}})

    with pytest.raises(AuthenticationError, match="code expired"):
        await exchange_openrouter_code("old", "secret", transport=httpx.MockTransport(handler))


async def test_browser_login_explains_when_a_callback_cannot_listen(monkeypatch):
    async def cannot_listen(*args, **kwargs):
        raise OSError("loopback blocked")

    monkeypatch.setattr("tinyharness.llm.auth.asyncio.start_server", cannot_listen)
    with pytest.raises(AuthenticationError, match="remote/headless"):
        await openrouter_browser_login()


async def test_github_copilot_device_flow_returns_the_durable_github_token():
    polls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        assert request.headers["accept"] == "application/json"
        if request.url.path == "/login/device/code":
            return httpx.Response(
                200,
                json={
                    "device_code": "device",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 1,
                    "expires_in": 900,
                },
            )
        polls += 1
        if polls == 1:
            return httpx.Response(200, json={"error": "authorization_pending"})
        return httpx.Response(200, json={"access_token": "ghu_durable"})

    transport = httpx.MockTransport(handler)
    device = await start_github_copilot_login(transport=transport)

    async def no_sleep(seconds: float) -> None:
        assert seconds == 1

    token = await finish_github_copilot_login(device, transport=transport, sleep=no_sleep)
    assert token == "ghu_durable"
    assert device.user_code == "ABCD-EFGH"


async def test_github_token_is_exchanged_for_copilot_endpoint_and_short_lived_token():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/copilot_internal/v2/token"
        assert request.headers["authorization"] == "Bearer ghu_durable"
        return httpx.Response(
            200,
            json={
                "token": "tid=1;proxy-ep=proxy.business.githubcopilot.com;other=x",
                "expires_at": 2_000_000_000,
                "sku": "copilot_for_business_seat_quota",
                "endpoints": {"api": "https://api.enterprise.githubcopilot.com/"},
            },
        )

    credential = await exchange_github_copilot_token("ghu_durable", transport=httpx.MockTransport(handler))
    assert credential.base_url == "https://api.enterprise.githubcopilot.com"
    assert credential.expires_at == 2_000_000_000
    assert credential.sku == "copilot_for_business_seat_quota"


async def test_copilot_catalog_drives_each_models_wire_format_and_endpoint(tmp_path):
    requests: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(
                200,
                json={
                    "token": "tid=1;proxy-ep=proxy.individual.githubcopilot.com;x=y",
                    "expires_at": 2_000_000_000,
                    "sku": "copilot_for_business_seat_quota",
                },
            )
        assert request.headers["authorization"].startswith("Bearer tid=1;")
        assert request.headers["copilot-integration-id"] == "vscode-chat"
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "claude-live",
                            "name": "Claude Live",
                            "model_picker_enabled": True,
                            "supported_endpoints": ["/v1/messages"],
                            "capabilities": {
                                "limits": {
                                    "max_context_window_tokens": 200_000,
                                    "max_output_tokens": 16_000,
                                },
                                "supports": {"tool_calls": True, "vision": True},
                            },
                        },
                        {
                            "id": "gpt-live",
                            "model_picker_enabled": True,
                            "supported_endpoints": ["/responses"],
                            "capabilities": {
                                "limits": {"max_prompt_tokens": 100_000},
                                "supports": {
                                    "tool_calls": True,
                                    "reasoning_effort": ["low", "high"],
                                },
                            },
                        },
                    ]
                },
            )
        assert request.url.path == "/v1/messages"
        assert request.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"
        return httpx.Response(200, json={})

    provider = load_catalog()["github_copilot"]
    client = LLMClient(
        provider,
        "ghu_durable",
        "claude-live",
        models=ModelDirectory(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    models = await client.list_models(refresh=True)
    assert [(model.id, model.api) for model in models] == [
        ("auto", "chat"),
        ("claude-live", "anthropic"),
        ("gpt-live", "responses"),
    ]
    assert models[1].context == 200_000 and models[1].images is True
    assert models[2].efforts == frozenset({"low", "high"})

    check = await client.ping("claude-live")
    assert check.ok
    assert requests[-1] == ("POST", "/v1/messages")


async def test_copilot_auto_sends_the_model_its_session_names_with_the_session_token(tmp_path):
    """Seen live: a Free account rejected a request for "auto" with no model id ("The
    requested model is not supported"). Auto is a session: GitHub names the model to send
    and a token to send with it, as VS Code's Copilot Chat does. Seen live next: the
    session route answered 404 until requests carried Copilot's API version."""
    sent: list[tuple[dict[str, object], str | None]] = []
    sessions: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(
                200,
                json={
                    "token": "tid=1;proxy-ep=proxy.individual.githubcopilot.com;x=y",
                    "expires_at": 2_000_000_000,
                    "sku": "free_limited_copilot",
                },
            )
        if request.url.path == "/models/session":
            assert request.headers["X-GitHub-Api-Version"] == "2026-08-01"
            sessions.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "available_models": ["gpt-5.4-mini-free-auto"],
                    "session_token": "auto-session",
                    "expires_at": 2_000_000_000,
                },
            )
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "gpt-5.4-mini-free-auto",
                            "name": "GPT-5.4 mini",
                            "model_picker_enabled": False,
                            "policy": {"state": "enabled"},
                            "supported_endpoints": ["/chat/completions"],
                            "capabilities": {"supports": {"tool_calls": True}},
                        }
                    ]
                },
            )
        assert request.url.path == "/chat/completions"
        assert request.headers["X-GitHub-Api-Version"] == "2026-08-01"
        payload = json.loads(request.content)
        sent.append((payload, request.headers.get("Copilot-Session-Token")))
        if payload.get("stream"):
            return httpx.Response(
                200,
                text='data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n',
            )
        return httpx.Response(200, json={})

    provider = load_catalog()["github_copilot"]
    client = LLMClient(
        provider,
        "ghu_durable",
        "auto",
        models=ModelDirectory(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    models = await client.list_models(refresh=True)

    assert [(model.id, model.display) for model in models] == [
        ("auto", "Auto · GitHub chooses the best available model")
    ]
    assert client.copilot_plan == "Copilot Free"
    assert (await client.ping("auto")).ok
    assert await client.one_shot("ping") == "ok"
    assert sessions == [{"auto_mode": {"model_hints": ["auto"]}}]  # reused until it expires
    assert [(payload["model"], token) for payload, token in sent] == [
        ("gpt-5.4-mini-free-auto", "auto-session"),
        ("gpt-5.4-mini-free-auto", "auto-session"),
    ]


async def test_copilot_auto_unavailable_says_to_choose_a_model(tmp_path):
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/copilot_internal/v2/token":
            return httpx.Response(200, json={"token": "tid=1", "expires_at": 2_000_000_000})
        if request.url.path == "/models/session":
            return httpx.Response(403, json={"error": {"message": "auto mode is not enabled"}})
        return httpx.Response(200, json={"data": []})

    client = LLMClient(
        load_catalog()["github_copilot"],
        "ghu_durable",
        "auto",
        models=ModelDirectory(tmp_path),
        transport=httpx.MockTransport(handler),
    )
    check = await client.ping("auto")
    assert not check.ok and "choose a model with /model" in check.message
