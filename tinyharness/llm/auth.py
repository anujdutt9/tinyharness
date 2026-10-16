"""Interactive authentication flows offered by model providers."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import os
import secrets
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import jwt

OPENROUTER_AUTHORIZE_URL = "https://openrouter.ai/auth"
OPENROUTER_EXCHANGE_URL = "https://openrouter.ai/api/v1/auth/keys"
CALLBACK_TIMEOUT = 300.0
GITHUB_CLIENT_ID = base64.b64decode("SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode()
GITHUB_DEVICE_URL = "https://github.com/login/device/code"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
COPILOT_HEADERS = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
# Copilot's own API (chat, models, Auto sessions) is versioned separately from api.github.com,
# so this goes only to the Copilot endpoint. It matches VS Code's Copilot Chat; without it,
# the Auto session route answered 404.
COPILOT_API_VERSION = "2026-08-01"
OPENAI_ISSUER = "https://auth.openai.com"
OPENAI_AUTHORIZE_URL = f"{OPENAI_ISSUER}/api/accounts/authorize"
OPENAI_TOKEN_URL = f"{OPENAI_ISSUER}/api/accounts/oauth/token"
OPENAI_DISCOVERY_URL = f"{OPENAI_ISSUER}/.well-known/openid-configuration"
OPENAI_RESOURCE = "https://api.openai.com/v1"
OPENAI_DYNAMIC_CLIENT = "dynamic_agent_client"
OPENAI_SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
OPENAI_PLAN_SCOPE = "chatgpt.tokens.use.direct"
_PUBLIC_JWT_ALGORITHMS = frozenset(
    {"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512", "EdDSA"}
)
_TERMINAL_REFRESH_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
)


class AuthenticationError(RuntimeError):
    """The provider could not complete an interactive connection."""


class ChatGPTReauthorizationRequired(AuthenticationError):
    """The saved ChatGPT registration remains valid but its renewable session does not."""


@dataclass(frozen=True)
class Authorization:
    """A PKCE authorization request waiting for a provider-issued code."""

    url: str
    verifier: str
    state: str


@dataclass(frozen=True)
class DeviceAuthorization:
    """A GitHub device login that is waiting for the user in a browser."""

    device_code: str
    user_code: str
    verification_url: str
    interval: float
    expires_in: float


@dataclass(frozen=True)
class CopilotCredential:
    """The short-lived token and endpoint derived from a GitHub OAuth token."""

    token: str
    expires_at: float
    base_url: str
    sku: str = ""


async def chatgpt_browser_login(
    host_id: str,
    *,
    existing: dict[str, Any] | None = None,
    on_registration: Callable[[str], None] | None = None,
    opener: Callable[[str], bool] = webbrowser.open,
    timeout: float = CALLBACK_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Register or reauthorize one ChatGPT plan profile through a loopback callback."""
    result: asyncio.Future[dict[str, str]] = asyncio.get_running_loop().create_future()
    state = secrets.token_urlsafe(24)
    nonce = secrets.token_urlsafe(24)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")

    async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = (await reader.readline()).decode("ascii", "replace")
            target = line.split(" ", 2)[1] if line.startswith("GET ") else ""
            answer = _chatgpt_callback(target, state)
            if not result.done():
                result.set_result(answer)
            body = "tinyharness is connected. You can close this page and return to your terminal."
            status = "200 OK"
        except (AuthenticationError, IndexError) as error:
            if not result.done():
                result.set_exception(error)
            body = f"tinyharness could not connect: {escape(str(error))}"
            status = "400 Bad Request"
        payload = body.encode()
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    try:
        server = await asyncio.start_server(callback, "127.0.0.1", 0)
    except OSError as error:
        raise AuthenticationError("couldn't open the local ChatGPT sign-in callback") from error
    port = server.sockets[0].getsockname()[1]
    redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
    saved_client = str((existing or {}).get("client_id") or "")
    requested_client = saved_client or OPENAI_DYNAMIC_CLIENT
    query = {
        "client_id": requested_client,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": OPENAI_SCOPES,
        "resource": OPENAI_RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if saved_client:
        if id_token := (existing or {}).get("id_token"):
            query["id_token_hint"] = str(id_token)
        if email := (existing or {}).get("email"):
            query["login_hint"] = str(email)
    else:
        query["agent_name_hint"] = "tinyharness"
    authorization_url = f"{OPENAI_AUTHORIZE_URL}?{urlencode(query)}"
    try:
        try:
            opened = await asyncio.to_thread(opener, authorization_url)
        except OSError as error:
            raise AuthenticationError("couldn't open a browser for ChatGPT sign-in") from error
        if not opened:
            raise AuthenticationError("couldn't open a browser for ChatGPT sign-in")
        async with server:
            answer = await asyncio.wait_for(result, timeout)
    except TimeoutError as error:
        raise AuthenticationError("ChatGPT sign-in timed out; try connecting again") from error
    finally:
        server.close()
        await server.wait_closed()

    returned_client = answer.get("client_id", "")
    if saved_client:
        if returned_client and not secrets.compare_digest(returned_client, saved_client):
            raise AuthenticationError("ChatGPT sign-in returned a different client registration")
        client_id = saved_client
    else:
        if not returned_client or returned_client == OPENAI_DYNAMIC_CLIENT:
            raise AuthenticationError("ChatGPT did not finish registering tinyharness")
        client_id = returned_client
        if on_registration is not None:
            on_registration(client_id)

    token = await _post_chatgpt_form(
        {
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": answer["code"],
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
            "resource": OPENAI_RESOURCE,
        },
        transport=transport,
    )
    identity = await _validate_openai_id_token(
        _required_string(token, "id_token"), client_id, nonce=nonce, transport=transport
    )
    if (
        existing
        and existing.get("subject")
        and (identity["sub"] != existing.get("subject") or identity["iss"] != existing.get("issuer"))
    ):
        raise AuthenticationError("ChatGPT sign-in returned a different account")
    scopes = _scopes(token.get("scope"))
    if OPENAI_PLAN_SCOPE not in scopes:
        raise AuthenticationError("ChatGPT plan usage was not enabled for tinyharness")
    now = time.time()
    expires_in = _positive_number(token.get("expires_in"), "expires_in")
    return {
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "issuer": identity["iss"],
        "subject": identity["sub"],
        "email": str(identity.get("email") or ""),
        "id_token": _required_string(token, "id_token"),
        "access_token": _required_string(token, "access_token"),
        "refresh_token": _required_string(token, "refresh_token"),
        "token_type": str(token.get("token_type") or "Bearer"),
        "expires_in": expires_in,
        "expires_at": now + expires_in,
        "saved_at": datetime.now(UTC).isoformat(),
        "scopes": sorted(scopes),
    }


async def refresh_chatgpt_token(
    profile: dict[str, Any], *, transport: httpx.AsyncBaseTransport | None = None
) -> dict[str, Any]:
    """Rotate one ChatGPT profile's access and refresh tokens."""
    client_id = _required_string(profile, "client_id")
    token = await _post_chatgpt_form(
        {
            "grant_type": "refresh_token",
            "client_id": client_id,
            "refresh_token": _required_string(profile, "refresh_token"),
            "resource": OPENAI_RESOURCE,
        },
        transport=transport,
    )
    updated = dict(profile)
    if id_token := token.get("id_token"):
        if not isinstance(id_token, str):
            raise AuthenticationError("OpenAI returned an invalid ID token")
        identity = await _validate_openai_id_token(id_token, client_id, transport=transport)
        if identity["sub"] != profile.get("subject") or identity["iss"] != profile.get("issuer"):
            raise AuthenticationError("OpenAI refreshed a different ChatGPT account")
        updated["id_token"] = id_token
    scopes = _scopes(token.get("scope")) or set(map(str, profile.get("scopes") or []))
    if OPENAI_PLAN_SCOPE not in scopes:
        raise ChatGPTReauthorizationRequired(
            "ChatGPT plan usage is no longer enabled; run /connect chatgpt to authorize it again"
        )
    now = time.time()
    expires_in = _positive_number(token.get("expires_in"), "expires_in")
    updated.update(
        access_token=_required_string(token, "access_token"),
        refresh_token=_required_string(token, "refresh_token"),
        token_type=str(token.get("token_type") or profile.get("token_type") or "Bearer"),
        expires_in=expires_in,
        expires_at=now + expires_in,
        saved_at=datetime.now(UTC).isoformat(),
        scopes=sorted(scopes),
    )
    return updated


@contextlib.asynccontextmanager
async def chatgpt_refresh_lock(timeout: float = 60.0) -> Any:
    """Serialize rotating refresh-token use across tinyharness processes."""
    from tinyharness.paths import auth_path

    lock = auth_path().with_name(".chatgpt-refresh.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd: int | None = None
    while fd is None:
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.write(fd, str(os.getpid()).encode())
        except FileExistsError:
            with contextlib.suppress(OSError):
                if time.time() - lock.stat().st_mtime > 120:
                    lock.unlink()
                    continue
            if time.monotonic() >= deadline:
                raise AuthenticationError("timed out waiting to refresh the ChatGPT connection") from None
            await asyncio.sleep(0.1)
    try:
        yield
    finally:
        os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def _chatgpt_callback(target: str, expected_state: str) -> dict[str, str]:
    """Validate the ChatGPT loopback result before any code exchange."""
    parsed = urlsplit(target)
    if parsed.path != "/auth/callback":
        raise AuthenticationError("ChatGPT sign-in returned to an invalid callback path")
    query = parse_qs(parsed.query)
    if not secrets.compare_digest(query.get("state", [""])[0], expected_state):
        raise AuthenticationError("ChatGPT sign-in returned an invalid state")
    if error := query.get("error", [""])[0]:
        detail = query.get("error_description", [error])[0]
        raise AuthenticationError(f"ChatGPT sign-in was declined ({detail})")
    code = query.get("code", [""])[0]
    if not code:
        raise AuthenticationError("ChatGPT sign-in returned no authorization code")
    return {"code": code, "client_id": query.get("client_id", [""])[0]}


async def _post_chatgpt_form(
    data: dict[str, str], *, transport: httpx.AsyncBaseTransport | None
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            response = await client.post(OPENAI_TOKEN_URL, data=data)
    except httpx.TransportError as error:
        raise AuthenticationError(f"couldn't reach OpenAI: {error}") from error
    if response.status_code != 200:
        if (
            data.get("grant_type") == "refresh_token"
            and _oauth_error_code(response) in _TERMINAL_REFRESH_ERRORS
        ):
            raise ChatGPTReauthorizationRequired(
                "the ChatGPT session can no longer be refreshed; run /connect chatgpt to sign in again"
            )
        raise AuthenticationError(f"OpenAI rejected the ChatGPT connection ({_error_detail(response)})")
    try:
        body = response.json()
    except ValueError as error:
        raise AuthenticationError("OpenAI returned an invalid token response") from error
    if not isinstance(body, dict):
        raise AuthenticationError("OpenAI returned an invalid token response")
    return body


async def _validate_openai_id_token(
    token: str,
    client_id: str,
    *,
    nonce: str | None = None,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            discovery_response = await client.get(OPENAI_DISCOVERY_URL)
            discovery_response.raise_for_status()
            discovery = discovery_response.json()
            if discovery.get("issuer") != OPENAI_ISSUER:
                raise AuthenticationError("OpenAI returned invalid identity metadata")
            jwks_uri = discovery["jwks_uri"]
            algorithms = [
                str(item)
                for item in discovery.get("id_token_signing_alg_values_supported") or []
                if item in _PUBLIC_JWT_ALGORITHMS
            ]
            if not isinstance(jwks_uri, str) or urlsplit(jwks_uri).scheme != "https" or not algorithms:
                raise AuthenticationError("OpenAI returned unsafe identity metadata")
            keys_response = await client.get(jwks_uri)
            keys_response.raise_for_status()
            key_set = jwt.PyJWKSet.from_dict(keys_response.json())
        header = jwt.get_unverified_header(token)
        key = next((item for item in key_set.keys if item.key_id == header.get("kid")), None)
        if key is None:
            raise AuthenticationError("OpenAI's signing key was not found")
        required = ["iss", "aud", "exp", "sub"]
        if nonce is not None:
            required.append("nonce")
        identity = jwt.decode(
            token,
            key.key,
            algorithms=algorithms,
            audience=client_id,
            issuer=OPENAI_ISSUER,
            options={"require": required},
        )
    except AuthenticationError:
        raise
    except (httpx.HTTPError, jwt.PyJWTError, KeyError, TypeError, ValueError) as error:
        raise AuthenticationError(f"couldn't validate the OpenAI identity token ({error})") from error
    if nonce is not None and not secrets.compare_digest(str(identity.get("nonce") or ""), nonce):
        raise AuthenticationError("OpenAI identity token returned an invalid nonce")
    return identity


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise AuthenticationError(f"OpenAI returned no {field.replace('_', ' ')}")
    return value


def _positive_number(value: object, field: str) -> float:
    if not isinstance(value, int | float) or value <= 0:
        raise AuthenticationError(f"OpenAI returned an invalid {field.replace('_', ' ')}")
    return float(value)


def _scopes(value: object) -> set[str]:
    if isinstance(value, str):
        return set(value.split())
    if isinstance(value, list):
        return {str(item) for item in value}
    return set()


def _oauth_error_code(response: httpx.Response) -> str:
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return ""
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        return str(error.get("code") or error.get("type") or "")
    return ""


async def start_github_copilot_login(
    *, transport: httpx.AsyncBaseTransport | None = None
) -> DeviceAuthorization:
    """Start the device flow used by GitHub Copilot clients."""
    response = await _post_github_form(
        GITHUB_DEVICE_URL,
        {"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
        transport=transport,
    )
    device_code = response.get("device_code")
    user_code = response.get("user_code")
    verification_url = response.get("verification_uri")
    interval = response.get("interval", 5)
    expires_in = response.get("expires_in")
    if not (
        isinstance(device_code, str)
        and isinstance(user_code, str)
        and isinstance(verification_url, str)
        and isinstance(interval, int | float)
        and isinstance(expires_in, int | float)
    ):
        raise AuthenticationError("GitHub returned an invalid device authorization")
    device = DeviceAuthorization(
        device_code=device_code,
        user_code=user_code,
        verification_url=verification_url,
        interval=max(1.0, float(interval)),
        expires_in=float(expires_in),
    )
    parsed = urlsplit(device.verification_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AuthenticationError("GitHub returned an unsafe verification URL")
    return device


async def finish_github_copilot_login(
    device: DeviceAuthorization,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> str:
    """Poll GitHub until the device flow returns the user's OAuth token."""
    deadline = time.monotonic() + device.expires_in
    delay = device.interval
    while time.monotonic() < deadline:
        await sleep(delay)
        response = await _post_github_form(
            GITHUB_TOKEN_URL,
            {
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            transport=transport,
        )
        token = response.get("access_token")
        if isinstance(token, str) and token.strip():
            return token.strip()
        error = response.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval = response.get("interval")
            delay = max(delay + 5, float(interval) if isinstance(interval, int | float) else 0)
            continue
        detail = response.get("error_description") or error or "invalid response"
        raise AuthenticationError(f"GitHub sign-in failed ({detail})")
    raise AuthenticationError("GitHub sign-in timed out; try connecting again")


async def github_copilot_login(
    *,
    notify: Callable[[DeviceAuthorization], None],
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Run GitHub's device flow and return the durable GitHub OAuth token."""
    device = await start_github_copilot_login(transport=transport)
    notify(device)
    return await finish_github_copilot_login(device, transport=transport)


async def exchange_github_copilot_token(
    github_token: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> CopilotCredential:
    """Exchange a GitHub OAuth token for Copilot's short-lived API credential."""
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            response = await client.get(
                GITHUB_COPILOT_TOKEN_URL,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {github_token}",
                    **COPILOT_HEADERS,
                },
            )
    except httpx.TransportError as error:
        raise AuthenticationError(f"couldn't reach GitHub Copilot: {error}") from error
    if response.status_code != 200:
        raise AuthenticationError(f"GitHub Copilot rejected the connection ({_error_detail(response)})")
    try:
        body = response.json()
        token = body["token"]
        expires_at = float(body["expires_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuthenticationError("GitHub Copilot returned an invalid access token") from error
    if not isinstance(token, str) or not token:
        raise AuthenticationError("GitHub Copilot returned no access token")
    sku = body.get("sku")
    return CopilotCredential(
        token,
        expires_at,
        _copilot_base_url(token, body),
        sku.strip() if isinstance(sku, str) else "",
    )


@dataclass(frozen=True)
class CopilotAutoSession:
    """The models GitHub's Auto route offers right now, and the token to send with them."""

    models: tuple[str, ...]
    token: str
    expires_at: float


async def open_github_copilot_auto_session(
    base_url: str, copilot_token: str, *, transport: httpx.AsyncBaseTransport | None = None
) -> CopilotAutoSession:
    """Ask Copilot's Auto route which models to use.

    Auto is not a model id: VS Code's Copilot Chat posts this request, sends the first
    model it names, and adds the session token as ``Copilot-Session-Token``.
    """
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/models/session",
                json={"auto_mode": {"model_hints": ["auto"]}},
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {copilot_token}",
                    **COPILOT_HEADERS,
                    "X-GitHub-Api-Version": COPILOT_API_VERSION,
                },
            )
    except httpx.TransportError as error:
        raise AuthenticationError(f"couldn't reach GitHub Copilot: {error}") from error
    if response.status_code != 200:
        raise AuthenticationError(
            f"GitHub Copilot Auto is not available ({_error_detail(response)}); choose a model with /model"
        )
    try:
        body = response.json()
        models = tuple(str(model) for model in body["available_models"])
        token = body["session_token"]
        expires_at = float(body["expires_at"])
    except (KeyError, TypeError, ValueError) as error:
        raise AuthenticationError("GitHub Copilot returned an invalid Auto session") from error
    if not models or not isinstance(token, str) or not token:
        raise AuthenticationError("GitHub Copilot Auto offered no models; choose a model with /model")
    return CopilotAutoSession(models, token, expires_at)


def _copilot_base_url(token: str, response: dict[str, object] | None = None) -> str:
    """Use the account endpoint GitHub returned, with the older token field as fallback."""
    endpoints = response.get("endpoints") if response else None
    api = endpoints.get("api") if isinstance(endpoints, dict) else None
    if isinstance(api, str):
        parsed = urlsplit(api)
        if parsed.scheme == "https" and parsed.netloc and parsed.username is None:
            return api.rstrip("/")
    for part in token.split(";"):
        if part.startswith("proxy-ep="):
            host = part.partition("=")[2]
            if host and "/" not in host:
                return f"https://{host.replace('proxy.', 'api.', 1)}"
    return "https://api.individual.githubcopilot.com"


async def _post_github_form(
    url: str,
    data: dict[str, str],
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> dict[str, object]:
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            response = await client.post(
                url,
                data=data,
                headers={"Accept": "application/json", "User-Agent": COPILOT_HEADERS["User-Agent"]},
            )
    except httpx.TransportError as error:
        raise AuthenticationError(f"couldn't reach GitHub: {error}") from error
    if response.status_code != 200:
        raise AuthenticationError(f"GitHub sign-in failed ({_error_detail(response)})")
    try:
        body = response.json()
    except ValueError as error:
        raise AuthenticationError("GitHub returned an invalid sign-in response") from error
    if not isinstance(body, dict):
        raise AuthenticationError("GitHub returned an invalid sign-in response")
    return body


def openrouter_authorization(callback_url: str | None) -> Authorization:
    """Create an OpenRouter PKCE request for a local or headless client."""
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
    state = secrets.token_urlsafe(24)
    query = {
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "key_label": "tinyharness",
    }
    if callback_url:
        query.update({"callback_url": callback_url, "state": state})
    return Authorization(f"{OPENROUTER_AUTHORIZE_URL}?{urlencode(query)}", verifier, state)


async def exchange_openrouter_code(
    code: str,
    verifier: str,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Exchange a one-use OpenRouter authorization code for the user's API key."""
    try:
        async with httpx.AsyncClient(timeout=30, transport=transport) as client:
            response = await client.post(
                OPENROUTER_EXCHANGE_URL,
                json={
                    "code": code,
                    "code_verifier": verifier,
                    "code_challenge_method": "S256",
                },
            )
    except httpx.TransportError as error:
        raise AuthenticationError(f"couldn't reach OpenRouter: {error}") from error
    if response.status_code != 200:
        detail = _error_detail(response)
        raise AuthenticationError(f"OpenRouter rejected the authorization ({detail})")
    try:
        key = response.json()["key"]
    except (KeyError, TypeError, ValueError) as error:
        raise AuthenticationError("OpenRouter returned no API key") from error
    if not isinstance(key, str) or not key.strip():
        raise AuthenticationError("OpenRouter returned no API key")
    return key.strip()


async def openrouter_browser_login(
    *,
    opener: Callable[[str], bool] = webbrowser.open,
    timeout: float = CALLBACK_TIMEOUT,
    transport: httpx.AsyncBaseTransport | None = None,
) -> str:
    """Open OpenRouter in the browser and return the key sent to a loopback callback."""
    result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    expected_state = ""

    async def callback(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = (await reader.readline()).decode("ascii", "replace")
            target = line.split(" ", 2)[1] if line.startswith("GET ") else ""
            code = _callback_code(target, expected_state)
            if not result.done():
                result.set_result(code)
            body = "tinyharness is connected. You can close this page and return to your terminal."
            status = "200 OK"
        except (AuthenticationError, IndexError) as error:
            if not result.done():
                result.set_exception(error)
            body = f"tinyharness could not connect: {escape(str(error))}"
            status = "400 Bad Request"
        payload = body.encode()
        writer.write(
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    try:
        server = await asyncio.start_server(callback, "127.0.0.1", 0)
    except OSError as error:
        raise AuthenticationError(
            "couldn't open a local sign-in callback; reconnect and choose the remote/headless option"
        ) from error
    port = server.sockets[0].getsockname()[1]
    authorization = openrouter_authorization(f"http://127.0.0.1:{port}/callback")
    expected_state = authorization.state
    try:
        try:
            opened = await asyncio.to_thread(opener, authorization.url)
        except OSError as error:
            raise AuthenticationError(
                "couldn't open the browser; reconnect and choose the remote/headless option"
            ) from error
        if not opened:
            raise AuthenticationError(
                "couldn't open the browser; reconnect and choose the remote/headless option"
            )
        async with server:
            code = await asyncio.wait_for(result, timeout)
    except TimeoutError as error:
        raise AuthenticationError("OpenRouter sign-in timed out; try connecting again") from error
    finally:
        server.close()
        await server.wait_closed()
    return await exchange_openrouter_code(code, authorization.verifier, transport=transport)


def _callback_code(target: str, expected_state: str) -> str:
    """Validate one loopback callback and return its authorization code."""
    query = parse_qs(urlsplit(target).query)
    if error := query.get("error", [""])[0]:
        raise AuthenticationError(f"OpenRouter sign-in was declined ({error})")
    if not secrets.compare_digest(query.get("state", [""])[0], expected_state):
        raise AuthenticationError("OpenRouter sign-in returned an invalid state")
    code = query.get("code", [""])[0]
    if not code:
        raise AuthenticationError("OpenRouter sign-in returned no authorization code")
    return code


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or f"HTTP {response.status_code}")
        if isinstance(error, str):
            return error
    return f"HTTP {response.status_code}"
