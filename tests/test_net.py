"""Network failures read as what happened, where, and a code: never a bare `ConnectError`."""

import errno
import socket
import ssl

import httpx
import pytest
from rich.console import Console

from conftest import call, make_ctx
from test_adapters import CONVERSATION, client, run
from tinyharness.cli.repl import report_error
from tinyharness.cli.theme import THEME
from tinyharness.llm.client import ProviderConnectionError
from tinyharness.net import describe
from tinyharness.tools.base import ToolError
from tinyharness.tools.web import WebFetchTool


def failing(make_error):
    """A transport whose every request fails the way `make_error(request)` says."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise make_error(request)

    return httpx.MockTransport(handler)


def fail(kind, message="", cause=None):
    def make(request):
        error = kind(message, request=request)
        error.__cause__ = cause
        return error

    return make


def failure(make_error, url="https://api.example.com/v1", timeout=30.0):
    request = httpx.Request("GET", url, extensions={"timeout": {"connect": timeout, "read": timeout}})
    return describe(make_error(request))


@pytest.mark.parametrize(
    ("error", "message", "code"),
    [
        (  # httpx's own message is empty here, as in the report that started this
            fail(httpx.ConnectError, cause=ConnectionResetError(errno.ECONNRESET, "reset")),
            "can't connect to api.example.com: connection reset by peer (ECONNRESET)",
            "ECONNRESET",
        ),
        (
            fail(httpx.ConnectError, "All connection attempts failed", OSError(errno.ECONNREFUSED, "x")),
            "can't connect to api.example.com: connection refused (ECONNREFUSED)",
            "ECONNREFUSED",
        ),
        (
            fail(httpx.ConnectError, cause=socket.gaierror(socket.EAI_NONAME, "not known")),
            "can't connect to api.example.com: DNS lookup failed (EAI_NONAME)",
            "EAI_NONAME",
        ),
        (
            fail(httpx.ConnectTimeout),
            "can't connect to api.example.com: timed out after 30s (ConnectTimeout)",
            "ConnectTimeout",
        ),
        (
            fail(httpx.RemoteProtocolError, "peer closed connection without sending complete message body"),
            "the connection to api.example.com dropped mid-response: peer closed connection without "
            "sending complete message body (RemoteProtocolError)",
            "RemoteProtocolError",
        ),
        (
            fail(httpx.ConnectError),  # nothing underneath at all
            "can't connect to api.example.com: no details from the network library (ConnectError)",
            "ConnectError",
        ),
    ],
)
def test_failures_say_what_where_and_a_code(error, message, code):
    described = failure(error)
    assert (described.message, described.code) == (message, code)


def test_tls_failures_give_the_reason():
    cause = ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
    cause.reason, cause.verify_message = "CERTIFICATE_VERIFY_FAILED", "Hostname mismatch."
    assert failure(fail(httpx.ConnectError, cause=cause)).message == (
        "can't connect to api.example.com: TLS certificate check failed: Hostname mismatch "
        "(CERTIFICATE_VERIFY_FAILED)"
    )


def test_a_slow_reply_is_told_apart_from_a_network_problem():
    slow = failure(fail(httpx.ReadTimeout), timeout=120)
    assert (
        slow.slow_reply
        and slow.message == "api.example.com stopped responding: timed out after 120s (ReadTimeout)"
    )
    assert not failure(fail(httpx.ConnectTimeout)).slow_reply


def test_a_pinned_connection_still_names_the_site():
    """WebFetch connects to an IP but sends the site's name in Host; errors use the name."""

    def make(request):
        return httpx.ConnectError("", request=request)

    request = httpx.Request("GET", "https://93.184.216.34/", headers={"Host": "example.com"})
    assert describe(make(request)).host == "example.com"


# ---- where they surface ----


async def test_the_llm_client_raises_a_provider_connection_error():
    reset = ConnectionResetError(errno.ECONNRESET, "reset")
    llm = client("openrouter", None)
    llm.transport = failing(fail(httpx.ConnectError, cause=reset))
    with pytest.raises(ProviderConnectionError) as caught:
        await run(llm, CONVERSATION)
    assert caught.value.code == "ECONNRESET" and caught.value.host == "openrouter.ai"
    assert str(caught.value) == "can't connect to openrouter.ai: connection reset by peer (ECONNRESET)"


def test_the_cli_shows_the_reason_and_what_to_do():
    console = Console(record=True, width=120, theme=THEME)
    error = ProviderConnectionError(
        "can't connect to openrouter.ai: connection reset by peer (ECONNRESET)",
        host="openrouter.ai",
        code="ECONNRESET",
    )
    report_error(console, error)
    shown = console.export_text()
    assert "connection reset by peer (ECONNRESET)" in shown
    assert "Check your network, VPN or proxy" in shown and "/ping" in shown


def test_a_slow_model_keeps_its_own_advice():
    console = Console(record=True, width=120, theme=THEME)
    error = ProviderConnectionError("x", host="h", code="ReadTimeout", slow_reply=True)
    report_error(console, error)
    assert "request_timeout" in console.export_text()


async def test_webfetch_explains_a_network_failure(tmp_path):
    tool = WebFetchTool(transport=failing(fail(httpx.ConnectError, cause=OSError(errno.ECONNREFUSED, "x"))))
    with pytest.raises(
        ToolError, match=r"^can't connect to example\.com: connection refused \(ECONNREFUSED\)$"
    ):
        await call(tool, {"url": "https://example.com/"}, make_ctx(tmp_path))
