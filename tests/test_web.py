"""WebFetch, WebSearch, and the providers' own web search tools."""

import httpx
import pytest

from conftest import call, make_ctx
from tinyharness.llm.adapters import ADAPTERS
from tinyharness.llm.adapters.anthropic import WEB_SEARCH_TOOL, _AnthropicParser
from tinyharness.llm.providers import load_catalog
from tinyharness.llm.types import Request
from tinyharness.permissions import Decision, PermissionGate, PermissionMode, PermissionRules
from tinyharness.tools.base import ToolError
from tinyharness.tools.web import UNTRUSTED, WebFetchTool, WebSearchTool, _check_url, html_to_text
from tinyharness.workspace import Workspace

PAGE = """<!doctype html><html><head><title>Laptops 2026</title><script>track()</script>
<style>p{}</style></head><body><nav>menu</nav>
<h1>Best laptops</h1><p>The <b>Aero 14</b> is light.   It weighs
1.2 kg.</p><ul><li>Fast</li><li>Quiet</li></ul>
<p>See <a href="/reviews/aero">the full review</a>.</p><footer>© site</footer></body></html>"""


def test_html_to_text_keeps_the_content_and_drops_the_chrome():
    text = html_to_text(PAGE, base_url="https://example.com/best")
    assert text.startswith("# Laptops 2026")
    assert "# Best laptops" in text
    assert "The Aero 14 is light. It weighs 1.2 kg." in text
    assert "- Fast\n- Quiet" in text
    assert "the full review (https://example.com/reviews/aero)" in text
    assert "track()" not in text and "menu" not in text and "© site" not in text


def serve(routes: dict[str, httpx.Response]) -> WebFetchTool:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url)) or httpx.Response(404)

    return WebFetchTool(transport=httpx.MockTransport(handler))


def html(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html; charset=utf-8"})


async def test_webfetch_returns_the_page_as_marked_untrusted_text(tmp_path):
    tool = serve({"https://example.com/best": html(PAGE)})
    out = await call(tool, {"url": "https://example.com/best"}, make_ctx(tmp_path))
    assert out.splitlines()[0].startswith("https://example.com/best")
    assert UNTRUSTED in out
    assert "The Aero 14 is light." in out


async def test_webfetch_follows_redirects_and_reads_long_pages_in_parts(tmp_path):
    tool = serve(
        {
            "https://example.com/old": httpx.Response(301, headers={"location": "/new"}),
            "https://example.com/new": httpx.Response(
                200, text="x" * 2500, headers={"content-type": "text/plain"}
            ),
        }
    )
    first = await call(tool, {"url": "https://example.com/old", "max_chars": 1000}, make_ctx(tmp_path))
    assert first.startswith("https://example.com/new")
    assert "offset=1000" in first
    rest = await call(
        tool, {"url": "https://example.com/new", "max_chars": 2000, "offset": 1000}, make_ctx(tmp_path)
    )
    assert "showing 1,000–2,500" in rest and "offset=" not in rest


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(404), "HTTP 404"),
        (
            httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
            "application/pdf",
        ),
    ],
)
async def test_webfetch_errors_are_explained(tmp_path, response, error):
    tool = serve({"https://example.com/x": response})
    with pytest.raises(ToolError, match=error):
        await call(tool, {"url": "https://example.com/x"}, make_ctx(tmp_path))


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://10.0.0.5/admin",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data",
    ],
)
async def test_webfetch_refuses_local_and_private_addresses(url):
    with pytest.raises(ToolError):
        await _check_url(url)


def resolving(*answers: list[str]):
    """A resolver that gives each answer in turn, like DNS that changes between lookups."""
    lookups = []

    async def resolver(host, port):
        lookups.append(host)
        return answers[min(len(lookups), len(answers)) - 1]

    resolver.lookups = lookups
    return resolver


async def test_webfetch_connects_to_the_address_it_checked(tmp_path, monkeypatch):
    """DNS rebinding: a second lookup could point elsewhere, so there isn't one."""
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers["host"], request.extensions.get("sni_hostname")))
        return html("<p>hi</p>")

    resolver = resolving(["93.184.216.34"], ["127.0.0.1"])
    tool = WebFetchTool(transport=httpx.MockTransport(handler), resolver=resolver)
    out = await call(tool, {"url": "https://example.com/page"}, make_ctx(tmp_path))
    assert seen == [("93.184.216.34", "example.com", "example.com")]  # TLS still checks example.com
    assert resolver.lookups == ["example.com"]  # the rebinding answer is never asked for
    assert out.startswith("https://example.com/page")  # shown and linked by name, not address


async def test_webfetch_refuses_a_name_that_resolves_to_a_private_address(tmp_path):
    tool = WebFetchTool(
        transport=httpx.MockTransport(lambda r: html("secret")), resolver=resolving(["10.0.0.5"])
    )
    with pytest.raises(ToolError, match="private network"):
        await call(tool, {"url": "https://intranet.example.com/"}, make_ctx(tmp_path))


async def test_webfetch_tries_the_next_address_when_one_is_down(tmp_path, monkeypatch):
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("https_proxy", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "2606:2800:220:1:248:1893:25c8:1946":
            raise httpx.ConnectError("unreachable", request=request)
        return html("<p>hi</p>")

    resolver = resolving(["2606:2800:220:1:248:1893:25c8:1946", "93.184.216.34"])
    tool = WebFetchTool(transport=httpx.MockTransport(handler), resolver=resolver)
    assert "hi" in await call(tool, {"url": "https://example.com:8443/"}, make_ctx(tmp_path))


async def test_webfetch_does_not_follow_a_redirect_to_another_site(tmp_path):
    """Permission is per site: the other site gets asked about when the model fetches it."""
    tool = serve(
        {
            "https://allowed.example/go": httpx.Response(
                302, headers={"location": "https://denied.example/x"}
            ),
            "https://denied.example/x": html("<p>should not be read</p>"),
        }
    )
    out = await call(tool, {"url": "https://allowed.example/go"}, make_ctx(tmp_path))
    assert "redirects to https://denied.example/x" in out and "call WebFetch with that URL" in out
    assert "should not be read" not in out


# ---- permissions: WebFetch asks per site ----


async def test_webfetch_asks_once_per_site(tmp_path):
    asked = []

    async def callback(request):
        asked.append(request.remember_label)
        return Decision(True, source="prompt", remember=True)

    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.DEFAULT, callback=callback)
    tool = WebFetchTool()

    async def check(url):
        args = tool.parse({"url": url})
        return await gate.check(tool, {"url": url}, None, target_host=tool.target_host(args))

    assert (await check("https://docs.python.org/3/")).allow
    assert asked == ["allow WebFetch for docs.python.org this session"]
    assert (await check("https://docs.python.org/3/library/")).allow
    assert len(asked) == 1  # remembered for that site
    await check("https://evil.example/")
    assert len(asked) == 2  # another site asks again


async def test_site_rules_from_the_config(tmp_path):
    rules = PermissionRules.parse(
        ["WebFetch(domain:*.github.com)"], ["WebFetch(domain:evil.example)"], tmp_path
    )
    gate = PermissionGate(workspace=Workspace(tmp_path), rules=rules)
    tool = WebFetchTool()
    assert (await gate.check(tool, {}, None, target_host="api.github.com")).allow
    assert not (await gate.check(tool, {}, None, target_host="evil.example")).allow


# ---- WebSearch: the provider's own search ----


class FakeLLM:
    def __init__(self, provider: str) -> None:
        self.provider = load_catalog()[provider]
        self.calls = []

    @property
    def supports_web_search(self):
        return bool(self.provider.web_search)

    async def one_shot_small(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "Aero 14 is best [review](https://example.com/aero)"


async def test_websearch_asks_the_model_to_search(tmp_path):
    llm = FakeLLM("anthropic")
    out = await call(WebSearchTool(lambda: llm), {"query": "best laptop 2026"}, make_ctx(tmp_path))
    assert llm.calls[0][1] == {"web_search": True}
    assert "best laptop 2026" in llm.calls[0][0]
    assert UNTRUSTED in out and "https://example.com/aero" in out


async def test_websearch_explains_when_the_provider_has_no_search(tmp_path):
    with pytest.raises(ToolError, match="isn't available with DeepSeek"):
        await call(WebSearchTool(lambda: FakeLLM("deepseek")), {"query": "anything"}, make_ctx(tmp_path))


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        ("anthropic", lambda p: {"type": WEB_SEARCH_TOOL, "name": "web_search", "max_uses": 5} in p["tools"]),
        ("openai", lambda p: {"type": "web_search"} in p["tools"]),
        ("xai", lambda p: {"type": "web_search"} in p["tools"]),
        ("gemini", lambda p: {"type": "google_search"} in p["tools"]),
        ("openrouter", lambda p: p["plugins"] == [{"id": "web"}]),
    ],
)
def test_each_provider_switches_on_its_own_search(provider, expected):
    info = load_catalog()[provider]
    payload = ADAPTERS[info.api](info).payload(Request(model="m", system="", messages=[], web_search=True))
    assert expected(payload)


def test_a_chat_provider_without_search_says_so():
    info = load_catalog()["mistral"]
    with pytest.raises(NotImplementedError):
        ADAPTERS[info.api](info).payload(Request(model="m", system="", messages=[], web_search=True))


def test_anthropic_server_search_is_not_mistaken_for_a_tool_call():
    parser = _AnthropicParser()
    start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "server_tool_use", "id": "s1", "name": "web_search"},
    }
    delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "input_json_delta", "partial_json": '{"query":'},
    }
    chunks = parser.feed(start) + parser.feed(delta)
    assert not any(c.tool_call_delta for c in chunks)
