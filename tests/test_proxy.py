"""The sandbox's allowlist proxy: sandboxed commands reach listed hosts through it and
nothing else (sandbox/proxy.py)."""

import asyncio

from tinyharness.sandbox import SandboxPolicy
from tinyharness.sandbox.proxy import DEFAULT_ALLOWED_HOSTS, AllowlistProxy, host_allowed
from tinyharness.sandbox.seatbelt import profile
from tinyharness.tools.bash import BashInput, BashTool


def test_hosts_match_exactly_or_by_subdomain_wildcard():
    allowed = ("pypi.org", "*.github.com")
    assert host_allowed("pypi.org", allowed) and host_allowed("PyPI.org.", allowed)
    assert host_allowed("api.github.com", allowed) and not host_allowed("github.com", allowed)
    assert not host_allowed("evilpypi.org", allowed) and not host_allowed("pypi.org.evil.com", allowed)
    assert {"pypi.org", "registry.npmjs.org", "cdn.jsdelivr.net"} <= set(DEFAULT_ALLOWED_HOSTS)


async def _echo_server():
    async def echo(reader, writer):
        writer.write(await reader.read(100))
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _connect(proxy_port, target):
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
    await writer.drain()
    return reader, writer


async def test_an_allowed_host_gets_a_tunnel_and_others_a_403():
    server, port = await _echo_server()
    proxy = AllowlistProxy(("127.0.0.1",))
    try:
        proxy_port = await proxy.start()
        reader, writer = await _connect(proxy_port, f"127.0.0.1:{port}")
        assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
        writer.write(b"ping")
        await writer.drain()
        assert await reader.read(100) == b"ping"
        writer.close()

        reader, writer = await _connect(proxy_port, "example.com:443")
        reply = await reader.read()
        assert reply.startswith(b"HTTP/1.1 403") and b"example.com is not in the allowed hosts" in reply
        assert proxy.denied == ["example.com"]
    finally:
        proxy.close()
        server.close()


def test_the_seatbelt_profile_lets_commands_reach_only_the_proxy():
    rules = profile(SandboxPolicy(write_roots=(), proxy_port=40123))
    assert '(allow network-outbound (remote ip "localhost:40123"))' in rules
    assert "(deny network*)" in rules and "localhost:*" not in rules


def test_proxy_settings_cover_the_common_tools():
    proxy = AllowlistProxy()
    proxy.port = 5555
    env = proxy.env()
    assert (
        env["HTTPS_PROXY"] == env["https_proxy"] == env["npm_config_https_proxy"] == "http://127.0.0.1:5555"
    )
    assert "127.0.0.1" in env["NO_PROXY"]


def test_a_blocked_host_is_named_in_the_commands_output():
    tool = BashTool(object(), SandboxPolicy(write_roots=()), proxy=AllowlistProxy())
    out = tool._report(
        "curl: (56) CONNECT tunnel failed", 56, False, BashInput(command="curl x"), True, ["example.com"]
    )
    assert "the sandbox blocked example.com" in out and out.endswith("(exit code 56)")
    assert "PyPI, npm), GitHub and common CDNs" in tool.description
