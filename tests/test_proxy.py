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


async def test_an_allowed_host_gets_a_tunnel_and_others_a_403(monkeypatch):
    # The echo server is on loopback, which the proxy otherwise refuses.
    monkeypatch.setattr("tinyharness.sandbox.proxy._local", lambda address: False)
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
    # Seen live: curl http://[::]:8765 went to the proxy, which refused it.
    assert {"::", "::1", "0.0.0.0"} <= set(env["NO_PROXY"].split(","))


def test_a_blocked_host_is_named_in_the_commands_output():
    tool = BashTool(object(), SandboxPolicy(write_roots=()), proxy=AllowlistProxy())
    out = tool._report(
        "curl: (56) CONNECT tunnel failed", 56, False, BashInput(command="curl x"), True, ["example.com"]
    )
    assert "network_destination_denied host=example.com" in out and out.endswith("(exit code 56)")
    assert "PyPI, npm), GitHub and common CDNs" in tool.description


async def test_an_allowed_name_that_points_at_this_machine_is_refused():
    """Whoever runs an allowed name's DNS decides where it points. Anthropic's Sandbox
    Runtime refuses local answers, including the cloud metadata service."""
    for answer in ("127.0.0.1", "169.254.169.254", "::ffff:127.0.0.1"):

        async def resolve(host, port, answer=answer):
            return [answer]

        proxy = AllowlistProxy(("rebind.example",), resolve=resolve)
        try:
            proxy_port = await proxy.start()
            reader, writer = await _connect(proxy_port, "rebind.example:443")
            reply = await reader.read()
            assert reply.startswith(b"HTTP/1.1 403") and b"resolves to a local address" in reply
            assert proxy.denied == ["rebind.example"] and proxy.reached == []
            writer.close()
        finally:
            proxy.close()


async def test_an_allowed_literal_local_address_is_refused():
    """Found in review: a literal address skipped the local check, so an approved
    `network_host=169.254.169.254` reached cloud metadata. Loopback goes through
    `allow_localhost`, not the proxy."""
    for host, target in (
        ("169.254.169.254", "169.254.169.254:80"),
        ("127.0.0.1", "127.0.0.1:8000"),
        ("::1", "[::1]:8000"),
        ("0.0.0.0", "0.0.0.0:8000"),
    ):
        proxy = AllowlistProxy((host,))
        try:
            proxy_port = await proxy.start()
            reader, writer = await _connect(proxy_port, target)
            reply = await reader.read()
            assert reply.startswith(b"HTTP/1.1 403") and b"local address" in reply, host
            assert proxy.denied == [host] and proxy.reached == []
            writer.close()
        finally:
            proxy.close()


# ---- asking about a new host while its connection waits ----


async def test_a_new_host_is_asked_about_once_while_its_connections_wait(monkeypatch):
    """Seen live: refused outright, `load_dataset` retried s3.amazonaws.com for 3 minutes
    until its timeout. Now the connection waits for the decision, and a library opening
    several connections at once gets one question."""
    # The echo server is on loopback, which the proxy otherwise refuses.
    monkeypatch.setattr("tinyharness.sandbox.proxy._local", lambda address: False)
    server, port = await _echo_server()
    asked, answer = [], asyncio.Event()

    async def ask(host):
        asked.append(host)
        await answer.wait()
        return True

    proxy = AllowlistProxy((), ask=ask)
    try:
        proxy_port = await proxy.start()
        connections = [await _connect(proxy_port, f"127.0.0.1:{port}") for _ in range(3)]
        await asyncio.sleep(0.1)
        answer.set()
        for reader, writer in connections:
            assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
            writer.close()
        assert asked == ["127.0.0.1"] and proxy.denied == []
    finally:
        proxy.close()
        server.close()


async def test_a_refused_host_stays_refused_without_asking_again():
    asked = []

    async def ask(host):
        asked.append(host)
        return False

    proxy = AllowlistProxy((), ask=ask)
    try:
        proxy_port = await proxy.start()
        for _ in range(2):
            reader, writer = await _connect(proxy_port, "data.example:443")
            assert (await reader.read()).startswith(b"HTTP/1.1 403")
            writer.close()
        assert asked == ["data.example"] and proxy.denied == ["data.example", "data.example"]
    finally:
        proxy.close()


async def test_a_yes_still_cannot_reach_a_name_that_points_at_this_machine():
    async def resolve(host, port):
        return ["127.0.0.1"]

    async def ask(host):
        return True

    proxy = AllowlistProxy((), resolve=resolve, ask=ask)
    try:
        proxy_port = await proxy.start()
        reader, writer = await _connect(proxy_port, "rebind.example:443")
        reply = await reader.read()
        assert reply.startswith(b"HTTP/1.1 403") and b"resolves to a local address" in reply
        writer.close()
    finally:
        proxy.close()


async def test_time_spent_deciding_is_counted():
    async def ask(host):
        await asyncio.sleep(0.2)
        return False

    proxy = AllowlistProxy((), ask=ask)
    assert not await proxy._granted("slow.example")
    assert proxy.asking_seconds >= 0.15  # Windows' clock can measure a 0.2s sleep a little short
