"""The sandbox's one way out: a small HTTP proxy on 127.0.0.1 that lets sandboxed commands
reach an allowlist of hosts (package registries, GitHub, CDNs) and nothing else.

The Seatbelt profile allows a sandboxed command to connect only to this proxy's port, and
the command's environment points HTTP(S)_PROXY at it, which curl, pip, uv, npm and git
follow. The proxy then decides by host name:

    CONNECT host:443   HTTPS: a tunnel to an allowed host (the proxy never sees the content)
    GET http://host/   plain HTTP to an allowed host, forwarded

Anything else gets a 403 that says which host was blocked and how to allow it, and the
host is recorded so Bash can tell the model. Programs that ignore proxy settings simply
have no network, as before.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import socket
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

# Reached without asking from inside the sandbox. This is a destination allowlist, not
# data-loss prevention: HTTPS tunnels are opaque, so an allowed host can also receive
# readable project data. Keep the defaults limited to dependency and rendering sources.
DEFAULT_ALLOWED_HOSTS = (
    # Python
    "pypi.org",
    "files.pythonhosted.org",
    # JavaScript
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    # Source code
    "github.com",
    "*.github.com",
    "*.githubusercontent.com",
    # Script and font CDNs used by web pages
    "cdn.jsdelivr.net",
    "cdnjs.cloudflare.com",
    "unpkg.com",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    # Playwright's browser downloads
    "cdn.playwright.dev",
    "playwright.download.prss.microsoft.com",
)

# Hosts a package manager installs from. Installing what the project declares from
# them is authorized setup, not outside content (tools/bash.py); a new package name is.
PACKAGE_REGISTRY_HOSTS = ("pypi.org", "files.pythonhosted.org", "registry.npmjs.org", "registry.yarnpkg.com")

# Services whose downloads hop across several exact hosts. Approving one of them opens the
# set, so a redirect isn't one more blocked command and one more prompt (seen live: a model
# download went from huggingface.co to cas-server.xethub.hf.co). Exact names, no wildcards.
SERVICE_HOSTS: dict[str, tuple[str, ...]] = {
    "Hugging Face": (
        "huggingface.co",
        "cdn-lfs.huggingface.co",
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
        "cas-bridge.xethub.hf.co",
        "cas-server.xethub.hf.co",
        "transfer.xethub.hf.co",
    ),
}


def service_hosts(host: str) -> tuple[str, ...]:
    """The known service set `host` belongs to, or just `host`."""
    host = host.lower().rstrip(".")
    return next((hosts for hosts in SERVICE_HOSTS.values() if host in hosts), (host,))


READ_LIMIT = 64 * 1024  # bytes of request head we'll read before giving up


def host_allowed(host: str, allowed: Iterable[str]) -> bool:
    """Exact names, or `*.example.com` for any subdomain (not example.com itself)."""
    host = host.lower().rstrip(".")
    for pattern in allowed:
        pattern = pattern.lower().strip()
        if pattern.startswith("*."):
            if host.endswith(pattern[1:]):
                return True
        elif host == pattern:
            return True
    return False


async def _resolve(host: str, port: int) -> list[str]:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


def _local(address: str) -> bool:
    """This machine, its link (the cloud metadata service at 169.254.169.254) or no one.

    Private ranges stay reachable: an allowlisted company mirror may live there.
    """
    ip = ipaddress.ip_address(address.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_link_local or ip.is_unspecified or ip.is_multicast or ip.is_reserved


@dataclass
class AllowlistProxy:
    allowed: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    denied: list[str] = field(default_factory=list)  # hosts refused, newest last
    reached: list[str] = field(default_factory=list)  # hosts connected to, newest last
    resolve: Callable[[str, int], Awaitable[list[str]]] = field(default=_resolve, repr=False)
    _server: asyncio.base_events.Server | None = None
    port: int = 0

    async def start(self) -> int:
        if self._server is None:
            self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
            self.port = self._server.sockets[0].getsockname()[1]
        return self.port

    def close(self) -> None:
        """Stop accepting connections (open tunnels end with their commands)."""
        if self._server is not None:
            self._server.close()
            self._server = None

    def env(self) -> dict[str, str]:
        """The environment that sends a command's traffic through this proxy."""
        url = f"http://127.0.0.1:{self.port}"
        return {
            "HTTP_PROXY": url,
            "HTTPS_PROXY": url,
            "http_proxy": url,
            "https_proxy": url,
            "npm_config_proxy": url,
            "npm_config_https_proxy": url,
            "NO_PROXY": "localhost,127.0.0.1",
            "no_proxy": "localhost,127.0.0.1",
        }

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
            writer.close()
            return
        if len(head) > READ_LIMIT:
            writer.close()
            return
        request_line, *header_lines = head.decode("latin-1").split("\r\n")
        try:
            method, target, version = request_line.split(" ", 2)
        except ValueError:
            writer.close()
            return
        if method.upper() == "CONNECT":
            host, _, port = target.rpartition(":")
            await self._tunnel(host.strip("[]"), int(port or 443), reader, writer)
        else:
            parts = urlsplit(target)
            if parts.scheme != "http" or not parts.hostname:
                await _reply(writer, 400, "only absolute http:// URLs and CONNECT are proxied")
                return
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            rewritten = f"{method} {path} {version}\r\n" + "\r\n".join(
                line for line in header_lines if line and not line.lower().startswith("proxy-")
            )
            await self._forward(
                parts.hostname, parts.port or 80, rewritten.encode("latin-1") + b"\r\n\r\n", reader, writer
            )

    async def _tunnel(
        self, host: str, port: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        upstream = await self._open(host, port, writer)
        if upstream is None:
            return
        writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        await writer.drain()
        await _pipe_both(reader, writer, *upstream)

    async def _forward(
        self, host: str, port: int, head: bytes, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        upstream = await self._open(host, port, writer)
        if upstream is None:
            return
        up_reader, up_writer = upstream
        up_writer.write(head)
        await up_writer.drain()
        await _pipe_both(reader, writer, up_reader, up_writer)

    async def _open(
        self, host: str, port: int, writer: asyncio.StreamWriter
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        if not host_allowed(host, self.allowed):
            self.denied.append(host)
            log.info("sandbox proxy blocked %s", host)
            await _reply(
                writer,
                403,
                f"tinyharness sandbox: {host} is not in the allowed hosts. Add the exact host "
                "to [sandbox] allow_hosts in config.toml if you trust it.",
            )
            return None
        # The allowlist matches names, and whoever runs a name's DNS decides where it points.
        # Look it up once, refuse local answers, and connect to that same address, so a
        # second lookup can't point the connection somewhere else (DNS rebinding).
        address = host
        try:
            ipaddress.ip_address(host)  # an address listed on purpose is the user's choice
        except ValueError:
            try:
                addresses = await asyncio.wait_for(self.resolve(host, port), timeout=30)
            except (OSError, TimeoutError) as e:
                await _reply(writer, 502, f"tinyharness sandbox: couldn't resolve {host}: {e}")
                return None
            if not addresses or any(_local(a) for a in addresses):
                self.denied.append(host)
                log.info("sandbox proxy blocked %s: it resolves to a local address", host)
                await _reply(
                    writer,
                    403,
                    f"tinyharness sandbox: {host} resolves to a local address ({', '.join(addresses)}); "
                    "an allowed name can't reach this machine or its network.",
                )
                return None
            address = addresses[0]
        self.reached.append(host)
        try:
            return await asyncio.wait_for(asyncio.open_connection(address, port), timeout=30)
        except (OSError, TimeoutError) as e:
            await _reply(writer, 502, f"tinyharness sandbox: couldn't reach {host}:{port}: {e}")
            return None


async def _reply(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    body = (message + "\n").encode()
    reason = {400: "Bad Request", 403: "Forbidden", 502: "Bad Gateway"}[status]
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n"
        f"Connection: close\r\n\r\n".encode()
        + body
    )
    with contextlib.suppress(ConnectionError):
        await writer.drain()
    writer.close()


async def _pipe_both(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    up_reader: asyncio.StreamReader,
    up_writer: asyncio.StreamWriter,
) -> None:
    await asyncio.gather(_pipe(reader, up_writer), _pipe(up_reader, writer), return_exceptions=True)


async def _pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    try:
        while data := await source.read(64 * 1024):
            sink.write(data)
            await sink.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            sink.close()
