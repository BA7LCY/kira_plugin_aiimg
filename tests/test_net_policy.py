"""Result download policy: public hosts pass, private ones need an opt-in."""

from __future__ import annotations

import socket

import pytest

from core.net import (
    URLFetchPolicy,
    ensure_peer_allowed,
    ensure_url_allowed,
    origin,
)

PUBLIC = "93.184.216.34"
PRIVATE = "10.1.2.3"
LOOPBACK = "127.0.0.1"


@pytest.fixture()
def fake_dns(monkeypatch):
    """Resolve every hostname to whatever the test asks for."""

    def use(address: str):
        def resolve(host, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

        monkeypatch.setattr(socket, "getaddrinfo", resolve)

    return use


@pytest.mark.asyncio
async def test_public_result_url_is_allowed(fake_dns):
    fake_dns(PUBLIC)
    await ensure_url_allowed("https://cdn.example.com/a.png", URLFetchPolicy())


@pytest.mark.asyncio
async def test_private_result_url_is_rejected_by_default(fake_dns):
    fake_dns(PRIVATE)
    with pytest.raises(ValueError, match="内网"):
        await ensure_url_allowed("https://files.internal/a.png", URLFetchPolicy())


@pytest.mark.asyncio
async def test_trusted_result_origin_still_needs_the_switch(fake_dns):
    fake_dns(PRIVATE)
    policy = URLFetchPolicy(
        trusted_result_origins=frozenset({"https://files.internal"}),
    )
    with pytest.raises(ValueError, match="内网"):
        await ensure_url_allowed("https://files.internal/a.png", policy)


@pytest.mark.asyncio
async def test_trusted_result_origin_with_switch_allows_private(fake_dns):
    fake_dns(PRIVATE)
    policy = URLFetchPolicy(
        trusted_result_origins=frozenset({"https://files.internal"}),
        allow_private_result_origins=True,
    )
    await ensure_url_allowed("https://files.internal/a.png", policy)


@pytest.mark.asyncio
async def test_switch_alone_does_not_open_untracked_hosts(fake_dns):
    fake_dns(LOOPBACK)
    policy = URLFetchPolicy(allow_private_result_origins=True)
    with pytest.raises(ValueError, match="内网"):
        await ensure_url_allowed("https://other.internal/a.png", policy)


@pytest.mark.asyncio
async def test_provider_origin_is_always_allowed(fake_dns):
    fake_dns(LOOPBACK)
    policy = URLFetchPolicy(trusted_origins=frozenset({"http://127.0.0.1:8188"}))
    await ensure_url_allowed("http://127.0.0.1:8188/view/1.png", policy)


@pytest.mark.asyncio
async def test_loopback_hostname_is_rejected_without_dns(fake_dns):
    fake_dns(PUBLIC)
    with pytest.raises(ValueError, match="本机"):
        await ensure_url_allowed("http://localhost:8080/a.png", URLFetchPolicy())


@pytest.mark.asyncio
async def test_non_http_scheme_is_rejected():
    with pytest.raises(ValueError, match="无效"):
        await ensure_url_allowed("file:///etc/passwd", URLFetchPolicy())


def test_origin_normalises_trailing_slash_and_path():
    assert origin("https://files.internal/a/b.png?x=1") == "https://files.internal"
    assert origin("") == ""


class _Conn:
    def __init__(self, peer):
        self.transport = _Transport(peer)


class _Transport:
    def __init__(self, peer):
        self._peer = peer

    def get_extra_info(self, name):
        return self._peer if name == "peername" else None


class _Response:
    def __init__(self, peer):
        self.connection = _Conn(peer) if peer else None


@pytest.mark.asyncio
async def test_peer_check_passes_for_a_public_address():
    await ensure_peer_allowed(
        _Response((PUBLIC, 443)), "https://cdn.example.com/a.png", URLFetchPolicy()
    )


@pytest.mark.asyncio
async def test_peer_check_blocks_a_rebound_host():
    """The name resolved public, but the socket landed on a private address."""
    with pytest.raises(ValueError, match="内网"):
        await ensure_peer_allowed(
            _Response((PRIVATE, 443)),
            "https://cdn.example.com/a.png",
            URLFetchPolicy(),
        )


@pytest.mark.asyncio
async def test_peer_check_respects_the_private_result_switch():
    policy = URLFetchPolicy(
        trusted_result_origins=frozenset({"https://files.internal"}),
        allow_private_result_origins=True,
    )
    await ensure_peer_allowed(
        _Response((PRIVATE, 443)), "https://files.internal/a.png", policy
    )


@pytest.mark.asyncio
async def test_peer_check_rejects_unknown_peers():
    """拿不到对端地址时不再放行：放行就等于复核形同虚设。"""
    with pytest.raises(ValueError, match="无法确认"):
        await ensure_peer_allowed(
            _Response(None), "https://cdn.example.com/a.png", URLFetchPolicy()
        )


@pytest.mark.asyncio
async def test_peer_check_rejects_unparsable_peer():
    """有对端地址但解析不出 IP，同样属于无法确认，按拒绝处理。"""
    with pytest.raises(ValueError, match="无法确认"):
        await ensure_peer_allowed(
            _Response(("not-an-ip", 443)),
            "https://cdn.example.com/a.png",
            URLFetchPolicy(),
        )


@pytest.mark.asyncio
async def test_peer_check_still_trusts_allowlisted_origin():
    """白名单里的 origin 不看对端地址，照旧直接放行。"""
    policy = URLFetchPolicy(trusted_origins=frozenset({"https://cdn.example.com"}))
    await ensure_peer_allowed(
        _Response(None), "https://cdn.example.com/a.png", policy
    )
