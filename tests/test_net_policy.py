"""Result download policy: public hosts pass, private ones need an opt-in."""

from __future__ import annotations

import socket

import pytest

from core.net import URLFetchPolicy, ensure_url_allowed, origin

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
