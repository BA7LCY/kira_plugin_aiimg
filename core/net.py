"""Conservative media URL checks used when downloading provider results."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class URLFetchPolicy:
    trusted_origins: frozenset[str] = frozenset()
    dns_timeout_seconds: float = 2.0
    # Result URLs from these origins may resolve to private/loopback IPs.
    # The provider origin is always allowed; this covers file hosts the
    # image service redirects to (e.g. a CDN on the same private network).
    trusted_result_origins: frozenset[str] = frozenset()
    # When false, result downloads to private IPs are rejected outright.
    # When true, they are allowed only when the origin is trusted above.
    allow_private_result_origins: bool = False


def origin(url: str) -> str:
    parts = urlsplit(str(url or ""))
    if not parts.scheme or not parts.netloc:
        return ""
    return f"{parts.scheme}://{parts.netloc}".rstrip("/")


def trusted_origins_from_config(config: dict) -> frozenset[str]:
    values: set[str] = set()
    for provider in config.get("providers", []) or []:
        if not isinstance(provider, dict):
            continue
        for key in (
            "base_url",
            "api_url",
            "full_generate_url",
            "full_edit_url",
            "server_url",
        ):
            value = origin(str(provider.get(key) or ""))
            if value:
                values.add(value)
    return frozenset(values)


async def ensure_url_allowed(url: str, policy: URLFetchPolicy) -> None:
    parts = urlsplit(str(url or ""))
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("图片地址无效")
    current_origin = origin(str(url))
    if current_origin and current_origin in policy.trusted_origins:
        return
    host = (parts.hostname or "").lower()
    if host in {"localhost"} or host.endswith((".localhost", ".local")):
        raise ValueError("图片地址不允许指向本机或内网名称")

    def resolve() -> list[str]:
        result: list[str] = []
        for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
            address = item[4][0]
            if address not in result:
                result.append(address)
        return result

    try:
        addresses = await asyncio.wait_for(
            asyncio.to_thread(resolve), timeout=policy.dns_timeout_seconds
        )
    except Exception as exc:
        raise ValueError("图片域名解析失败") from exc
    if not addresses:
        raise ValueError("图片域名解析失败")
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError as exc:
            raise ValueError("图片域名解析结果无效") from exc
        if ip.is_global:
            continue
        # Private/loopback addresses are only reachable when the caller opted
        # in with a trusted result origin.
        if (
            current_origin
            and policy.allow_private_result_origins
            and current_origin in policy.trusted_result_origins
        ):
            continue
        raise ValueError("图片地址不允许指向内网")


def peer_address(response: object) -> str:
    """Actual peer IP of a live HTTP response connection, or ""."""
    connection = getattr(response, "connection", None)
    transport = getattr(connection, "transport", None) if connection is not None else None
    peer = transport.get_extra_info("peername") if transport is not None else None
    if not peer:
        return ""
    return str(peer[0] if isinstance(peer, (tuple, list)) else peer)


async def ensure_peer_allowed(
    response: object, url: str, policy: URLFetchPolicy
) -> None:
    """Re-check the address that was actually connected to.

    ``ensure_url_allowed`` resolves the name first; a hostname could still
    resolve to something else by the time the socket connects. Comparing the
    peer address closes that window for result downloads.
    """
    current_origin = origin(str(url))
    if current_origin and current_origin in policy.trusted_origins:
        return
    if (
        current_origin
        and policy.allow_private_result_origins
        and current_origin in policy.trusted_result_origins
    ):
        return
    address = peer_address(response)
    if not address:
        return
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return
    if not ip.is_global:
        raise ValueError("图片地址实际连接到内网")
