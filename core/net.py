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
        if not ip.is_global:
            raise ValueError("图片地址不允许指向内网")
