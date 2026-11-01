"""Small HTTP helpers shared by provider implementations."""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable
from typing import Any

import aiohttp

from .image_format import validate_image_bytes
from .logbook import clip, get_logger, mask
from .net import URLFetchPolicy, ensure_url_allowed

logger = get_logger("http")


def _safe_url(url: str) -> str:
    """Never log query strings: they sometimes carry the API key."""
    text = str(url or "")
    head, sep, _query = text.partition("?")
    return f"{head}?…" if sep else head


def backoff_delay(attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """Jittered exponential delay for a zero-based retry attempt."""
    delay = min(cap, base * (2 ** max(0, int(attempt))))
    delay += delay * 0.15 * (2 * random.random() - 1)
    return max(0.05, min(cap, round(delay, 3)))


def retry_after_seconds(value: str | None) -> float | None:
    """Parse a Retry-After header; HTTP dates are ignored on purpose."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return max(0.0, min(300.0, seconds))


class ProviderHTTP:
    """Aiohttp wrapper kept injectable for unit tests."""

    def __init__(self, timeout: int = 600, policy: URLFetchPolicy | None = None):
        self.timeout = timeout
        self.policy = policy or URLFetchPolicy()

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        data: Any | None = None,
    ) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=max(1, int(self.timeout)))
        started = time.monotonic()
        logger.info(
            "-> %s %s json=%s auth=%s",
            method.upper(),
            _safe_url(url),
            bool(json_body),
            mask(((headers or {}).get("Authorization") or "").replace("Bearer ", "")) or "-",
        )
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                url,
                headers=headers,
                json=json_body,
                data=data,
            ) as response:
                body = await response.read()
                cost = int((time.monotonic() - started) * 1000)
                if response.status < 200 or response.status >= 300:
                    detail = body[:500].decode("utf-8", errors="replace")
                    logger.warning(
                        "<- %s %s HTTP %s in %dms: %s",
                        method.upper(), _safe_url(url), response.status, cost, clip(detail, 240),
                    )
                    raise RuntimeError(f"接口返回 HTTP {response.status}: {detail}")
                logger.info(
                    "<- %s %s HTTP %s in %dms body=%dB",
                    method.upper(), _safe_url(url), response.status, cost, len(body),
                )
                if not body:
                    return {}
                try:
                    result = json.loads(body)
                except Exception as exc:
                    raise RuntimeError("接口返回的内容不是有效 JSON") from exc
                if not isinstance(result, dict):
                    raise RuntimeError("接口返回的 JSON 格式不正确")
                return result

    async def download_image(self, url: str) -> bytes:
        await ensure_url_allowed(url, self.policy)
        timeout = aiohttp.ClientTimeout(total=max(1, int(self.timeout)))
        started = time.monotonic()
        logger.info("download %s", _safe_url(url))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    logger.warning("download failed: HTTP %s for %s", response.status, _safe_url(url))
                    raise RuntimeError(f"图片下载失败: HTTP {response.status}")
                data = await response.read()
        logger.info(
            "downloaded %dB in %dms",
            len(data),
            int((time.monotonic() - started) * 1000),
        )
        if len(data) > 24 * 1024 * 1024:
            raise ValueError("图片超过 24 MB")
        validate_image_bytes(data)
        return data
