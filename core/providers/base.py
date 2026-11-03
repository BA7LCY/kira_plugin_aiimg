"""Shared result types and backend contracts for providers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ImageResult:
    """One generated/edited image carried as raw bytes."""

    data: bytes
    mime: str
    ext: str
    size: str
    model: str


@dataclass(frozen=True)
class VideoResult:
    """A generated video referenced by its remote URL."""

    url: str
    model: str
    size: str
    duration: str = ""


class StopProviderChain(Exception):
    """Raised when a provider decides fallback must stop.

    Transport errors, authentication failures and hard endpoint failures do
    not fall back to the next provider in the chain.
    """


class KeyPoolMixin:
    """Per-key cooldown shared by all image backends.

    A key that returns 401/403/429 is parked until `key_cooldown_seconds`
    elapses (0 disables parking). `_next_key` then only returns keys that are
    not parked, so a pool survives one bad key without burning every request.
    """

    # HTTP statuses that mean "this key is bad, stop using it for a while".
    _KEY_BAD_STATUSES = frozenset({401, 403, 429})

    def _init_key_pool(self, settings: ProviderSettings) -> None:
        self._key_cooldown_seconds = max(0, int(settings.key_cooldown_seconds or 0))
        self._key_cooldown_until: dict[str, float] = {}
        self._key_index = 0

    def _mark_key_bad(self, key: str) -> None:
        if self._key_cooldown_seconds <= 0 or not key:
            return
        self._key_cooldown_until[key] = time.monotonic() + self._key_cooldown_seconds
        logger = self._key_logger()
        if logger is not None:
            logger.warning(
                "密钥被服务商拒绝，暂时停用 %d 秒（剩余可用 %d 个）",
                self._key_cooldown_seconds,
                len(self._key_pool()),
            )

    def _key_pool(self) -> list[str]:
        raise NotImplementedError

    def _key_logger(self):
        return None

    def _pick_key(self) -> str:
        keys = self._key_pool()
        if not keys:
            raise RuntimeError("provider has no API key configured")
        now = time.monotonic()
        for _ in range(len(keys)):
            key = keys[self._key_index % len(keys)]
            self._key_index = (self._key_index + 1) % len(keys)
            until = self._key_cooldown_until.get(key, 0.0)
            if until > now:
                continue
            return key
        # Every key is parked; fall back to the next one anyway so the caller
        # surfaces the real provider error instead of a confusing "no key".
        return keys[(self._key_index - 1) % len(keys)]

    def _handle_key_status(self, key: str, status: int) -> None:
        if status in self._KEY_BAD_STATUSES:
            self._mark_key_bad(key)


@dataclass
class ProviderSettings:
    """Normalized view of one config entry in the `providers` array."""

    id: str
    template_key: str = ""
    base_url: str = ""
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    default_size: str = "1024x1024"
    supports_edit: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    proxy_url: str = ""
    output_format: str = "jpeg"
    timeout: int = 600
    max_retries: int = 2
    max_concurrent: int = 0
    full_generate_url: str = ""
    full_edit_url: str = ""
    edit_model: str = ""
    generate_request_mode: str = ""
    edit_request_mode: str = ""
    enable_stream_generate: bool = False
    enable_stream_edit: bool = False
    prompt_template: str = ""
    server_url: str = ""
    interval: int = 61
    label: str = ""
    mode: str = "auto"
    seconds: str = "12"
    size: str = "720P"
    aspect_ratio: str = "16:9"
    request_timeout: int = 120
    poll_interval: int = 61
    poll_retries: int = 3
    allowed_sizes: list[str] = field(default_factory=list)
    ratio_default_sizes: dict[str, str] = field(default_factory=dict)
    async_mode: str = "auto"
    # Per-key cooldown in seconds: a key that returns 401/403/429 is parked
    # for this long before it can be tried again (0 disables the feature).
    key_cooldown_seconds: int = 300


class ImageBackend:
    """Base class for image providers (generate + optional edit)."""

    provider_id: str = ""
    supports_edit: bool = True

    async def generate(
        self,
        prompt: str,
        *,
        intent: Any = None,
        size_hint: str = "",
        model: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[ImageResult]:
        raise NotImplementedError

    async def edit(
        self,
        prompt: str,
        images: list[Path],
        *,
        intent: Any = None,
        size_hint: str = "",
        model: str = "",
        extra: dict[str, Any] | None = None,
    ) -> list[ImageResult]:
        raise NotImplementedError

    async def close(self) -> None:
        pass


class VideoBackend:
    """Base class for video providers."""

    provider_id: str = ""

    async def generate_video(
        self,
        prompt: str,
        *,
        seconds: str = "",
        mode: str = "",
        images: list[Path] | None = None,
        audios: list[Path] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> VideoResult:
        raise NotImplementedError

    async def close(self) -> None:
        pass
