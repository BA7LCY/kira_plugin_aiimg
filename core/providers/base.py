"""Shared result types and backend contracts for providers."""

from __future__ import annotations

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
