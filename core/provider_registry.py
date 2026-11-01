"""Provider backend registry: builds and caches backends from config.

The `providers` config array holds one entry per provider. Each entry carries
its template key (or a short alias), endpoint settings and model. Backends
are built lazily and cached, so a broken provider does not prevent the plugin
from loading and does not block other providers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .net import URLFetchPolicy
from .providers.agnes_video import AgnesVideoService
from .providers.base import ImageBackend, ProviderSettings, VideoBackend
from .providers.openai_chat import OpenAIChatImageBackend
from .providers.openai_compat import OpenAICompatBackend
from .providers.openai_full_url import OpenAIFullURLBackend
from .sizes import SUPPORTED_SIZES, normalize_ratio_default_sizes, normalize_size_text

DEFAULT_PROVIDER_TIMEOUT_SECONDS = 600

# Template keys, with the short aliases accepted in older configs.
TEMPLATE_ALIASES: dict[str, str] = {
    "openai": "openai_images",
    "openai_compat": "openai_images",
    "openai_images": "openai_images",
    "openai_chat": "openai_chat",
    "openai_full_url": "openai_full_url_images",
    "openai_full_url_images": "openai_full_url_images",
    "agnes_video": "agnes_video",
}

_IMAGE_TEMPLATES = {"openai_images", "openai_chat", "openai_full_url_images"}
_VIDEO_TEMPLATES = {"agnes_video"}

# Minimal fields each template needs to be usable.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "openai_images": ("base_url", "model"),
    "openai_chat": ("base_url", "model"),
    "openai_full_url_images": ("full_generate_url",),
    "agnes_video": ("base_url", "api_keys"),
}


def normalize_template_key(raw: Any) -> str:
    key = str(raw or "").strip()
    if not key:
        return ""
    return TEMPLATE_ALIASES.get(key, key)


class ProviderRegistry:
    """Loads provider config entries and builds backend instances on demand."""

    def __init__(
        self,
        config: dict,
        *,
        data_dir: Path,
        policy: URLFetchPolicy | None = None,
    ):
        self._config = config or {}
        self._data_dir = Path(data_dir)
        self._policy = policy or URLFetchPolicy()
        self._image: dict[str, ProviderSettings] = {}
        self._video: dict[str, ProviderSettings] = {}
        self._image_backends: dict[str, ImageBackend] = {}
        self._video_backends: dict[str, VideoBackend] = {}
        self._errors: list[str] = []
        self._load()

    # -- loading ------------------------------------------------------------

    def _load(self) -> None:
        entries = self._config.get("providers", []) or []
        if not isinstance(entries, list):
            self._errors.append("providers 配置不是数组")
            return
        for entry in entries:
            if not isinstance(entry, dict):
                self._errors.append("providers 配置项不是对象")
                continue
            provider_id = str(entry.get("id") or "").strip()
            if not provider_id:
                self._errors.append("providers 配置项缺少 id")
                continue
            try:
                settings = self._settings_for(entry)
            except Exception as exc:
                self._errors.append(f"provider {provider_id}: {exc}")
                continue
            if settings.template_key in _IMAGE_TEMPLATES:
                self._image[provider_id] = settings
            elif settings.template_key in _VIDEO_TEMPLATES:
                self._video[provider_id] = settings
            else:
                self._errors.append(f"provider {provider_id}: 未知模板 {settings.template_key!r}")

    def _settings_for(self, entry: dict) -> ProviderSettings:
        template_key = normalize_template_key(
            entry.get("__template_key")
            or entry.get("template_key")
            or entry.get("type")
            or entry.get("provider_type")
            or entry.get("id")
        )
        settings = ProviderSettings(
            id=str(entry.get("id") or "").strip(),
            template_key=template_key,
            base_url=str(entry.get("base_url") or "").strip(),
            api_keys=[
                str(k).strip()
                for k in (entry.get("api_keys") or [])
                if str(k).strip()
            ],
            model=str(entry.get("model") or "").strip(),
            default_size=normalize_size_text(
                str(entry.get("default_size") or "1024x1024")
            ),
            supports_edit=(
                bool(str(entry.get("full_edit_url") or "").strip())
                if template_key == "openai_full_url_images"
                else bool(entry.get("supports_edit", True))
            ),
            extra_body=dict(entry.get("extra_body") or {}),
            proxy_url=str(entry.get("proxy_url") or "").strip(),
            output_format=str(entry.get("output_format") or "jpeg").strip().lower(),
            timeout=int(
                entry.get("timeout_seconds")
                or entry.get("timeout")
                or DEFAULT_PROVIDER_TIMEOUT_SECONDS
            ),
            max_retries=int(entry.get("max_retries") or 2),
            max_concurrent=max(0, min(32, int(entry.get("max_concurrent") or 0))),
            full_generate_url=str(entry.get("full_generate_url") or "").strip(),
            full_edit_url=str(entry.get("full_edit_url") or "").strip(),
            edit_model=str(entry.get("edit_model") or "").strip(),
            generate_request_mode=str(entry.get("generate_request_mode") or "").strip(),
            edit_request_mode=str(entry.get("edit_request_mode") or "").strip(),
            enable_stream_generate=bool(entry.get("enable_stream_generate", False)),
            enable_stream_edit=bool(entry.get("enable_stream_edit", False)),
            prompt_template=str(entry.get("prompt_template") or "{{prompt}}").strip(),
            server_url=str(entry.get("server_url") or "").strip(),
            interval=int(entry.get("request_interval_seconds") or entry.get("interval") or 61),
            label=str(entry.get("label") or "").strip(),
            mode=str(entry.get("mode") or "auto").strip().lower(),
            seconds=str(entry.get("seconds") or "12").strip(),
            size=str(entry.get("size") or "720P").strip(),
            aspect_ratio=str(entry.get("aspect_ratio") or "16:9").strip(),
            request_timeout=int(entry.get("request_timeout_seconds") or 120),
            poll_interval=int(entry.get("poll_interval_seconds") or 61),
            poll_retries=min(5, max(0, int(entry.get("max_retries") or 3))),
            allowed_sizes=[
                normalize_size_text(s)
                for s in (entry.get("allowed_sizes") or [])
                if normalize_size_text(s)
            ],
            ratio_default_sizes=normalize_ratio_default_sizes(
                entry.get("ratio_default_sizes") or {}
            ),
            async_mode=str(entry.get("async_mode") or "auto").strip().lower(),
        )

        if template_key == "openai_images" and not settings.allowed_sizes:
            settings.allowed_sizes = list(SUPPORTED_SIZES)

        missing = [
            field
            for field in REQUIRED_FIELDS.get(template_key, ())
            if not getattr(settings, field)
        ]
        if missing:
            raise ValueError(f"缺少必填配置: {', '.join(missing)}")
        return settings

    # -- queries ------------------------------------------------------------

    def has(self, provider_id: str) -> bool:
        return provider_id in self._image or provider_id in self._video

    def image_provider_ids(self) -> list[str]:
        return list(self._image)

    def video_provider_ids(self) -> list[str]:
        return list(self._video)

    def get_settings(self, provider_id: str) -> ProviderSettings | None:
        return self._image.get(provider_id) or self._video.get(provider_id)

    def list_settings(self) -> list[dict]:
        """Full provider settings for the WebUI editor round-trip."""
        from dataclasses import asdict

        return [
            asdict(settings)
            for settings in [*self._image.values(), *self._video.values()]
        ]

    def validate(self) -> list[str]:
        return list(self._errors)

    def list_providers(self) -> list[dict]:
        out: list[dict] = []
        for provider_id, settings in {**self._image, **self._video}.items():
            out.append(
                {
                    "id": provider_id,
                    "template_key": settings.template_key,
                    "base_url": settings.base_url,
                    "model": settings.model,
                    "supports_edit": settings.supports_edit,
                    "has_keys": bool(settings.api_keys),
                    "capabilities": (
                        ["generate", "edit"] if settings.supports_edit else ["generate"]
                    )
                    if provider_id in self._image
                    else ["video"],
                }
            )
        return out

    # -- backend construction ------------------------------------------------

    def get_backend(self, provider_id: str) -> ImageBackend:
        settings = self._image.get(provider_id)
        if settings is None:
            raise KeyError(f"图片服务商不存在: {provider_id}")
        backend = self._image_backends.get(provider_id)
        if backend is None:
            if settings.template_key == "openai_chat":
                backend = OpenAIChatImageBackend(settings, policy=self._policy)
            elif settings.template_key == "openai_full_url_images":
                backend = OpenAIFullURLBackend(settings, policy=self._policy)
            else:
                backend = OpenAICompatBackend(settings, policy=self._policy)
            self._image_backends[provider_id] = backend
        return backend

    def get_video_backend(self, provider_id: str) -> VideoBackend:
        settings = self._video.get(provider_id)
        if settings is None:
            raise KeyError(f"视频服务商不存在: {provider_id}")
        backend = self._video_backends.get(provider_id)
        if backend is None:
            backend = AgnesVideoService(settings, data_dir=self._data_dir)
            self._video_backends[provider_id] = backend
        return backend

    async def close(self) -> None:
        for backend in self._image_backends.values():
            try:
                await backend.close()
            except Exception:
                pass
        for backend in self._video_backends.values():
            try:
                await backend.close()
            except Exception:
                pass
        self._image_backends.clear()
        self._video_backends.clear()
