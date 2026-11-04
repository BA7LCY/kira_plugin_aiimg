"""Full-URL image backend (generate + edit) for providers with arbitrary REST endpoints."""

from __future__ import annotations

import asyncio
import base64
import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp

from ..http import backoff_delay, current_session, retry_after_seconds
from ..logbook import get_logger
from ..image_format import (
    decode_base64_image,
    guess_image_mime_and_ext,
    is_placeholder_image,
    log_size_mismatch,
    validate_image_bytes,
)
from ..net import URLFetchPolicy, ensure_url_allowed
from ..output_spec import OutputIntent, size_from_ratio_and_resolution
from ..sizes import normalize_size_text
from .base import ImageBackend, ImageResult, KeyPoolMixin, ProviderSettings

logger = get_logger("provider.full_url")


def _looks_like_pixels(value: str | None) -> bool:
    return bool(re.fullmatch(r"\d{2,5}x\d{2,5}", str(value or "").strip()))


def _is_valid_data_image_ref(raw: str) -> bool:
    if not raw or not raw.startswith("data:image/"):
        return False
    head, _, _ = raw.partition(",")
    if ";" not in head:
        return False
    if "base64" not in head:
        return False
    return len(raw) >= 64


def _extract_image_ref(obj, depth: int = 0) -> str | None:
    """Recursively find an image URL/base64 inside a provider JSON response."""
    if depth > 8:
        return None
    if isinstance(obj, str):
        if obj.startswith(("http://", "https://", "data:image/", "base64://")):
            return obj
        return None
    if isinstance(obj, dict):
        for key in (
            "data",
            "url",
            "output",
            "b64_json",
            "image_url",
            "image",
            "src",
            "uri",
            "link",
            "final_image_url",
            "fifeUrl",
            "finalUrl",
            "fife",
            "images",
            "image_urls",
            "attachments",
            "media",
            "result",
            "response",
        ):
            value = obj.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                if value.startswith(("http://", "https://", "data:image/", "base64://")):
                    return value
                continue
            found = _extract_image_ref(value, depth + 1)
            if found:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract_image_ref(item, depth + 1)
            if found:
                return found
    return None


def build_edit_multipart_fields(
    payload: dict,
    images: list[Path],
) -> list[tuple[str, object, dict]]:
    """按「先图片、后文字」的顺序排好编辑请求的表单字段。

    返回 (字段名, 值, add_field 参数) 三元组列表，图片文件字段在前、
    提示词等文本字段在后。服务端一般按字段顺序处理表单，图片先到、
    文字后到可以避免个别服务只取第一个字段当输入图而忽略参考图。
    """
    fields: list[tuple[str, object, dict]] = []
    for index, path in enumerate(images, start=1):
        data = Path(path).read_bytes()
        mime, ext = guess_image_mime_and_ext(data)
        fields.append(
            (
                "image",
                data,
                {"filename": f"image_{index}.{ext}", "content_type": mime},
            )
        )
    for key, value in payload.items():
        fields.append((str(key), str(value), {}))
    return fields


def build_edit_multipart(payload: dict, images: list[Path]) -> aiohttp.FormData:
    """用排好序的字段构造 multipart 表单（图片在前、文字在后）。"""
    form = aiohttp.FormData()
    for name, value, options in build_edit_multipart_fields(payload, images):
        form.add_field(name, value, **options)
    return form


class OpenAIFullURLBackend(KeyPoolMixin, ImageBackend):
    """POSTs a JSON body to a configured endpoint and extracts the image."""

    def __init__(
        self,
        settings: ProviderSettings,
        *,
        policy: URLFetchPolicy | None = None,
        image_max_bytes: int | None = None,
    ):
        self.settings = settings
        self.provider_id = settings.id
        self.supports_edit = settings.supports_edit
        self._image_max_bytes = image_max_bytes
        self.timeout = int(settings.timeout or 600)
        self.max_retries = settings.max_retries
        self.default_model = str(settings.model or "").strip()
        self.default_size = normalize_size_text(str(settings.default_size or "1024x1024"))
        self.extra_body = dict(settings.extra_body or {})
        self.proxy_url = str(settings.proxy_url or "").strip() or None
        self.full_generate_url = str(settings.full_generate_url or "").strip()
        self.full_edit_url = str(settings.full_edit_url or "").strip()
        self.edit_model = str(settings.edit_model or "").strip()
        self._policy = policy or URLFetchPolicy()
        self._init_key_pool(settings)

    # -- session ------------------------------------------------------------

    def _request_timeout(self) -> aiohttp.ClientTimeout:
        """单次请求超时：沿用原来的总超时，不依赖共享会话的默认值。"""
        return aiohttp.ClientTimeout(total=max(1, int(self.timeout)))

    async def _session_get(self) -> aiohttp.ClientSession:
        """复用进程内共享会话，不再为每个服务商单独建连接池。"""
        logger.debug("完整地址接口复用共享会话")
        return await current_session()

    async def close(self) -> None:
        """共享会话由插件统一关闭，服务商这里只保留空实现。"""
        logger.debug("完整地址接口关闭：共享会话由插件统一管理，此处不关闭")

    def _headers(self, key: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _key_pool(self) -> list[str]:
        return [k for k in (self.settings.api_keys or []) if str(k).strip()]

    def _key_logger(self):
        return logger

    def _next_key(self) -> str:
        return self._pick_key()

    async def _post_json(self, url: str, payload: dict) -> dict | str:
        session = await self._session_get()
        attempt = 0
        while True:
            key = self._next_key()
            async with session.post(
                url,
                headers=self._headers(key),
                json=payload,
                proxy=self.proxy_url or None,
                timeout=self._request_timeout(),
            ) as resp:
                ctype = resp.headers.get("Content-Type", "").lower()
                body = await resp.read()
                if resp.status in {400, 422} and "size" in str(payload.get("size", "")).lower():
                    # Some full-URL services reject size hints; retry with auto.
                    if payload.get("size") and payload.get("size") != "auto":
                        payload["size"] = "auto"
                        continue
                if resp.status in range(200, 300):
                    if ctype.startswith("image/"):
                        validate_image_bytes(body)
                        return body
                    if not body:
                        return {}
                    try:
                        return json.loads(body)
                    except Exception:
                        return {}
                detail = body[:300].decode("utf-8", errors="replace")
                self._handle_key_status(key, resp.status)
                error = RuntimeError(f"HTTP {resp.status}: {detail}")
                if resp.status in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                    delay = retry_after if retry_after is not None else backoff_delay(attempt)
                    attempt += 1
                    logger.warning(
                        "完整地址接口 HTTP %s，第 %d/%d 次重试，等 %.1f 秒，响应 %dB",
                        resp.status, attempt, self.max_retries, delay, len(body),
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("完整地址接口 HTTP %s 最终失败，响应 %dB", resp.status, len(body))
                raise error

    async def _download(self, url: str) -> bytes:
        await ensure_url_allowed(url, self._policy)
        session = await self._session_get()
        async with session.get(
            url,
            proxy=self.proxy_url or None,
            allow_redirects=False,
            timeout=self._request_timeout(),
        ) as resp:
            if resp.status != 200:
                raise RuntimeError(f"image download failed: HTTP {resp.status}")
            data = await resp.read()
        cap = int(getattr(self, "_image_max_bytes", 0) or 0) or (24 * 1024 * 1024)
        if len(data) > cap:
            raise ValueError(f"图片超过 {cap // (1024 * 1024)} MB")
        validate_image_bytes(data, cap)
        return data

    def _rewrite_loopback_ref(self, raw: str, endpoint: str) -> str:
        """Rewrite 127.0.0.1 references to the endpoint host (services behind proxies)."""
        try:
            target = urlsplit(raw)
            if target.hostname not in ("127.0.0.1", "localhost", "0.0.0.0"):
                return raw
            base = urlsplit(endpoint)
            host = base.hostname or ""
            port = base.port
            netloc = host if not port else f"{host}:{port}"
            return urljoin(endpoint, f"{target.path}?{target.query}".lstrip("/"))
        except Exception:
            return raw

    async def _save_ref(self, raw: str, endpoint: str, model: str, size: str) -> ImageResult:
        if raw.startswith("data:image/") or raw.startswith("base64://"):
            if raw.startswith("base64://"):
                raw = raw[len("base64://") :]
            else:
                _, _, b64 = raw.partition(",")
                raw = b64
            try:
                data = base64.b64decode(raw)
            except Exception as exc:
                raise RuntimeError("provider returned invalid base64 image") from exc
            validate_image_bytes(data)
        elif raw.startswith("http://") or raw.startswith("https://"):
            data = await self._download(raw)
        elif raw.startswith("/"):
            data = await self._download(self._rewrite_loopback_ref(raw, endpoint))
        else:
            raise RuntimeError("provider response contains no readable image reference")
        if is_placeholder_image(data):
            logger.warning("[FullURL] 服务商返回了占位图（%dB），丢弃", len(data))
            raise RuntimeError("服务商返回了占位图，无法使用")
        log_size_mismatch(logger, data, requested_size=size, source="FullURL")
        mime, ext = guess_image_mime_and_ext(data)
        return ImageResult(data=data, mime=mime, ext=ext, size=size, model=model)

    async def generate(
        self,
        prompt: str,
        *,
        intent: OutputIntent | None = None,
        size_hint: str = "",
        model: str = "",
        extra: dict | None = None,
    ) -> list[ImageResult]:
        if not self.full_generate_url:
            raise RuntimeError("provider has no full_generate_url configured")
        final_model = str(model or self.default_model or "").strip()
        final_size = normalize_size_text(size_hint) or self.default_size
        if not _looks_like_pixels(final_size):
            final_size = self.default_size
        if intent and intent.exact_size:
            final_size = intent.exact_size
        elif intent and intent.aspect_ratio:
            selected = size_from_ratio_and_resolution(
                intent.aspect_ratio,
                intent.resolution,
                default_size=self.default_size,
            )
            if selected:
                final_size = selected

        payload = {
            "model": final_model,
            "prompt": prompt,
            "size": final_size,
        }
        payload.update(self.extra_body)
        payload.update(extra or {})

        resp = await self._post_json(self.full_generate_url, payload)
        if isinstance(resp, bytes):
            mime, ext = guess_image_mime_and_ext(resp)
            return [ImageResult(data=resp, mime=mime, ext=ext, size=final_size, model=final_model)]

        raw = _extract_image_ref(resp)
        if raw is None:
            raise RuntimeError("provider response does not contain an image reference")
        result = await self._save_ref(raw, self.full_generate_url, final_model, final_size)
        return [result]

    async def edit(
        self,
        prompt: str,
        images: list[Path],
        *,
        intent: OutputIntent | None = None,
        size_hint: str = "",
        model: str = "",
        extra: dict | None = None,
    ) -> list[ImageResult]:
        if not self.supports_edit:
            raise RuntimeError("provider does not support edit / image-to-image")
        if not images:
            raise ValueError("edit requires at least one image")
        endpoint = self.full_edit_url or self.full_generate_url
        if not endpoint:
            raise RuntimeError("provider has no full_edit_url / full_generate_url configured")

        final_model = str(model or self.edit_model or self.default_model or "").strip()
        final_size = normalize_size_text(size_hint) or self.default_size
        if not _looks_like_pixels(final_size):
            final_size = self.default_size
        if intent and intent.exact_size:
            final_size = intent.exact_size
        elif intent and intent.aspect_ratio:
            selected = size_from_ratio_and_resolution(
                intent.aspect_ratio,
                intent.resolution,
                default_size=self.default_size,
            )
            if selected:
                final_size = selected

        image_data_urls: list[str] = []
        for path in images:
            with open(path, "rb") as fh:
                data = fh.read()
            image_data_urls.append(_data_url_of(data))

        payload = {
            "model": final_model,
            "prompt": prompt,
            "size": final_size,
        }
        payload.update(self.extra_body)
        payload.update(extra or {})

        # Prefer a JSON field shape; fall back to multipart on 400/415/422.
        json_shapes = [
            {"image": image_data_urls[0]},
            {"images": image_data_urls},
            {"image_url": image_data_urls[0]},
            {"object_refs": [{"url": u, "type": "image_url"} for u in image_data_urls]},
        ]
        resp = None
        last_error: Exception | None = None
        for shape in json_shapes:
            # 图片字段先写进请求体、文字字段跟在后面：先图后文的顺序同时适用
            # 于 JSON 请求体，服务端会按字段出现顺序解析输入。
            # 这里用 setdefault 合并，保持原先「图片字段优先」的取值规则。
            trial = dict(shape)
            for key, value in payload.items():
                trial.setdefault(key, value)
            try:
                resp = await self._post_json(endpoint, trial)
                last_error = None
                break
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                if not any(f"HTTP {code}" in message for code in ("400", "415", "422")):
                    break
        if resp is None:
            if last_error is not None and not any(
                f"HTTP {code}" in str(last_error) for code in ("400", "415", "422")
            ):
                raise last_error
            return await self._edit_multipart(endpoint, payload, images, final_model, final_size)

        if isinstance(resp, bytes):
            mime, ext = guess_image_mime_and_ext(resp)
            return [ImageResult(data=resp, mime=mime, ext=ext, size=final_size, model=final_model)]
        raw = _extract_image_ref(resp)
        if raw is None:
            raise RuntimeError("provider response does not contain an image reference")
        result = await self._save_ref(raw, endpoint, final_model, final_size)
        return [result]

    async def _edit_multipart(
        self,
        endpoint: str,
        payload: dict,
        images: list[Path],
        model: str,
        size: str,
    ) -> list[ImageResult]:
        session = await self._session_get()
        # 表单里图片文件字段排在提示词等文字字段之前，见 build_edit_multipart_fields。
        form = build_edit_multipart(payload, images)
        async with session.post(
            endpoint,
            headers=self._headers(self._next_key()),
            data=form,
            proxy=self.proxy_url or None,
            timeout=self._request_timeout(),
        ) as resp:
            body = await resp.read()
            if resp.status not in range(200, 300):
                detail = body[:300].decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {resp.status}: {detail}")
            ctype = resp.headers.get("Content-Type", "").lower()
            if ctype.startswith("image/"):
                validate_image_bytes(body)
                mime, ext = guess_image_mime_and_ext(body)
                return [ImageResult(data=body, mime=mime, ext=ext, size=size, model=model)]
            try:
                parsed = json.loads(body) if body else {}
            except Exception:
                parsed = {}
        raw = _extract_image_ref(parsed)
        if raw is None:
            raise RuntimeError("provider response does not contain an image reference")
        result = await self._save_ref(raw, endpoint, model, size)
        return [result]


def _data_url_of(data: bytes) -> str:
    from ..image_format import image_data_url

    return image_data_url(data)
