"""OpenAI-compatible Images API backend (generate + edit), HTTP implementation.

Plain aiohttp calls against `/images/generations` and `/images/edits` handle
size resolution, the 404 cooldown and the 4096x4096 degradation path. No
hidden prompts are introduced.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from ..http import backoff_delay, retry_after_seconds
from ..logbook import get_logger
from ..image_format import guess_image_mime_and_ext, validate_image_bytes
from ..net import URLFetchPolicy, ensure_url_allowed
from ..output_spec import (
    OutputIntent,
    is_gpt_image_2_model,
    resolve_gpt_image_2_size,
    select_allowed_size,
    size_from_ratio_and_resolution,
    to_request_size_args,
)
from ..sizes import (
    normalize_size_text,
    ratio_defaults_from_sizes,
    size_to_ratio,
)
from .base import ImageBackend, ImageResult, ProviderSettings

logger = get_logger("provider.compat")


def root_url(raw: str) -> str:
    """Strip the API version segment so service-level endpoints can be built."""
    parts = urlsplit(str(raw or "").strip())
    if not parts.scheme or not parts.netloc:
        return str(raw or "").strip().rstrip("/")
    path = parts.path.rstrip("/")
    for tail in ("/v1", "/openai/v1", "/api/v1"):
        if path.endswith(tail):
            path = path[: -len(tail)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def normalize_openai_compat_base_url(raw: str) -> str:
    """Normalize an OpenAI-compatible base_url.

    Accepts forms like https://api.example.com, https://api.example.com/v1,
    or https://proxy.example.com/openai/v1, and strips trailing endpoint
    paths a user may have pasted.
    """
    s = str(raw or "").strip()
    if not s:
        return ""
    s = s.rstrip("/")

    lower = s.lower()
    for suffix in (
        "/v1/chat/completions",
        "/chat/completions",
        "/v1/images/generations",
        "/images/generations",
        "/v1/images/edits",
        "/images/edits",
        "/v1/images/edit",
        "/images/edit",
        "/v1/images",
        "/images",
    ):
        if lower.endswith(suffix):
            s = s[: -len(suffix)].rstrip("/")
            break

    if re.search(r"/v1($|/)", s):
        return s

    try:
        parts = urlsplit(s)
        if parts.scheme and parts.netloc:
            path = (parts.path or "").rstrip("/") + "/v1"
            return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")
    except Exception:
        pass

    return f"{s}/v1"


def _looks_like_size(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2,5}x\d{2,5}", (s or "").strip()))


def resolution_to_size(resolution: str) -> str | None:
    r = (resolution or "").strip().upper()
    if not r or r == "AUTO":
        return None
    if r in {"1K", "1024"}:
        return "1024x1024"
    if r in {"2K", "2048"}:
        return "2048x2048"
    if r in {"4K", "4096"}:
        return "4096x4096"
    if _looks_like_size(r.lower()):
        return r.lower()
    return None


def _build_collage(images: list[bytes]) -> bytes:
    """Combine reference images into one image for single-input backends."""
    if not images:
        return b""
    if len(images) == 1:
        return images[0]
    try:
        from PIL import Image as PILImage
    except Exception:
        return images[0]

    pil_images: list[PILImage.Image] = []
    for b in images:
        try:
            pil_images.append(PILImage.open(io.BytesIO(b)).convert("RGB"))
        except Exception:
            continue
    if not pil_images:
        return images[0]

    target_h = 768
    resized: list[PILImage.Image] = []
    for im in pil_images:
        w, h = im.size
        if h <= 0:
            continue
        new_w = max(1, int(w * (target_h / h)))
        resized.append(im.resize((new_w, target_h)))
    if not resized:
        return images[0]

    total_w = sum(im.size[0] for im in resized)
    canvas = PILImage.new("RGB", (total_w, target_h), color=(0, 0, 0))
    x = 0
    for im in resized:
        canvas.paste(im, (x, 0))
        x += im.size[0]

    out = io.BytesIO()
    canvas.save(out, format="JPEG", quality=90)
    return out.getvalue()


class OpenAICompatBackend(ImageBackend):
    """HTTP port of the OpenAI-compatible images backend."""

    _COOLDOWN_SECONDS = 600

    def __init__(self, settings: ProviderSettings, *, policy: URLFetchPolicy | None = None):
        self.settings = settings
        self.provider_id = settings.id
        self.supports_edit = settings.supports_edit
        self.base_url = normalize_openai_compat_base_url(settings.base_url)
        self.api_keys = [
            str(k).strip() for k in (settings.api_keys or []) if str(k).strip()
        ]
        self.timeout = int(settings.timeout or 600)
        self.max_retries = settings.max_retries
        self.default_model = str(settings.model or "").strip()
        self.default_size = normalize_size_text(str(settings.default_size or "1024x1024"))
        self.extra_body = dict(settings.extra_body or {})
        self.output_format = str(settings.output_format or "jpeg").strip().lower()
        self.proxy_url = str(settings.proxy_url or "").strip() or None
        self.allowed_sizes = [
            normalize_size_text(s)
            for s in (settings.allowed_sizes or [])
            if normalize_size_text(s)
        ]
        self._ratio_defaults = (
            ratio_defaults_from_sizes(self.allowed_sizes) if self.allowed_sizes else {}
        )
        if settings.ratio_default_sizes and self.allowed_sizes:
            for ratio, size in settings.ratio_default_sizes.items():
                r = str(ratio or "").strip()
                s = normalize_size_text(size)
                if r and s and s in self.allowed_sizes:
                    self._ratio_defaults[r] = s

        self._policy = policy or URLFetchPolicy()
        self._key_index = 0
        self._generate_disabled_until = 0.0
        self._edit_disabled_until = 0.0
        self.async_mode = str(settings.async_mode or "auto").strip().lower()
        self._session: aiohttp.ClientSession | None = None

    # -- session ------------------------------------------------------------

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=max(1, int(self.timeout)))
            connector = aiohttp.TCPConnector(limit=8, limit_per_host=4)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # -- helpers ------------------------------------------------------------

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("provider has no API key configured")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    @staticmethod
    def _supports_multiple_images(model: str) -> bool:
        return model.startswith("gpt-image-") or model == "chatgpt-image-latest"

    def _is_generate_disabled(self) -> bool:
        return time.time() < self._generate_disabled_until

    def _is_edit_disabled(self) -> bool:
        return time.time() < self._edit_disabled_until

    @staticmethod
    def _is_invalid_size_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "size" not in msg:
            return False
        return any(
            key in msg
            for key in (
                "invalid",
                "unsupported",
                "not supported",
                "allowed",
                "must be",
                "not one of",
            )
        )

    def _resolve_size(
        self,
        size: str | None,
        resolution: str | None,
        size_hint: str = "",
    ) -> tuple[str, str, bool]:
        raw = normalize_size_text(size) or normalize_size_text(size_hint)
        if raw and not _looks_like_size(raw):
            raw = normalize_size_text(resolution_to_size(raw))
        if not raw:
            raw = normalize_size_text(resolution_to_size(resolution or ""))
        if not raw:
            raw = self.default_size

        if not self.allowed_sizes:
            return raw, raw, False

        if raw in self.allowed_sizes:
            return raw, raw, False

        requested_ratio = size_to_ratio(raw)
        fallback = ""
        if requested_ratio:
            default_ratio = size_to_ratio(self.default_size)
            if (
                self.default_size in self.allowed_sizes
                and default_ratio == requested_ratio
            ):
                fallback = self.default_size
            else:
                fallback = self._ratio_defaults.get(requested_ratio, "")

        if not fallback and self.default_size in self.allowed_sizes:
            fallback = self.default_size
        if not fallback and self.allowed_sizes:
            fallback = self.allowed_sizes[0]

        return fallback or raw, raw, True

    def resolve_output_intent(self, intent: OutputIntent) -> dict[str, str]:
        if is_gpt_image_2_model(self.default_model):
            selected = resolve_gpt_image_2_size(intent, default_size=self.default_size)
            if selected:
                return {"size": selected}
        if intent.exact_size:
            return {"size": intent.exact_size}
        if self.allowed_sizes:
            selected = select_allowed_size(
                intent,
                self.allowed_sizes,
                default_size=self.default_size,
            )
            if selected:
                return {"size": selected}
        if intent.aspect_ratio:
            selected = size_from_ratio_and_resolution(
                intent.aspect_ratio,
                intent.resolution,
                default_size=self.default_size,
            )
            if selected:
                return {"size": selected}
        return to_request_size_args(intent)

    async def _post_json(
        self,
        url: str,
        key: str,
        payload: dict,
        *,
        retries: int = 0,
    ) -> dict:
        session = await self._session_get()
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                proxy=self.proxy_url or None,
            ) as resp:
                body = await resp.read()
                if resp.status in range(200, 300):
                    if not body:
                        return {}
                    return json.loads(body)
                detail = body[:300].decode("utf-8", errors="replace")
                error = RuntimeError(f"HTTP {resp.status}: {detail}")
                if resp.status in {408, 429, 500, 502, 503, 504} and attempt < retries:
                    retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                    delay = retry_after if retry_after is not None else backoff_delay(attempt)
                    attempt += 1
                    logger.warning(
                        "图片接口 HTTP %s，第 %d/%d 次重试，等 %.1f 秒，响应 %dB",
                        resp.status, attempt, retries, delay, len(body),
                    )
                    await asyncio_sleep(delay)
                    continue
                logger.warning("图片接口 HTTP %s 最终失败，响应 %dB", resp.status, len(body))
                raise error

    async def _post_form(
        self,
        url: str,
        key: str,
        fields: dict,
        files: list[tuple[str, str, bytes, str]],
        *,
        retries: int = 0,
    ) -> dict:
        session = await self._session_get()
        headers = {"Authorization": f"Bearer {key}"}
        form = aiohttp.FormData()
        for name, value in fields.items():
            form.add_field(name, str(value))
        for name, filename, data, mime in files:
            form.add_field(name, data, filename=filename, content_type=mime or "application/octet-stream")
        attempt = 0
        while True:
            async with session.post(
                url,
                headers=headers,
                data=form,
                proxy=self.proxy_url or None,
            ) as resp:
                body = await resp.read()
                if resp.status in range(200, 300):
                    if not body:
                        return {}
                    return json.loads(body)
                detail = body[:300].decode("utf-8", errors="replace")
                error = RuntimeError(f"HTTP {resp.status}: {detail}")
                if resp.status in {408, 429, 500, 502, 503, 504} and attempt < retries:
                    retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                    delay = retry_after if retry_after is not None else backoff_delay(attempt)
                    attempt += 1
                    logger.warning(
                        "图片接口 HTTP %s，第 %d/%d 次重试，等 %.1f 秒，响应 %dB",
                        resp.status, attempt, retries, delay, len(body),
                    )
                    await asyncio_sleep(delay)
                    continue
                logger.warning("图片接口 HTTP %s 最终失败，响应 %dB", resp.status, len(body))
                raise error

    async def _download(self, url: str) -> bytes:
        await ensure_url_allowed(url, self._policy)
        session = await self._session_get()
        async with session.get(url, proxy=self.proxy_url or None, allow_redirects=False) as resp:
            if resp.status != 200:
                raise RuntimeError(f"image download failed: HTTP {resp.status}")
            data = await resp.read()
        if len(data) > 24 * 1024 * 1024:
            raise ValueError("image exceeds 24 MB")
        validate_image_bytes(data)
        return data

    async def _extract_image(self, resp: dict, model: str, size: str) -> ImageResult:
        item = self._pick_item(resp)
        url = item.get("url") or item.get("image_url")
        b64 = item.get("b64_json")
        raw: bytes | None = None
        # Prefer inline bytes: gateways sometimes return a signed URL that points
        # at an internal host, which the download policy must reject.
        if b64:
            try:
                candidate = base64.b64decode(str(b64))
                validate_image_bytes(candidate)
                raw = candidate
            except Exception:
                raw = None
        if raw is None and url:
            raw = await self._download(str(url))
        if raw is None:
            raise RuntimeError("provider response contains no image")
        mime, ext = guess_image_mime_and_ext(raw)
        return ImageResult(data=raw, mime=mime, ext=ext, size=size, model=model)

    # -- async tasks --------------------------------------------------------

    @staticmethod
    def _pick_item(resp: dict) -> dict:
        data = resp.get("data")
        if isinstance(data, dict):
            return data
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return data[0]
        return {}

    @staticmethod
    def _task_id(resp: dict) -> str:
        item = OpenAICompatBackend._pick_item(resp)
        for holder in (resp, item):
            for key in ("task_id", "id"):
                value = holder.get(key)
                if value:
                    text = str(value).strip()
                    if text.startswith(("api_", "task", "img", "gen")):
                        return text
        return ""

    def _poll_url(self, resp: dict, task_id: str) -> str:
        item = self._pick_item(resp)
        for holder in (resp, item):
            value = str(holder.get("poll_url") or "").strip()
            if value:
                return value
        base = root_url(self.base_url)
        return f"{base}/api/image-tasks?ids={task_id}"

    async def _await_task(self, resp: dict, key: str, model: str, size: str) -> ImageResult:
        task_id = self._task_id(resp)
        if not task_id:
            raise RuntimeError("provider queued the image but returned no task id")
        url = self._poll_url(resp, task_id)
        interval = float(self.extra_body.get("poll_interval") or 5)
        deadline = time.monotonic() + max(60, int(self.settings.timeout or 600))
        session = await self._session_get()
        headers = {"Authorization": f"Bearer {key}"}
        failures = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"image task {task_id} timed out")
            await asyncio_sleep(min(max(1.0, interval), remaining))
            try:
                async with session.get(
                    url, headers=headers, proxy=self.proxy_url or None
                ) as poll:
                    body = await poll.read()
                    if poll.status != 200:
                        raise RuntimeError(f"HTTP {poll.status}: {body[:200].decode('utf-8', 'replace')}")
                    data = json.loads(body)
            except Exception as exc:
                failures += 1
                if failures > 3:
                    raise RuntimeError(f"polling task {task_id} failed: {exc}") from exc
                continue
            failures = 0
            state = self._task_state(data)
            if state in ("failed", "error", "cancelled", "canceled"):
                raise RuntimeError(f"image task {task_id} {state}")
            if state in ("success", "succeeded", "completed", "done"):
                item = self._task_result(data)
                return await self._extract_image({"data": [item]}, model, size)

    @staticmethod
    def _task_items(data: dict) -> list[dict]:
        for key in ("items", "tasks", "data"):
            value = data.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [data] if isinstance(data, dict) else []

    @classmethod
    def _task_state(cls, data: dict) -> str:
        for item in cls._task_items(data):
            for key in ("status", "state"):
                value = str(item.get(key) or "").strip().lower()
                if value:
                    return value
        return ""

    @classmethod
    def _task_result(cls, data: dict) -> dict:
        for item in cls._task_items(data):
            results = item.get("results") or item.get("images") or item.get("outputs")
            if isinstance(results, list):
                for entry in results:
                    if isinstance(entry, dict) and (entry.get("url") or entry.get("b64_json")):
                        return entry
            if item.get("url") or item.get("b64_json") or item.get("image_url"):
                return item
            for key in ("result", "output", "data"):
                value = item.get(key)
                if isinstance(value, dict) and (value.get("url") or value.get("b64_json")):
                    return value
                if isinstance(value, list):
                    for entry in value:
                        if isinstance(entry, dict) and (entry.get("url") or entry.get("b64_json")):
                            return entry
        raise RuntimeError("image task finished without an image")

    # -- generate -----------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        intent: OutputIntent | None = None,
        size_hint: str = "",
        model: str = "",
        extra: dict | None = None,
    ) -> list[ImageResult]:
        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("provider has no model configured")

        resolved = self.resolve_output_intent(intent or OutputIntent())
        final_size, raw_size, fallback_used = self._resolve_size(
            resolved.get("size"),
            resolved.get("resolution"),
            size_hint=size_hint,
        )

        url = f"{self.base_url}/images/generations"
        payload = {"model": final_model, "prompt": prompt, "size": final_size}
        mode = str(self.async_mode or "auto").strip().lower()
        if mode == "on" or (mode == "auto" and self.extra_body.get("async")):
            payload["async"] = True
        payload.update(self.extra_body)
        payload.update(extra or {})

        key = self._next_key()
        if self._is_generate_disabled():
            raise RuntimeError(
                "provider images.generate temporarily disabled after a 404 cooldown"
            )
        try:
            resp = await self._post_json(url, key, payload, retries=self.max_retries)
        except Exception as exc:
            message = str(exc)
            if "404" in message:
                self._generate_disabled_until = time.time() + self._COOLDOWN_SECONDS
                raise RuntimeError(
                    "404: base_url may be wrong or the service does not expose "
                    "an OpenAI-compatible /v1/images endpoint"
                ) from exc
            if "response_format" not in payload and ("HTTP 400" in message or "HTTP 422" in message):
                # Some gateways refuse without an explicit inline-image request.
                payload["response_format"] = "b64_json"
                resp = await self._post_json(url, key, payload, retries=self.max_retries)
            elif (
                final_size == "4096x4096"
                and not self._supports_multiple_images(final_model)
                and self._is_invalid_size_error(exc)
            ):
                payload["size"] = "2048x2048"
                resp = await self._post_json(url, key, payload, retries=self.max_retries)
                final_size = "2048x2048"
            else:
                raise
        self._generate_disabled_until = 0.0
        if self._task_id(resp):
            result = await self._await_task(resp, key, final_model, final_size)
            return [result]
        result = await self._extract_image(resp, final_model, final_size)
        return [result]

    # -- edit ---------------------------------------------------------------

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

        final_model = str(model or self.default_model or "").strip()
        if not final_model:
            raise RuntimeError("provider has no model configured")

        resolved = self.resolve_output_intent(intent or OutputIntent())
        final_size, raw_size, fallback_used = self._resolve_size(
            resolved.get("size"),
            resolved.get("resolution"),
            size_hint=size_hint,
        )

        image_bytes: list[bytes] = []
        for path in images:
            with open(path, "rb") as fh:
                image_bytes.append(fh.read())

        files: list[tuple[str, str, bytes, str]] = []
        if self._supports_multiple_images(final_model) and len(image_bytes) > 1:
            for index, data in enumerate(image_bytes, start=1):
                _, ext = guess_image_mime_and_ext(data)
                files.append((f"reference_{index}.{ext}", f"reference_{index}.{ext}", data, _mime_of(data)))
        else:
            packed = _build_collage(image_bytes) if len(image_bytes) > 1 else image_bytes[0]
            _, ext = guess_image_mime_and_ext(packed)
            files.append(("image", f"input.{ext}", packed, _mime_of(packed)))

        url = f"{self.base_url}/images/edits"
        fields = {"model": final_model, "prompt": prompt, "size": final_size}
        mode = str(self.async_mode or "auto").strip().lower()
        if mode == "on" or (mode == "auto" and self.extra_body.get("async")):
            fields["async"] = "true"
        fields.update({str(k): str(v) for k, v in self.extra_body.items()})
        fields.update({str(k): str(v) for k, v in (extra or {}).items()})

        key = self._next_key()
        if self._is_edit_disabled():
            raise RuntimeError(
                "provider images.edit temporarily disabled after a 404 cooldown"
            )
        try:
            resp = await self._post_form(url, key, fields, files, retries=self.max_retries)
        except Exception as exc:
            message = str(exc)
            if "404" in message:
                self._edit_disabled_until = time.time() + self._COOLDOWN_SECONDS
                raise RuntimeError(
                    "404: the service does not expose an images.edit endpoint"
                ) from exc
            if "response_format" not in fields and ("HTTP 400" in message or "HTTP 422" in message):
                fields["response_format"] = "b64_json"
                resp = await self._post_form(url, key, fields, files, retries=self.max_retries)
            elif (
                final_size == "4096x4096"
                and not self._supports_multiple_images(final_model)
                and self._is_invalid_size_error(exc)
            ):
                fields["size"] = "2048x2048"
                resp = await self._post_form(url, key, fields, files, retries=self.max_retries)
                final_size = "2048x2048"
            else:
                raise
        self._edit_disabled_until = 0.0
        if self._task_id(resp):
            result = await self._await_task(resp, key, final_model, final_size)
            return [result]
        result = await self._extract_image(resp, final_model, final_size)
        return [result]


def _mime_of(data: bytes) -> str:
    mime, _ = guess_image_mime_and_ext(data)
    return mime


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
