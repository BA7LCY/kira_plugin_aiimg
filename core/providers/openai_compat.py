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

from ..async_task import (
    TaskSpec,
    as_text,
    await_task,
    build_poll_url,
    detect_task_id,
    first_value,
    has_inline_image,
    spec_from_config,
    task_state,
)
from ..http import backoff_delay, current_session, retry_after_seconds
from ..logbook import get_logger
from ..image_format import (
    guess_image_mime_and_ext,
    is_placeholder_image,
    log_size_mismatch,
    looks_like_svg,
    validate_image_bytes,
    validate_svg_bytes,
)
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
from .base import (
    FormatMismatchError,
    ImageBackend,
    ImageResult,
    KeyPoolMixin,
    ProviderSettings,
)

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


class OpenAICompatBackend(KeyPoolMixin, ImageBackend):
    """HTTP port of the OpenAI-compatible images backend."""

    _COOLDOWN_SECONDS = 600

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
        self._init_key_pool(settings)
        self._generate_disabled_until = 0.0
        self._edit_disabled_until = 0.0
        self.async_mode = str(settings.async_mode or "auto").strip().lower()

    # -- session ------------------------------------------------------------

    def _request_timeout(self) -> aiohttp.ClientTimeout:
        """单次请求超时：沿用插件配置的总超时，不依赖共享会话的默认值。"""
        return aiohttp.ClientTimeout(total=max(1, int(self.timeout)))

    async def _session_get(self) -> aiohttp.ClientSession:
        """复用进程内共享会话，不再为每个服务商单独建连接池。"""
        logger.debug("图片接口复用共享会话")
        return await current_session()

    async def close(self) -> None:
        """共享会话由插件统一关闭，服务商这里只保留空实现。"""
        logger.debug("图片接口关闭：共享会话由插件统一管理，此处不关闭")

    # -- helpers ------------------------------------------------------------

    def _key_pool(self) -> list[str]:
        return self.api_keys

    def _key_logger(self):
        return logger

    def _next_key(self) -> str:
        return self._pick_key()

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
                timeout=self._request_timeout(),
            ) as resp:
                body = await resp.read()
                if resp.status in range(200, 300):
                    if not body:
                        return {}
                    return json.loads(body)
                self._handle_key_status(key, resp.status)
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
                timeout=self._request_timeout(),
            ) as resp:
                body = await resp.read()
                if resp.status in range(200, 300):
                    if not body:
                        return {}
                    return json.loads(body)
                self._handle_key_status(key, resp.status)
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
        return data

    async def _extract_images(
        self, resp: dict, model: str, size: str, expected_format: str = ""
    ) -> list[ImageResult]:
        """Decode every image in a response, in the order the provider sent them."""
        items = self._pick_items(resp)
        if not items:
            raise RuntimeError("provider response contains no image")
        results: list[ImageResult] = []
        for item in items:
            results.append(await self._extract_item(item, model, size, expected_format))
        return results

    async def _extract_image(self, resp: dict, model: str, size: str) -> ImageResult:
        results = await self._extract_images(resp, model, size)
        return results[0]

    async def _extract_item(
        self, item: dict, model: str, size: str, expected_format: str = ""
    ) -> ImageResult:
        url = item.get("url") or item.get("image_url")
        b64 = item.get("b64_json")
        raw: bytes | None = None
        # Prefer inline bytes: gateways sometimes return a signed URL that points
        # at an internal host, which the download policy must reject.
        if b64:
            try:
                raw = base64.b64decode(str(b64))
            except Exception:
                raw = None
        if raw is None and url:
            raw = await self._download(str(url))
        if raw is None:
            raise RuntimeError("provider response contains no image")

        expected = str(expected_format or "").strip().lower().replace("jpg", "jpeg")
        cap = int(getattr(self, "_image_max_bytes", 0) or 0) or (24 * 1024 * 1024)
        is_svg = looks_like_svg(raw)
        if expected == "svg":
            if not is_svg:
                raise FormatMismatchError("服务商没有返回 SVG")
            validate_svg_bytes(raw, cap)
        elif is_svg:
            validate_svg_bytes(raw, cap)
            if expected:
                raise FormatMismatchError("服务商没有返回请求的位图格式")
        else:
            validate_image_bytes(raw, cap)
            # GIF 是明确要动的图，静态图不能冒充；PNG/JPEG/WebP
            # 之间不转码，服务商给哪个就存哪个、发哪个。
            if expected == "gif":
                mime, _ext = guess_image_mime_and_ext(raw)
                if mime != "image/gif":
                    raise FormatMismatchError(
                        f"服务商没有返回 GIF（实际是 {mime}）"
                    )

        if not is_svg and is_placeholder_image(raw):
            logger.warning("[OpenAICompat] 服务商返回了占位图（%dB），丢弃", len(raw))
            raise RuntimeError("服务商返回了占位图，无法使用")
        if not is_svg:
            log_size_mismatch(logger, raw, requested_size=size, source="OpenAICompat")
        mime, ext = guess_image_mime_and_ext(raw)
        return ImageResult(data=raw, mime=mime, ext=ext, size=size, model=model)

    # -- async tasks --------------------------------------------------------

    @staticmethod
    def _pick_items(resp: dict) -> list[dict]:
        data = resp.get("data")
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []

    @staticmethod
    def _pick_item(resp: dict) -> dict:
        items = OpenAICompatBackend._pick_items(resp)
        return items[0] if items else {}

    def _task_spec(self) -> TaskSpec:
        """Polling dialect for this provider, built once per backend."""
        cached = getattr(self, "_cached_task_spec", None)
        if cached is not None:
            return cached
        raw = getattr(self.settings, "async_task", None) or self.extra_body.get("async_task")
        spec = spec_from_config(
            raw,
            base_interval=float(self.extra_body.get("poll_interval") or 5),
            base_timeout=max(60, int(self.settings.timeout or 600)),
        )
        self._cached_task_spec = spec
        return spec

    def _task_id(self, resp: dict) -> str:
        spec = self._task_spec()
        # A response that already carries the picture is done, even when the
        # gateway also echoes a request id.
        if has_inline_image(resp):
            return ""
        task_id = detect_task_id(resp, spec)
        if not task_id:
            return ""
        # A bare "id" is not proof of a queued job: an error body can carry one
        # too. Treat the response as asynchronous only when something else
        # points that way — a configured dialect, a reported state, a poll
        # address, an explicit task field, or the id naming itself a task.
        if spec.configured or task_state(resp, spec) or self._response_poll_url(resp):
            return task_id
        if _has_task_field(resp):
            return task_id
        if task_id.startswith(_TASK_ID_PREFIXES):
            return task_id
        return ""

    @staticmethod
    def _response_poll_url(resp: dict) -> str:
        item = OpenAICompatBackend._pick_item(resp)
        for holder in (resp, item):
            value = str(holder.get("poll_url") or "").strip()
            if value:
                return value
        return ""

    def _poll_url(self, resp: dict, task_id: str) -> str:
        explicit = self._response_poll_url(resp)
        if explicit:
            return explicit
        return build_poll_url(self._task_spec(), self.base_url, task_id)

    async def _await_task(
        self, resp: dict, key: str, model: str, size: str, expected_format: str = ""
    ) -> ImageResult:
        spec = self._task_spec()
        task_id = self._task_id(resp)
        if not task_id:
            raise RuntimeError("provider queued the image but returned no task id")
        url = self._poll_url(resp, task_id)
        session = await self._session_get()
        headers = {"Authorization": f"Bearer {key}"}

        async def fetch(target: str) -> tuple[int, bytes]:
            async with session.get(
                target,
                headers=headers,
                proxy=self.proxy_url or None,
                timeout=self._request_timeout(),
            ) as poll:
                body = await poll.read()
                return poll.status, body

        def on_status(_task_id: str, status: int) -> None:
            self._handle_key_status(key, status)

        item = await await_task(
            fetch=fetch,
            spec=spec,
            task_id=task_id,
            poll_url=url,
            sleep=asyncio_sleep,
            clock=time.monotonic,
            on_status=on_status,
            logger=logger,
        )
        results = await self._extract_images(
            {"data": [item]}, model, size, expected_format
        )
        return results[0]

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
        expected_format = str((extra or {}).get("output_format") or "").strip().lower()
        if self._task_id(resp):
            result = await self._await_task(resp, key, final_model, final_size, expected_format)
            return [result]
        return await self._extract_images(resp, final_model, final_size, expected_format)

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

        # images/edits 的图片与提示词是各自独立的表单字段（图片走文件字段，
        # 提示词走 prompt 文本字段），客户端按字段名传参，不存在「先图后文」
        # 的先后顺序问题。这里只需要保证一次请求里图片字段和 prompt 字段都在。
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
        # 附加参数可能带上同名字段，最后再确认一次文字指令在场，避免被挤掉。
        fields["prompt"] = prompt

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
        expected_format = str((extra or {}).get("output_format") or "").strip().lower()
        if self._task_id(resp):
            result = await self._await_task(resp, key, final_model, final_size, expected_format)
            return [result]
        return await self._extract_images(resp, final_model, final_size, expected_format)


def _mime_of(data: bytes) -> str:
    mime, _ = guess_image_mime_and_ext(data)
    return mime


_TASK_ID_PREFIXES = ("api_", "task", "img", "gen")


def _has_task_field(payload: dict) -> bool:
    """True when the body names a task explicitly rather than a generic id."""
    item = OpenAICompatBackend._pick_item(payload)
    for holder in (payload, item):
        for key in ("task_id", "taskId", "task"):
            if str(holder.get(key) or "").strip():
                return True
    return False


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
