"""OpenAI chat-completions image backend (generate + edit), HTTP implementation.

Targets gateway-style providers that return images through chat completions
content (markdown `![](...)`, `data:image/...`, or tool calls). Plain aiohttp
handles JSON and SSE. No hidden prompt suffixes are used: the visible
per-provider `prompt_template` (default `{{prompt}}`) composes the user
message, so every added word can be seen and edited in the settings page.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp

from ..http import backoff_delay, retry_after_seconds
from ..logbook import get_logger
from ..image_format import (
    decode_base64_image,
    guess_image_mime_and_ext,
    image_data_url,
    validate_image_bytes,
)
from ..net import URLFetchPolicy, ensure_url_allowed
from ..output_spec import OutputIntent, is_gpt_image_2_model
from ..prompt import render_template
from ..sizes import normalize_size_text
from .base import ImageBackend, ImageResult, ProviderSettings
from .openai_compat import OpenAICompatBackend, normalize_openai_compat_base_url

_MARKDOWN_IMAGE_RE = re.compile(r"!\[.*?\]\((.*?)\)")
_DATA_IMAGE_RE = re.compile(r"(data:image/[^\s)]+)")
_HTML_IMG_RE = re.compile(r'<img[^>]*src=["\']([^"\'>]+)["\']', re.IGNORECASE)
_IMAGE_URL_RE = re.compile(
    r"(https?://[^\s<>\"')\]]+?\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s<>\"')\]]*)?)",
    re.IGNORECASE,
)
_JSON_URL_FIELD_RE = re.compile(
    r'"(?:image_url|imageUrl|url|image|src|uri|link|href|fifeUrl|fife_url|final_image_url|origin_image_url)"\s*:\s*"([^"]+)"'
)
_BASE64_PREFIX_RE = re.compile(r"^(?:b64|base64)\s*:\s*", re.IGNORECASE)
_LOCAL_MEDIA_HOSTS = {"0.0.0.0", "127.0.0.1", "localhost", "host.docker.internal"}
_REQUEST_MODE_ALIASES = {
    "auto": "auto",
    "stream": "stream",
    "non_stream": "non_stream",
    "non-stream": "non_stream",
    "nonstream": "non_stream",
    "non stream": "non_stream",
}


def _normalize_request_mode(value: object) -> str:
    if isinstance(value, bool):
        return "stream" if value else "non_stream"
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return _REQUEST_MODE_ALIASES.get(text, "")


def _looks_like_video_url(url: str) -> bool:
    u = (url or "").strip().lower()
    if not u.startswith(("http://", "https://")):
        return False
    if any(ext in u for ext in (".mp4", ".webm", ".mov")):
        return True
    return "generated_video" in u


def _looks_like_relative_media_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s or s.startswith(("data:image/", "http://", "https://", "file://")):
        return False
    if "://" in s:
        return False
    lowered = s.lower()
    return any(ext in lowered for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif"))


def _is_local_media_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    if not h:
        return False
    if h in _LOCAL_MEDIA_HOSTS or h.endswith(".localhost") or h.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


def _rewrite_local_media_url(ref: str, *, base_url: str) -> str:
    s = str(ref or "").strip()
    if not s:
        return ""
    try:
        parts = urlsplit(s)
    except Exception:
        return s
    if not parts.scheme or not parts.netloc or not _is_local_media_host(parts.hostname or ""):
        return s
    base_parts = urlsplit(str(base_url or "").strip())
    if not base_parts.hostname:
        return s
    host = base_parts.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port is not None else host
    return urlunsplit(
        (
            parts.scheme or base_parts.scheme or "http",
            netloc,
            parts.path or "/",
            parts.query,
            parts.fragment,
        )
    )


def _decode_base64_bytes(text: str) -> bytes:
    s = re.sub(r"\s+", "", str(text or "").strip())
    if not s:
        return b""
    candidates = [s, s.replace("-", "+").replace("_", "/")]
    for cand in candidates:
        pad = "=" * ((4 - len(cand) % 4) % 4)
        try:
            raw = base64.b64decode(cand + pad, validate=False)
            if raw:
                return raw
        except Exception:
            continue
    try:
        raw = base64.urlsafe_b64decode(s + ("=" * ((4 - len(s) % 4) % 4)))
        if raw:
            return raw
    except Exception:
        pass
    return b""


def _guess_mime_from_magic(image_bytes: bytes) -> str | None:
    if len(image_bytes) >= 3 and image_bytes[0:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(image_bytes) >= 8 and image_bytes[0:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(image_bytes) >= 6 and (image_bytes[0:6] == b"GIF87a" or image_bytes[0:6] == b"GIF89a"):
        return "image/gif"
    if (
        len(image_bytes) >= 12
        and image_bytes[0:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def _is_valid_data_image_ref(ref: str) -> bool:
    s = str(ref or "").strip()
    if not s.startswith("data:image/"):
        return False
    if "," not in s:
        return False
    _header, b64 = s.split(",", 1)
    b64 = re.sub(r"\s+", "", (b64 or "").strip())
    if not b64 or b64 == "...":
        return False
    if len(b64) < 16:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", b64[:2048]):
        return False
    if len(b64) < 128:
        raw = _decode_base64_bytes(b64)
        if not raw or not _guess_mime_from_magic(raw):
            return False
    return True


def _base64_to_data_image_ref(text: str) -> str | None:
    s = (text or "").strip().strip('"').strip("'")
    s = _BASE64_PREFIX_RE.sub("", s).strip()
    s = re.sub(r"\s+", "", s)
    if len(s) < 128:
        return None
    raw = _decode_base64_bytes(s)
    if not raw:
        return None
    mime = _guess_mime_from_magic(raw)
    if not mime:
        return None
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def _strip_markdown_target(target: str) -> str | None:
    s = (target or "").strip()
    if not s:
        return None
    if s.startswith("<") and ">" in s:
        right = s.find(">")
        if right > 1:
            s = s[1:right].strip()
    m = re.match(r"^(?P<url>\S+)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*$", s)
    if m:
        s = m.group("url")
    s = s.strip().strip('"').strip("'")
    return s or None


def _iter_strings(obj: object) -> list[str]:
    out: list[str] = []
    seen: set[int] = set()

    def walk(x: object) -> None:
        if x is None:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(x, str):
            out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
            return
        if isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
            return

    walk(obj)
    return out


def _extract_first_image_ref(text: str) -> str | None:
    s = (text or "").strip()
    if not s:
        return None
    if s.startswith("data:image/"):
        compact = re.sub(r"\s+", "", s)
        if _is_valid_data_image_ref(compact):
            return compact
    m = _MARKDOWN_IMAGE_RE.search(s)
    if m:
        cand = _strip_markdown_target(m.group(1))
        if cand:
            if cand.startswith("data:image/"):
                cand = re.sub(r"\s+", "", cand)
                if _is_valid_data_image_ref(cand):
                    return cand
            elif not _looks_like_video_url(cand):
                return cand
    for m in _DATA_IMAGE_RE.finditer(s):
        cand = re.sub(r"\s+", "", m.group(1).strip())
        if _is_valid_data_image_ref(cand):
            return cand
    m = _HTML_IMG_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    m = _IMAGE_URL_RE.search(s)
    if m:
        url = m.group(1).strip()
        if url and not _looks_like_video_url(url):
            return url
    if s.startswith("http://") or s.startswith("https://"):
        return None if _looks_like_video_url(s) else s
    if s.startswith("//"):
        return s
    if s.startswith("/") and _looks_like_relative_media_ref(s):
        return s
    if _looks_like_relative_media_ref(s):
        return s
    for m in _JSON_URL_FIELD_RE.finditer(s):
        cand = m.group(1).strip().replace("\\/", "/")
        cand = _strip_markdown_target(cand) or cand
        if cand.startswith("data:image/"):
            cand = re.sub(r"\s+", "", cand)
            if _is_valid_data_image_ref(cand):
                return cand
        if cand.startswith(("http://", "https://")) and not _looks_like_video_url(cand):
            return cand
        if cand.startswith("//"):
            return cand
        if cand.startswith("/") and _looks_like_relative_media_ref(cand):
            return cand
        if _looks_like_relative_media_ref(cand):
            return cand
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            parsed = json.loads(s)
        except Exception:
            parsed = None
        if parsed is not None:
            for v in _iter_strings(parsed):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref
    ref = _base64_to_data_image_ref(s)
    return ref or None


def _extract_image_ref_from_content(content: object) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return _extract_first_image_ref(content)
    if isinstance(content, list):
        for part in content:
            ref = _extract_image_ref_from_content(part)
            if ref:
                return ref
        return None
    if isinstance(content, dict):
        if str(content.get("type") or "").lower() == "image_url":
            image_url = content.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.strip():
                    return url.strip()
            if isinstance(image_url, str) and image_url.strip():
                return image_url.strip()
        if str(content.get("type") or "").lower() == "text":
            text = content.get("text")
            if isinstance(text, str):
                ref = _extract_first_image_ref(text)
                if ref:
                    return ref
        for k in ("b64_json", "b64", "base64", "image_b64", "image_base64", "imageB64"):
            v = content.get(k)
            if isinstance(v, str):
                ref = _base64_to_data_image_ref(v)
                if ref:
                    return ref
        inline = content.get("inlineData")
        if isinstance(inline, dict):
            b64 = inline.get("data")
            if isinstance(b64, str):
                ref = _base64_to_data_image_ref(b64)
                if ref:
                    return ref
        for k in (
            "url",
            "image",
            "image_url",
            "data",
            "src",
            "uri",
            "link",
            "href",
            "final_image_url",
            "origin_image_url",
            "fifeUrl",
            "fife_url",
            "thumbnail",
        ):
            v = content.get(k)
            if isinstance(v, str):
                ref = _extract_first_image_ref(v)
                if ref:
                    return ref
            ref = _extract_image_ref_from_content(v)
            if ref:
                return ref
        for k in ("images", "image_urls", "attachments", "media", "result", "response"):
            ref = _extract_image_ref_from_content(content.get(k))
            if ref:
                return ref
        for s in _iter_strings(content):
            ref = _extract_first_image_ref(s)
            if ref:
                return ref
        return None
    for s in _iter_strings(content):
        ref = _extract_first_image_ref(s)
        if ref:
            return ref
    return None


def _media_refs_from_sse_text(text: str) -> list[str]:
    image_refs: list[str] = []
    full_text = ""
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("```") or not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            obj = json.loads(data_str)
        except Exception:
            continue
        ref = _extract_image_ref_from_content(obj)
        if ref and ref not in image_refs:
            image_refs.append(ref)
        choice0 = (obj.get("choices") or [{}])[0] if isinstance(obj, dict) else {}
        if not isinstance(choice0, dict):
            choice0 = {}
        delta = choice0.get("delta") or {}
        message = choice0.get("message") or {}
        delta_content = delta.get("content") if "content" in delta else message.get("content")
        if delta_content is None and "reasoning_content" in delta:
            delta_content = delta.get("reasoning_content")
        if delta_content is None and "reasoning_content" in message:
            delta_content = message.get("reasoning_content")
        if isinstance(delta_content, str):
            full_text += delta_content
        elif isinstance(delta_content, list):
            for part in delta_content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    full_text += part["text"]
    final = _extract_first_image_ref(full_text)
    if final and final not in image_refs:
        image_refs.append(final)
    return image_refs


def _collect_refs_from_response(resp: object) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def add_ref(value: object) -> None:
        if not isinstance(value, str):
            return
        s = value.strip().strip('"').strip("'")
        if not s or s in seen:
            return
        if s.startswith(("http://", "https://")) and _looks_like_video_url(s):
            return
        seen.add(s)
        refs.append(s)

    def collect(content: object) -> None:
        if content is None:
            return
        if isinstance(content, str):
            direct = _extract_first_image_ref(content)
            if direct:
                add_ref(direct)
            add_ref(content)
            return
        for s in _iter_strings(content):
            direct = _extract_first_image_ref(s)
            if direct:
                add_ref(direct)
            add_ref(s)

    if isinstance(resp, dict):
        choices = resp.get("choices") or []
        if isinstance(choices, list):
            for choice in choices[:4]:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message") or {}
                if not isinstance(message, dict):
                    continue
                for field in ("images", "content", "tool_calls"):
                    if message.get(field) is not None:
                        collect(message[field])
    collect(resp)
    return refs


def _normalize_image_size_hint(*, model: str | None, size: str | None, resolution: str | None) -> str | None:
    raw = str(size or "").strip()
    if not raw:
        raw = str(resolution or "").strip()
    if not raw:
        model_name = str(model or "").lower()
        if "4k" in model_name:
            raw = "4K"
        elif "2k" in model_name:
            raw = "2K"
        elif "1k" in model_name:
            raw = "1K"
        elif "512px" in model_name:
            raw = "512px"
    if not raw:
        return None
    normalized = raw.strip().upper().replace("×", "X")
    size_map = {"1024X1024": "1K", "2048X2048": "2K", "4096X4096": "4K"}
    return size_map.get(normalized, raw.strip())


def _aspect_from_size(size: str | None) -> str | None:
    s = str(size or "").strip().lower().replace("×", "x")
    if "x" not in s:
        return None
    try:
        w, h = s.split("x", 1)
        w_i, h_i = int(w), int(h)
    except Exception:
        return None
    if w_i <= 0 or h_i <= 0:
        return None
    gcd = math.gcd(w_i, h_i)
    return f"{w_i // gcd}:{h_i // gcd}"


logger = get_logger("provider.chat")

class OpenAIChatImageBackend(ImageBackend):
    """chat.completions image backend over aiohttp."""

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
        self.extra_body = dict(settings.extra_body or {})
        self.proxy_url = str(settings.proxy_url or "").strip() or None
        self.output_format = str(settings.output_format or "jpeg").strip().lower()
        self.prompt_template = str(settings.prompt_template or "{{prompt}}")
        self.default_size = normalize_size_text(str(settings.default_size or "1024x1024"))
        self.generate_request_mode = _normalize_request_mode(settings.generate_request_mode)
        self.edit_request_mode = _normalize_request_mode(settings.edit_request_mode)
        if not self.generate_request_mode:
            self.generate_request_mode = "stream" if settings.enable_stream_generate else "auto"
        if not self.edit_request_mode:
            self.edit_request_mode = "stream" if settings.enable_stream_edit else "auto"
        self._policy = policy or URLFetchPolicy()
        self._key_index = 0
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

    def _next_key(self) -> str:
        if not self.api_keys:
            raise RuntimeError("provider has no API key configured")
        key = self.api_keys[self._key_index]
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        return key

    def _should_try_stream(self, operation: str) -> bool:
        mode = self.generate_request_mode if operation == "generate" else self.edit_request_mode
        if mode == "stream":
            return True
        if mode == "non_stream":
            return False
        return operation == "generate"

    def _chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _compose_user_message(self, prompt: str, *, size: str, resolution: str) -> str:
        return render_template(
            self.prompt_template,
            prompt=prompt,
            size=size or "",
            resolution=resolution or "",
        )

    def _apply_size_extras(
        self,
        extra_body: dict,
        *,
        model: str,
        size: str,
        resolution: str,
        aspect_ratio: str | None = None,
    ) -> dict:
        merged = dict(extra_body)
        model_name = str(model or "").lower()
        if "gemini" in model_name:
            image_size = _normalize_image_size_hint(model=model, size=size, resolution=resolution)
            if image_size:
                merged.setdefault("imageSize", image_size)
            ratio = aspect_ratio or _aspect_from_size(size or resolution)
            if ratio:
                merged.setdefault("aspect_ratio", ratio)
                merged.setdefault("aspectRatio", ratio)
                raw_ic = merged.get("image_config")
                ic = dict(raw_ic) if isinstance(raw_ic, dict) else {}
                ic.setdefault("aspect_ratio", ratio)
                ic.setdefault("aspectRatio", ratio)
                merged["image_config"] = ic
                raw_cc = merged.get("generationConfig")
                if isinstance(raw_cc, dict):
                    raw_ic_camel = raw_cc.get("imageConfig")
                    if isinstance(raw_ic_camel, dict):
                        raw_ic_camel.setdefault("aspectRatio", ratio)
        if is_gpt_image_2_model(model) and str(size or "").strip():
            merged["size"] = str(size).strip()
        return merged

    # -- requests -----------------------------------------------------------

    async def _post_chat(
        self,
        *,
        key: str,
        model: str,
        messages: list[dict],
        extra_body: dict,
        stream: bool,
    ) -> list[str]:
        session = await self._session_get()
        payload: dict = {"model": model, "messages": messages, "stream": stream}
        if extra_body:
            payload.update(extra_body)
            payload["stream"] = stream
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        refs: list[str] = []
        attempt = 0
        while True:
            async with session.post(
                self._chat_completions_url(),
                headers=headers,
                json=payload,
                proxy=self.proxy_url or None,
            ) as resp:
                body = await resp.read()
                if resp.status not in range(200, 300):
                    detail = body[:300].decode("utf-8", errors="replace")
                    error = RuntimeError(f"HTTP {resp.status}: {detail}")
                    if resp.status in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                        retry_after = retry_after_seconds(resp.headers.get("Retry-After"))
                        delay = (
                            retry_after
                            if retry_after is not None
                            else backoff_delay(attempt)
                        )
                        attempt += 1
                        logger.warning(
                            "对话接口 HTTP %s，第 %d/%d 次重试，等 %.1f 秒，响应 %dB",
                            resp.status, attempt, self.max_retries, delay, len(body),
                        )
                        await asyncio.sleep(delay)
                        continue
                    logger.warning("对话接口 HTTP %s 最终失败，响应 %dB", resp.status, len(body))
                    raise error
                ctype = resp.headers.get("Content-Type", "").lower()
                if stream and "text/event-stream" in ctype:
                    text = body.decode("utf-8", errors="ignore")
                    refs = _media_refs_from_sse_text(text)
                    logger.info(
                        "对话接口流式返回 %dB，解析出 %d 个媒体引用", len(body), len(refs)
                    )
                    return refs
                if not body:
                    return []
                try:
                    obj = json.loads(body)
                except Exception:
                    return []
                return _collect_refs_from_response(obj)

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

    def _absolutize_media_ref(self, ref: str) -> str:
        s = str(ref or "").strip()
        if not s:
            return ""
        if s.startswith(("data:image/", "http://", "https://")):
            return _rewrite_local_media_url(s, base_url=self.base_url)
        split = urlsplit(self.base_url)
        origin = f"{split.scheme}://{split.netloc}" if split.scheme and split.netloc else ""
        if s.startswith("//"):
            return f"{split.scheme}:{s}" if split.scheme else f"https:{s}"
        if s.startswith("/"):
            return urljoin(origin + "/", s.lstrip("/")) if origin else s
        if _looks_like_relative_media_ref(s):
            if origin:
                return urljoin(origin + "/", s.lstrip("/"))
            return urljoin(self.base_url.rstrip("/") + "/", s)
        return s

    async def _save_ref(self, ref: str, *, model: str, size: str) -> ImageResult:
        ref = self._absolutize_media_ref(ref)
        if ref.startswith("data:image/"):
            try:
                raw = decode_base64_image(ref)
            except Exception as exc:
                raise RuntimeError("chat returned an invalid data:image") from exc
            validate_image_bytes(raw)
        elif ref.startswith(("http://", "https://")):
            raw = await self._download(ref)
        else:
            raise RuntimeError(f"unsupported image reference: {ref[:80]!r}")
        mime, ext = guess_image_mime_and_ext(raw)
        return ImageResult(data=raw, mime=mime, ext=ext, size=size, model=model)

    async def _save_from_refs(self, refs: list[str], *, model: str, size: str) -> ImageResult:
        last_error: Exception | None = None
        for ref in refs:
            try:
                return await self._save_ref(ref, model=model, size=size)
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"chat image save failed: {last_error}")

    def resolve_output_intent(self, intent: OutputIntent) -> dict[str, str]:
        if is_gpt_image_2_model(self.default_model):
            from ..output_spec import resolve_gpt_image_2_size

            selected = resolve_gpt_image_2_size(intent)
            if selected:
                return {"size": selected}
        if intent.exact_size:
            return {"size": intent.exact_size, "resolution": intent.exact_size}
        result: dict[str, str] = {}
        if intent.resolution:
            result["resolution"] = intent.resolution
        if intent.aspect_ratio:
            result["aspect_ratio"] = intent.aspect_ratio
        return result

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

        intent = intent or OutputIntent()
        resolved = self.resolve_output_intent(intent)
        size = str(resolved.get("size") or size_hint or "").strip()
        resolution = str(resolved.get("resolution") or "").strip()
        aspect_ratio = resolved.get("aspect_ratio") or ""

        extra_body = dict(self.extra_body)
        extra_body.update(extra or {})
        extra_body = self._apply_size_extras(
            extra_body,
            model=final_model,
            size=size,
            resolution=resolution,
            aspect_ratio=aspect_ratio or None,
        )

        user_text = self._compose_user_message(prompt, size=size, resolution=resolution)
        key = self._next_key()

        refs: list[str] = []
        if self._should_try_stream("generate"):
            try:
                refs = await self._post_chat(
                    key=key,
                    model=final_model,
                    messages=[{"role": "user", "content": user_text}],
                    extra_body=extra_body,
                    stream=True,
                )
            except Exception:
                refs = []
        if not refs:
            refs = await self._post_chat(
                key=key,
                model=final_model,
                messages=[{"role": "user", "content": user_text}],
                extra_body=extra_body,
                stream=False,
            )
        if not refs:
            raise RuntimeError(
                "chat response contained no image (need markdown image, "
                "data:image, or an image URL)"
            )
        result = await self._save_from_refs(refs, model=final_model, size=size or self.default_size)
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

        intent = intent or OutputIntent()
        resolved = self.resolve_output_intent(intent)
        size = str(resolved.get("size") or size_hint or "").strip()
        resolution = str(resolved.get("resolution") or "").strip()
        aspect_ratio = resolved.get("aspect_ratio") or ""

        extra_body = dict(self.extra_body)
        extra_body.update(extra or {})
        extra_body = self._apply_size_extras(
            extra_body,
            model=final_model,
            size=size,
            resolution=resolution,
            aspect_ratio=aspect_ratio or None,
        )

        image_bytes: list[bytes] = []
        for path in images:
            image_bytes.append(path.read_bytes())

        def build_parts(text: str) -> list[dict]:
            parts: list[dict] = [{"type": "text", "text": text}]
            for data in image_bytes:
                mime, _ext = guess_image_mime_and_ext(data)
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(data, mime=mime)},
                    }
                )
            return parts

        user_text = self._compose_user_message(prompt, size=size, resolution=resolution)
        key = self._next_key()

        refs: list[str] = []
        stream_error: Exception | None = None
        if self._should_try_stream("edit"):
            try:
                refs = await self._post_chat(
                    key=key,
                    model=final_model,
                    messages=[{"role": "user", "content": build_parts(user_text)}],
                    extra_body=extra_body,
                    stream=True,
                )
            except Exception as exc:
                stream_error = exc
                refs = []
        if not refs:
            try:
                refs = await self._post_chat(
                    key=key,
                    model=final_model,
                    messages=[{"role": "user", "content": build_parts(user_text)}],
                    extra_body=extra_body,
                    stream=False,
                )
            except Exception as exc:
                if stream_error is not None:
                    raise RuntimeError(
                        f"chat edit failed: {exc} (stream also failed: {stream_error})"
                    ) from exc
                raise
        if not refs:
            raise RuntimeError(
                "chat edit response contained no image (need markdown image, "
                "data:image, or an image URL)"
            )
        result = await self._save_from_refs(refs, model=final_model, size=size or self.default_size)
        return [result]
