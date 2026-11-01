"""Agnes video generation: JSON create plus video_id polling.

One durable rate budget covers creation and polling, keyed by endpoint and
API key so restarts keep the same spacing.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote, urlsplit

import aiohttp

from ..image_format import guess_image_mime_and_ext
from ..logbook import get_logger
from .base import ProviderSettings, VideoBackend, VideoResult

_DEFAULT_BASE = "https://apihub.agnes-ai.com/v1"
_DEFAULT_MODEL = "agnes-video-2.5-flash"
_SUPPORTED_SECONDS = frozenset(str(value) for value in range(4, 13))
_SUPPORTED_ASPECTS = frozenset({"21:9", "16:9", "4:3", "1:1", "3:4", "9:16"})
_MAX_MEDIA_BYTES = 15 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_POLL_DONE = frozenset({"completed", "complete", "succeeded", "success", "finished", "done"})
_POLL_FAILED = frozenset({"failed", "error", "cancelled", "canceled"})
_VIDEO_KEYS = (
    "remixed_from_video_id",
    "url",
    "video_url",
    "output_url",
    "download_url",
    "file_url",
    "mp4_url",
)


logger = get_logger("provider.agnes")


class AgnesVideoError(RuntimeError):
    def __init__(self, message: str, *, stop_provider_chain: bool = False):
        super().__init__(message)
        self.stop_provider_chain = stop_provider_chain


def _normalize_base(raw: str) -> tuple[str, str]:
    """Return (api_url, status_url) and reject unusable endpoint input."""
    base = str(raw or _DEFAULT_BASE).strip().rstrip("/")
    parts = urlsplit(base)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.netloc
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Agnes 接口地址无效")
    base = base.removesuffix("/videos")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/videos", base[:-3] + "/agnesapi"


class _RateStore:
    """Persistent per-identity spacing so retries never bunch up."""

    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS rate_limit ("
                "identity TEXT PRIMARY KEY, next_at REAL NOT NULL)"
            )

    def reserve(self, identity: str, interval: float) -> float:
        with sqlite3.connect(self.path, timeout=5) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT next_at FROM rate_limit WHERE identity=?", (identity,)
            ).fetchone()
            now = time.time()
            wait = max(0.0, row[0] - now) if row else 0.0
            if not wait:
                db.execute(
                    "INSERT OR REPLACE INTO rate_limit VALUES (?,?)",
                    (identity, now + interval),
                )
            return wait


class AgnesVideoService(VideoBackend):
    def __init__(self, settings: ProviderSettings, data_dir: Path):
        self.settings = copy.deepcopy(settings)
        self.provider_id = settings.id
        self.api_url, self.status_url = _normalize_base(settings.base_url)
        keys = [str(item).strip() for item in (settings.api_keys or []) if str(item).strip()]
        self.api_key = keys[0] if keys else ""
        self.interval = max(61.0, float(settings.interval or 61))
        self.timeout = max(120, min(3600, int(settings.timeout or 1800)))
        self.request_timeout = max(10, min(300, int(settings.request_timeout or 120)))
        self.poll_interval = max(1.0, float(settings.poll_interval or 61))
        self.poll_retries = max(0, min(5, int(settings.poll_retries or 3)))
        self.seconds = str(settings.seconds or "12").strip()
        self.mode = str(settings.mode or "auto").strip().lower()
        self.size = str(settings.size or "720P").strip()
        self.aspect_ratio = str(settings.aspect_ratio or "16:9").strip()
        self.model = str(settings.model or _DEFAULT_MODEL).strip()
        self.extra_body = dict(settings.extra_body or {})
        self.proxy_url = str(settings.proxy_url or "").strip() or None
        netloc = urlsplit(self.api_url).netloc
        self.rate_identity = hashlib.sha256(
            (netloc + "\0" + self.api_key).encode("utf-8")
        ).hexdigest()
        self._rate = _RateStore(Path(data_dir) / "agnes_video" / "requests.sqlite3")
        self._session: aiohttp.ClientSession | None = None

    async def _session_get(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=max(1, int(self.request_timeout)))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    # -- rate limit ---------------------------------------------------------

    async def _wait_request(self, deadline: float) -> None:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Agnes 请求等待超时")
            wait = await asyncio.to_thread(
                self._rate.reserve, self.rate_identity, self.interval
            )
            if not wait:
                return
            if time.monotonic() + wait >= deadline:
                raise TimeoutError("Agnes 请求等待超时")
            await asyncio.sleep(wait)

    # -- payload ------------------------------------------------------------

    def build_payload(
        self,
        prompt: str,
        *,
        seconds: str = "",
        mode: str = "",
        images: list[Path] | None = None,
        audios: list[Path] | None = None,
        extra: dict | None = None,
    ) -> dict:
        merged_extra = {**self.extra_body, **(extra or {})}
        merged_extra.pop("aspect_ratio", None)
        payload: dict = {
            "model": self.model,
            "seconds": str(seconds or self.seconds).strip(),
            "size": self.size,
            "aspect_ratio": str((extra or {}).get("aspect_ratio") or self.aspect_ratio).strip(),
            "n": 1,
            **copy.deepcopy(merged_extra),
        }
        payload["prompt"] = str(prompt or "").strip()
        if not payload["prompt"]:
            raise ValueError("请填写视频提示词")
        if payload["model"] != _DEFAULT_MODEL:
            raise ValueError("Agnes 视频模型不受支持")

        image_urls = [self._image_data_url(path) for path in images or []]
        audio_urls = [self._audio_data_url(path) for path in audios or []]

        resolved = str(mode or payload.get("mode") or self.mode or "auto").strip().lower()
        if resolved in ("", "auto"):
            if image_urls or payload.get("first_frame") or payload.get("last_frame"):
                resolved = "keyframe"
            elif payload.get("images") or payload.get("audios"):
                resolved = "reference"
            else:
                resolved = "text" if not (image_urls or audio_urls) else "reference"
        payload["mode"] = resolved

        if image_urls:
            if resolved == "keyframe":
                if payload.get("first_frame"):
                    raise ValueError("消息图片与已配置首帧冲突, 请移除固定首帧")
                payload["first_frame"] = image_urls[0]
                image_urls = image_urls[1:]
            if image_urls:
                if resolved != "reference":
                    raise ValueError("收到图片时不能使用纯文本模式")
                existing = payload.get("images") or []
                if not isinstance(existing, list):
                    raise ValueError("Agnes images 必须为列表")
                payload["images"] = [*image_urls, *existing]
        if audio_urls:
            if resolved != "reference":
                raise ValueError("音频只能在参考模式下使用")
            existing_audio = payload.get("audios") or []
            if not isinstance(existing_audio, list):
                raise ValueError("Agnes audios 必须为列表")
            payload["audios"] = [*audio_urls, *existing_audio]

        self.validate_payload(payload)
        return payload

    def _image_data_url(self, path: Path) -> str:
        data = Path(path).read_bytes()
        if len(data) >= _MAX_MEDIA_BYTES:
            raise ValueError("Agnes 图片需小于 15 MB")
        mime, _ = guess_image_mime_and_ext(data)
        return "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii")

    @staticmethod
    def _audio_data_url(path: Path) -> str:
        data = Path(path).read_bytes()
        if len(data) >= _MAX_MEDIA_BYTES:
            raise ValueError("Agnes 音频需小于 15 MB")
        return "data:audio/mpeg;base64," + base64.b64encode(data).decode("ascii")

    @staticmethod
    def validate_payload(payload: dict) -> None:
        if payload["size"] != "720P":
            raise ValueError("Agnes Flash 仅支持 720P")
        if payload["seconds"] not in _SUPPORTED_SECONDS:
            raise ValueError("Agnes seconds 必须为字符串 4 至 12")
        if type(payload["n"]) is not int or payload["n"] != 1:
            raise ValueError("Agnes 每个任务仅支持 n=1")
        if payload["aspect_ratio"] not in _SUPPORTED_ASPECTS:
            raise ValueError("Agnes 画幅不受支持")
        if "seed" in payload and type(payload["seed"]) is not int:
            raise ValueError("Agnes seed 必须为整数")
        allowed = {
            "text": set(),
            "keyframe": {"first_frame", "last_frame"},
            "reference": {"images", "audios"},
        }
        mode = payload["mode"]
        if mode not in allowed:
            raise ValueError("Agnes mode 需为 text, keyframe 或 reference")
        media = {
            key
            for key in ("first_frame", "last_frame", "images", "audios", "videos")
            if payload.get(key)
        }
        if media - allowed[mode] or (mode != "text" and not media):
            raise ValueError("Agnes 媒体与生成模式不匹配, Flash 不接受视频参考")
        for name, limit in (("images", 5), ("audios", 3)):
            value = payload.get(name)
            if isinstance(value, list) and len(value) > limit:
                raise ValueError(f"Agnes {name} 必须为列表且最多 {limit} 项")
        urls = [payload[key] for key in ("first_frame", "last_frame") if payload.get(key)]
        urls += payload.get("images", []) + payload.get("audios", [])
        if any(
            not isinstance(item, str)
            or not item.startswith(("https://", "http://", "data:image/"))
            for item in urls
        ):
            raise ValueError("Agnes 媒体需为有效 URL")

    # -- http ---------------------------------------------------------------

    def _redact(self, text: str) -> str:
        if self.api_key:
            return str(text).replace(self.api_key, "[redacted]")
        return str(text)

    def _error_detail(self, data) -> str:
        detail = (
            data.get("error")
            or data.get("detail")
            or data.get("message")
            or data.get("code")
            or "未提供错误详情"
        )
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or "上游任务失败"
        return self._redact(str(detail))[:300]

    async def _request(
        self,
        method: str,
        url: str,
        *,
        deadline: float,
        payload: dict | None = None,
    ) -> tuple[int, dict, dict]:
        await self._wait_request(deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Agnes 请求超时")
        session = await self._session_get()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        async with session.request(
            method,
            url,
            headers=headers,
            json=payload,
            proxy=self.proxy_url or None,
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=min(self.request_timeout, remaining)),
        ) as resp:
            body = await resp.read()
            if len(body) > _MAX_RESPONSE_BYTES:
                raise AgnesVideoError("Agnes 响应过大", stop_provider_chain=True)
            try:
                data = json.loads(body) if body else {}
            except ValueError:
                data = {}
            if not isinstance(data, dict):
                data = {}
            return resp.status, data, dict(resp.headers)

    # -- polling ------------------------------------------------------------

    @staticmethod
    def _extract_video_url(data: dict) -> str | None:
        candidates = [data.get(key) for key in _VIDEO_KEYS]
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            candidates.append(metadata.get("url"))
        output = data.get("output") or data.get("result") or data.get("data")
        if isinstance(output, dict):
            candidates.extend(output.get(key) for key in _VIDEO_KEYS)
        elif isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    candidates.extend(item.get(key) for key in _VIDEO_KEYS)
                elif isinstance(item, str):
                    candidates.append(item)
        for value in candidates:
            if isinstance(value, str) and value.startswith(("https://", "http://")):
                return value
        for match in re.findall(r"https?://[^\"]+", json.dumps(data)):
            if ".mp4" in match:
                return match
            if "video" in match and "webhook" not in match:
                return match
        return None

    def _build_poll_endpoints(self, video_id: str, task_id: str, model: str) -> list[str]:
        endpoints: list[str] = []
        if video_id:
            endpoints.append(
                self.status_url
                + "?video_id="
                + quote(video_id, safe="")
                + "&model_name="
                + quote(model, safe="")
            )
        target = task_id or video_id
        if target:
            endpoints.append(self.api_url + "/" + quote(target, safe=""))
        return endpoints

    async def _poll(self, video_id: str, task_id: str, model: str, deadline: float) -> str:
        endpoints = self._build_poll_endpoints(video_id, task_id, model)
        if not endpoints:
            raise AgnesVideoError(
                "Agnes 创建响应缺少可轮询的任务 ID", stop_provider_chain=True
            )
        last_status = "unknown"
        last_progress = None
        last_error = None
        last_endpoint = endpoints[0]
        polls = 0
        logger.info(
            "[aiimg] Agnes 轮询开始：video_id=%s task_id=%s 间隔 %.0f 秒",
            video_id or "-", task_id or "-", self.poll_interval,
        )
        while time.monotonic() < deadline:
            await asyncio.sleep(min(self.poll_interval, max(0.0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                break
            for endpoint in endpoints:
                last_endpoint = endpoint
                try:
                    status, data, _headers = await self._request(
                        "GET", endpoint, deadline=deadline
                    )
                except AgnesVideoError:
                    raise
                except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
                    last_error = self._redact(str(exc))
                    continue
                if status >= 400:
                    last_error = f"HTTP {status}: {self._error_detail(data)}"
                    continue
                if not data:
                    continue
                state = str(data.get("status") or data.get("state") or "").lower()
                last_status = state or "unknown"
                last_progress = data.get("progress")
                polls += 1
                logger.info(
                    "[aiimg] Agnes 第 %d 次轮询：状态 %s，进度 %s",
                    polls, last_status, last_progress if last_progress is not None else "N/A",
                )
                last_error = data.get("error") or data.get("message") or data.get("detail")
                url = self._extract_video_url(data)
                if url:
                    logger.info("[aiimg] Agnes 出片完成，共轮询 %d 次", polls)
                    return url
                if state in _POLL_DONE:
                    raise AgnesVideoError(
                        f"Agnes 已完成但未返回视频地址, video_id={video_id or task_id}",
                        stop_provider_chain=True,
                    )
                if state in _POLL_FAILED:
                    raise AgnesVideoError(f"Agnes 任务失败: {self._error_detail(data)}")
        raise AgnesVideoError(
            f"Agnes 等待超时, video_id={video_id or task_id}; 请到平台查询, 不会重新创建任务; "
            f"最后状态={last_status}; "
            f"进度={last_progress if last_progress is not None else 'N/A'}; "
            f"最后错误={last_error or 'N/A'}; "
            f"最后轮询接口={last_endpoint}",
            stop_provider_chain=True,
        )

    # -- entry --------------------------------------------------------------

    async def generate_video(
        self,
        prompt: str,
        *,
        seconds: str = "",
        mode: str = "",
        images: list[Path] | None = None,
        audios: list[Path] | None = None,
        extra: dict | None = None,
    ) -> VideoResult:
        if not self.api_key:
            raise AgnesVideoError("请配置 Agnes API Key", stop_provider_chain=True)
        try:
            payload = self.build_payload(
                prompt,
                seconds=seconds,
                mode=mode,
                images=images,
                audios=audios,
                extra=extra,
            )
        except ValueError as exc:
            # Bad request shape: no other provider can satisfy it either.
            raise AgnesVideoError(str(exc), stop_provider_chain=True) from exc
        deadline = time.monotonic() + self.timeout
        data: dict | None = None
        attempt = 0
        while data is None:
            try:
                status, body, headers = await self._request(
                    "POST", self.api_url, deadline=deadline, payload=payload
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, TimeoutError) as exc:
                raise AgnesVideoError(
                    "Agnes 创建请求结果不明, 不会自动重建任务, 请先到平台检查",
                    stop_provider_chain=True,
                ) from exc
            if status < 400:
                data = body
                continue
            detail = f"Agnes HTTP {status}: {self._error_detail(body)}"
            transient = status == 429 or (
                status == 503
                and (
                    "queue is full" in detail.lower()
                    or "video_queue_full" in detail.lower()
                )
            )
            if not transient:
                raise AgnesVideoError(
                    detail, stop_provider_chain=(status >= 500 or status == 408)
                )
            retry_after = _retry_after_seconds(headers.get("Retry-After"))
            delay = float(retry_after) if retry_after else min(90.0, 20.0 + 15.0 * attempt)
            attempt += 1
            if time.monotonic() + min(delay, 1.0) >= deadline:
                raise AgnesVideoError(detail, stop_provider_chain=True)
            await asyncio.sleep(delay)

        if not data:
            raise AgnesVideoError("Agnes 返回无效任务响应", stop_provider_chain=True)
        video_id = data.get("video_id")
        task_id = data.get("id") or data.get("task_id")
        if not (
            isinstance(video_id, str) and video_id.strip()
        ) and not (isinstance(task_id, str) and task_id.strip()):
            raise AgnesVideoError(
                "Agnes 创建响应缺少 video_id/task_id, 无法轮询任务",
                stop_provider_chain=True,
            )
        url = await self._poll(
            str(video_id or ""), str(task_id or ""), str(payload["model"]), deadline
        )
        return VideoResult(
            url=url,
            model=str(payload["model"]),
            size=str(payload["size"]),
            duration=str(payload.get("seconds") or ""),
        )


def _retry_after_seconds(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        value = float(str(raw).strip())
    except ValueError:
        return None
    return value if value > 0 else None
