"""Bounded downloads and container checks for generated MP4/MOV files."""

from __future__ import annotations

import asyncio
import json
import os
import re
import struct
import uuid
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import aiohttp

from .http import current_session
from .net import URLFetchPolicy, ensure_peer_allowed, ensure_url_allowed, origin

MAX_VIDEO_BYTES = 256 * 1024 * 1024

_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_REDIRECTS = 6
_MAX_ATTEMPTS = 3
# Ranges only pay off on files worth splitting; small clips stream faster whole.
_RANGE_THRESHOLD = 4 * 1024 * 1024
_RANGE_CHUNK = 4 * 1024 * 1024
_RANGE_CONCURRENCY = 4
_SNIFF_LIMIT = 256 * 1024
_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".m4v")
_VIDEO_URL_PATTERN = re.compile(
    r"https?://[^\s\"'<>]+?\.(?:mp4|webm|mov|m4v)(?:\?[^\s\"'<>]*)?",
    re.IGNORECASE,
)
_RANGE_PATTERN = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+)", re.IGNORECASE)


def validate_video_file(path: Path) -> None:
    """Check complete ISO-BMFF boxes and a video track, not codec decodability."""
    total = path.stat().st_size
    if not total or total > MAX_VIDEO_BYTES:
        raise ValueError("视频文件为空或超过 256 MB")
    kinds: set[bytes] = set()
    has_video = False
    has_payload = False
    with path.open("rb") as stream:
        def boxes(start: int, end: int, depth: int = 0) -> None:
            nonlocal has_video, has_payload
            if depth > 8:
                raise ValueError("视频容器嵌套过深")
            pos = start
            while pos < end:
                if end - pos < 8:
                    raise ValueError("视频文件不完整")
                stream.seek(pos)
                size, kind = struct.unpack(">I4s", stream.read(8))
                header = 8
                if size == 1:
                    if end - pos < 16:
                        raise ValueError("视频文件不完整")
                    size = struct.unpack(">Q", stream.read(8))[0]
                    header = 16
                elif size == 0:
                    size = end - pos
                if size < header or pos + size > end:
                    raise ValueError("视频文件不完整")
                if depth == 0:
                    kinds.add(kind)
                    if kind == b"mdat" and size > header:
                        has_payload = True
                if kind in {b"moov", b"trak", b"mdia"}:
                    boxes(pos + header, pos + size, depth + 1)
                elif kind == b"hdlr" and depth == 3 and size >= header + 12:
                    stream.seek(pos + header + 8)
                    has_video |= stream.read(4) == b"vide"
                pos += size
        boxes(0, total)
    if b"moov" not in kinds or not has_payload or not has_video:
        raise ValueError("返回内容不是完整的 MP4/MOV 视频")


def _video_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = urlsplit(text).path.lower()
    return text if path.endswith(_VIDEO_SUFFIXES) else ""


def _walk_video_url(payload, base: str) -> str:
    """Find a video link inside a JSON wrapper the provider returned."""
    if isinstance(payload, dict):
        for key in ("url", "video_url", "videoUrl", "download_url", "downloadUrl", "src"):
            found = _video_url(str(payload.get(key) or ""))
            if found:
                return urljoin(base, found)
        for value in payload.values():
            found = _walk_video_url(value, base)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _walk_video_url(item, base)
            if found:
                return found
    return ""


async def _read_sniff_body(response) -> bytes:
    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) >= _SNIFF_LIMIT:
            break
    return bytes(body)


async def _resolve_direct_url(
    session: aiohttp.ClientSession,
    url: str,
    policy: URLFetchPolicy,
    headers: dict[str, str],
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> str:
    """Follow redirects and unwrap HTML/JSON landing pages down to a media file."""
    current = str(url or "").strip()
    for _ in range(_MAX_REDIRECTS):
        if _video_url(current):
            return current
        await ensure_url_allowed(current, policy)
        async with session.get(
            current,
            headers=headers or None,
            allow_redirects=False,
            timeout=timeout,
        ) as resp:
            await ensure_peer_allowed(resp, current, policy)
            if resp.status in _REDIRECT_STATUSES:
                location = (resp.headers.get("Location") or "").strip()
                if not location:
                    raise ValueError("视频重定向缺少地址")
                current = urljoin(current, location)
                continue
            if resp.status >= 400:
                return current
            kind = (resp.headers.get("Content-Type") or "").lower()
            if kind.startswith("video/") or kind.startswith("application/octet-stream"):
                return current
            body = await _read_sniff_body(resp)
        text = body.decode("utf-8", errors="replace")
        if "json" in kind or text.lstrip().startswith(("{", "[")):
            try:
                found = _walk_video_url(json.loads(text), current)
            except (TypeError, ValueError):
                found = ""
            if found:
                return found
        match = _VIDEO_URL_PATTERN.search(text)
        if match:
            return match.group(0)
        return current
    raise ValueError("视频重定向次数过多")


async def _probe(
    session: aiohttp.ClientSession,
    url: str,
    policy: URLFetchPolicy,
    headers: dict[str, str],
    cap: int,
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> tuple[int, str]:
    """Ask for one byte to learn the total size and whether ranges work."""
    probe_headers = dict(headers or {})
    probe_headers["Range"] = "bytes=0-0"
    await ensure_url_allowed(url, policy)
    async with session.get(
        url,
        headers=probe_headers,
        allow_redirects=False,
        timeout=timeout,
    ) as resp:
        await ensure_peer_allowed(resp, url, policy)
        if resp.status != 206:
            return 0, ""
        match = _RANGE_PATTERN.fullmatch((resp.headers.get("Content-Range") or "").strip())
        if not match:
            return 0, ""
        total = int(match.group(3))
        if total <= 0 or total > cap:
            return 0, ""
        kind = (resp.headers.get("Content-Type") or "").lower()
        return total, kind


async def _download_ranges(
    session: aiohttp.ClientSession,
    url: str,
    target: Path,
    total: int,
    headers: dict[str, str],
    cap: int,
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> None:
    """Pull byte ranges in parallel; each worker retries its own slice."""
    parts = [target.with_name(f"{target.name}.{index:05d}") for index in
             range(0, total, _RANGE_CHUNK)]
    ranges = [
        (start, min(total - 1, start + _RANGE_CHUNK - 1))
        for start in range(0, total, _RANGE_CHUNK)
    ]
    next_index = 0
    lock = asyncio.Lock()

    async def fetch(index: int, start: int, end: int) -> None:
        expected = end - start + 1
        request_headers = dict(headers or {})
        request_headers["Range"] = f"bytes={start}-{end}"
        last_error: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                body = bytearray()
                async with session.get(
                    url,
                    headers=request_headers,
                    allow_redirects=False,
                    timeout=timeout,
                ) as resp:
                    await ensure_peer_allowed(resp, url, policy)
                    if resp.status != 206:
                        raise RuntimeError(f"视频分片下载失败: HTTP {resp.status}")
                    async for chunk in resp.content.iter_chunked(64 * 1024):
                        body.extend(chunk)
                        if len(body) > expected:
                            raise RuntimeError("视频分片长度超出预期")
                if len(body) != expected:
                    raise RuntimeError("视频分片下载不完整")
                await asyncio.to_thread(parts[index].write_bytes, bytes(body))
                return
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
                last_error = exc
                if attempt + 1 < _MAX_ATTEMPTS:
                    await asyncio.sleep(min(2 ** attempt, 8))
        raise last_error or RuntimeError("视频分片下载失败")

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= len(ranges):
                    return
                index = next_index
                next_index += 1
            start, end = ranges[index]
            await fetch(index, start, end)

    workers = [asyncio.create_task(worker()) for _ in range(min(_RANGE_CONCURRENCY, len(ranges)))]
    try:
        await asyncio.gather(*workers)
        with target.open("wb") as output:
            for part in parts:
                output.write(await asyncio.to_thread(part.read_bytes))
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        for part in parts:
            part.unlink(missing_ok=True)
    if target.stat().st_size > cap:
        raise ValueError("视频超过大小上限")


async def _download_stream(
    session: aiohttp.ClientSession,
    url: str,
    target: Path,
    policy: URLFetchPolicy,
    headers: dict[str, str],
    cap: int,
    *,
    timeout: aiohttp.ClientTimeout | None = None,
) -> None:
    """Stream to disk, resuming from whatever bytes survived an interruption."""
    current = url
    redirects = 0
    for attempt in range(_MAX_ATTEMPTS):
        resume = target.stat().st_size if target.exists() else 0
        if resume >= cap:
            raise ValueError("视频超过大小上限")
        request_headers = dict(headers or {})
        if resume:
            request_headers["Range"] = f"bytes={resume}-"
        try:
            while True:
                await ensure_url_allowed(current, policy)
                async with session.get(current, headers=request_headers or None,
                                       allow_redirects=False, timeout=timeout) as resp:
                    await ensure_peer_allowed(resp, current, policy)
                    if resp.status in _REDIRECT_STATUSES:
                        redirects += 1
                        if redirects > _MAX_REDIRECTS:
                            raise ValueError("视频重定向次数过多")
                        location = (resp.headers.get("Location") or "").strip()
                        if not location:
                            raise ValueError("视频重定向缺少地址")
                        current = urljoin(current, location)
                        continue
                    if resp.status == 416 and resume:
                        # Server says our offset is already past the end.
                        return
                    if resp.status not in {200, 206}:
                        raise ValueError(f"视频下载失败: HTTP {resp.status}")
                    if resp.status == 206:
                        match = _RANGE_PATTERN.match(
                            (resp.headers.get("Content-Range") or "").strip()
                        )
                        if not match or int(match.group(1)) != resume:
                            raise RuntimeError("服务商返回了意外的视频字节范围")
                        mode = "ab"
                    else:
                        # A server that ignores Range restarts from byte zero.
                        resume = 0
                        mode = "wb"
                    written = resume
                    with target.open(mode) as output:
                        async for chunk in resp.content.iter_chunked(64 * 1024):
                            written += len(chunk)
                            if written > cap:
                                raise ValueError("视频超过大小上限")
                            output.write(chunk)
                    declared = resp.headers.get("Content-Length")
                    if declared and declared.isdigit() and resp.status == 200:
                        if written != int(declared):
                            raise RuntimeError("视频文件下载不完整")
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as exc:
            if attempt + 1 >= _MAX_ATTEMPTS:
                raise
            kept = target.stat().st_size if target.exists() else 0
            if not kept:
                raise
            await asyncio.sleep(min(2 ** attempt, 8))


async def download_video(
    url: str,
    directory: Path,
    policy: URLFetchPolicy,
    *,
    max_bytes: int | None = None,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
    parallel: bool = True,
) -> Path:
    """Download fully before publishing; validate every redirect and clean failures.

    `headers` are provider credentials and are only sent to the provider origin,
    never to a CDN the provider redirects to. Interrupted transfers resume from
    the bytes already on disk; large files use parallel byte ranges when the
    server supports them.
    """
    cap = max_bytes if max_bytes is not None else MAX_VIDEO_BYTES
    target = str(url or "").strip()
    if not target:
        raise ValueError("缺少视频地址")
    auth = dict(headers or {})
    auth_origin = origin(target)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex}.mp4"
    partial = destination.with_name(f"{destination.name}.part")
    timeout_total = aiohttp.ClientTimeout(total=timeout or 180, sock_read=30)
    # 复用进程内共享会话，连接与 DNS 缓存跨任务生效；关会话由插件统一负责。
    session = await current_session()
    try:
        resolved = await _resolve_direct_url(
            session, target, policy, auth, timeout=timeout_total
        )
        use_headers = auth if auth and origin(resolved) == auth_origin else {}
        if parallel and resolved == target:
            total, _ = await _probe(
                session, resolved, policy, use_headers, cap, timeout=timeout_total
            )
            if total > _RANGE_THRESHOLD:
                await _download_ranges(
                    session, resolved, partial, total, use_headers, cap,
                    timeout=timeout_total,
                )
                await asyncio.to_thread(validate_video_file, partial)
                os.replace(partial, destination)
                return destination
        await _download_stream(
            session, resolved, partial, policy, use_headers, cap, timeout=timeout_total
        )
        await asyncio.to_thread(validate_video_file, partial)
        os.replace(partial, destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)
