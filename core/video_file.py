"""Bounded downloads and container checks for generated MP4/MOV files."""

from __future__ import annotations

import asyncio
import os
import struct
import uuid
from pathlib import Path
from urllib.parse import urljoin

import aiohttp

from .net import URLFetchPolicy, ensure_url_allowed

MAX_VIDEO_BYTES = 256 * 1024 * 1024


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


async def download_video(url: str, directory: Path, policy: URLFetchPolicy) -> Path:
    """Download fully before publishing; validate every redirect and clean failures."""
    target = str(url or "").strip()
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{uuid.uuid4().hex}.mp4"
    partial = destination.with_suffix(".part")
    try:
        timeout = aiohttp.ClientTimeout(total=180, sock_read=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _ in range(6):
                await ensure_url_allowed(target, policy)
                async with session.get(target, allow_redirects=False) as response:
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location:
                            raise ValueError("视频重定向缺少地址")
                        target = urljoin(target, location)
                        continue
                    if response.status != 200:
                        raise ValueError(f"视频下载失败: HTTP {response.status}")
                    if response.content_length is not None and response.content_length > MAX_VIDEO_BYTES:
                        raise ValueError("视频超过 256 MB")
                    size = 0
                    with partial.open("wb") as output:
                        async for chunk in response.content.iter_chunked(64 * 1024):
                            size += len(chunk)
                            if size > MAX_VIDEO_BYTES:
                                raise ValueError("视频超过 256 MB")
                            output.write(chunk)
                    if response.content_length is not None and size != response.content_length:
                        raise ValueError("视频文件下载不完整")
                    await asyncio.to_thread(validate_video_file, partial)
                    os.replace(partial, destination)
                    return destination
            raise ValueError("视频重定向次数过多")
    finally:
        partial.unlink(missing_ok=True)
