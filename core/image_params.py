"""Validation and normalization for per-request image generation knobs."""

from __future__ import annotations

from typing import Any

from .logbook import get_logger

logger = get_logger("aiimg")


def build_image_params(
    *,
    n: int = 1,
    quality: str = "",
    complexity: str = "",
    background: str = "",
    output_format: str = "",
    output_compression: int = -1,
    moderation: str = "",
    frame_count: int = 9,
    frame_duration_ms: int = 200,
    markup: str = "",
    formats: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate the optional image knobs once, then pass them to the backend."""
    try:
        wanted_n = int(n or 1)
    except (TypeError, ValueError) as exc:
        raise ValueError("张数 n 要是 1 到 10 的整数") from exc
    if not 1 <= wanted_n <= 10:
        raise ValueError("张数 n 要在 1 到 10 之间")
    fmt = str(output_format or "").strip().lower().replace("jpg", "jpeg")
    if fmt and fmt not in ("png", "jpeg", "webp", "gif", "svg"):
        raise ValueError("输出格式只能是 png、jpeg、webp、gif 或 svg")
    if fmt and formats and fmt not in formats:
        raise ValueError("这个工具只支持 png、jpeg 或 webp 格式")
    quality = str(quality or "").strip().lower()
    if quality and quality not in ("low", "medium", "high", "auto"):
        raise ValueError("画质只能是 low、medium、high 或 auto")
    complexity = str(complexity or "").strip().lower()
    if complexity and complexity not in ("low", "medium", "high"):
        raise ValueError("复杂度只能是 low、medium 或 high")
    background = str(background or "").strip().lower()
    if background and background not in ("transparent", "opaque", "auto"):
        raise ValueError("背景只能是 transparent、opaque 或 auto")
    moderation = str(moderation or "").strip().lower()
    if moderation and moderation not in ("low", "auto"):
        raise ValueError("审核强度只能是 low 或 auto")
    try:
        compression = int(output_compression)
    except (TypeError, ValueError) as exc:
        raise ValueError("压缩率要是 0 到 100 的整数") from exc
    if compression > 100 or (compression < 0 and compression != -1):
        raise ValueError("压缩率要是 0 到 100 的整数")
    try:
        frames = int(frame_count or 9)
        frame_ms = int(frame_duration_ms or 200)
    except (TypeError, ValueError) as exc:
        raise ValueError("GIF 帧数和每帧时长要是整数") from exc
    if not 2 <= frames <= 64:
        raise ValueError("GIF 帧数要在 2 到 64 之间")
    if not 20 <= frame_ms <= 2000:
        raise ValueError("GIF 每帧时长要在 20 到 2000 毫秒之间")

    if fmt == "svg" and wanted_n != 1:
        logger.info("[aiimg] SVG 只支持一张，n 从 %d 改为 1", wanted_n)
        wanted_n = 1
    if fmt == "gif" and frames > 16:
        logger.warning("[aiimg] GIF 帧数 %d 较大，生成会明显变慢", frames)

    params: dict[str, Any] = {}
    if wanted_n != 1:
        params["n"] = wanted_n
    if quality:
        params["quality"] = quality
    if complexity:
        params["complexity"] = complexity
    if background:
        params["background"] = background
    if fmt:
        params["output_format"] = fmt
    if compression >= 0 and fmt not in ("gif", "svg"):
        params["output_compression"] = compression
    if moderation:
        params["moderation"] = moderation
    if fmt == "gif":
        params["frame_count"] = frames
        params["frame_duration_ms"] = frame_ms
    clean_markup = str(markup or "").strip()
    if clean_markup:
        params["markup"] = clean_markup[:8000]
    return params
