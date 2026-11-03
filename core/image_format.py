"""Image format helpers shared by providers and storage."""

from __future__ import annotations

import base64
import io
import re


def guess_image_mime_and_ext(image_bytes: bytes) -> tuple[str, str]:
    """Return a best-effort MIME type and extension from magic bytes."""
    data = image_bytes or b""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    return "image/jpeg", "jpg"


def validate_image_bytes(data: bytes) -> None:
    if not data:
        raise ValueError("图片内容为空")
    if len(data) > 24 * 1024 * 1024:
        raise ValueError("图片超过 24 MB")
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            image.verify()
        # Verification alone does not decode JPEG/WebP pixel data.
        with PILImage.open(io.BytesIO(data)) as image:
            image.load()
    except Exception as exc:
        raise ValueError("返回内容不是有效图片") from exc


def decode_base64_image(payload: str) -> bytes:
    text = re.sub(r"\s+", "", str(payload or ""))
    if text.startswith("base64://"):
        text = text.removeprefix("base64://")
    if text.startswith("data:"):
        _, separator, body = text.partition(",")
        if not separator or not body:
            raise ValueError("data:image 内容不完整")
        text = body
    try:
        data = base64.b64decode(text, validate=False)
    except Exception as exc:
        raise ValueError("图片 Base64 解码失败") from exc
    validate_image_bytes(data)
    return data


def image_data_url(data: bytes, mime: str = "") -> str:
    """Wrap image bytes as a data: URL, guessing the MIME type if omitted."""
    resolved = str(mime or "").strip() or guess_image_mime_and_ext(data)[0]
    return f"data:{resolved};base64,{base64.b64encode(data).decode('ascii')}"


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) of an image, or None on failure."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


def is_placeholder_image(data: bytes) -> bool:
    """Return True for tiny/blank images that are not worth keeping.

    Covers: empty or near-empty byte payloads (< 128 bytes), and 1×1 pixel
    images (the most common placeholder a provider returns when a request
    fails silently).
    """
    if len(data) <= 128:
        return True
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            w, h = image.size
            if w <= 1 or h <= 1:
                return True
    except Exception:
        pass
    return False


def log_size_mismatch(
    logger: object,
    data: bytes,
    *,
    requested_size: str = "",
    source: str = "",
) -> None:
    """Log a warning when the actual image dimensions differ from requested.

    This is a best-effort, non-blocking check — many providers reinterpret
    or ignore size hints.
    """
    dims = image_dimensions(data)
    if dims is None:
        return
    got = f"{dims[0]}x{dims[1]}"
    want = str(requested_size or "").strip()
    if not want:
        return
    # Both should look like "1024x1024" or "854x1842".
    want_norm = want.replace(" ", "").lower()
    got_norm = got.replace(" ", "").lower()
    if want_norm == got_norm:
        return
    prefix = f"[{source}] " if source else ""
    logger.warning(
        "%s输出尺寸与请求不一致: requested=%s got=%s (服务商可能忽略 size)",
        prefix,
        want,
        got,
    )


def thumbnail_webp(data: bytes, size: int = 480, quality: int = 82) -> bytes:
    """Downscale image bytes to a small WebP; return the input on failure."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            image.thumbnail((size, size))
            output = io.BytesIO()
            image.convert("RGB").save(output, format="WEBP", quality=quality)
            return output.getvalue()
    except Exception:
        return data


def preview_jpeg(data: bytes, size: int = 2048, quality: int = 90) -> bytes:
    """Downscale image bytes to a viewing JPEG; return the input on failure."""
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            image.thumbnail((size, size))
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG", quality=quality)
            return output.getvalue()
    except Exception:
        return data
