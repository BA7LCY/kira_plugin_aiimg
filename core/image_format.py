"""Image format helpers shared by providers and storage."""

from __future__ import annotations

import base64
import hashlib
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


# 解码前先按像素挡一道，避免压缩得很小的图把内存撑爆。
MAX_PIXELS = 50_000_000


def validate_image_bytes(
    data: bytes, max_bytes: int | None = None, *, max_pixels: int = MAX_PIXELS
) -> None:
    cap = max_bytes if max_bytes else 24 * 1024 * 1024
    if not data:
        raise ValueError("图片内容为空")
    if len(data) > cap:
        raise ValueError(f"图片超过 {cap // (1024 * 1024)} MB")
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            width, height = image.size
        if max_pixels and width * height > max_pixels:
            raise ValueError(f"图片尺寸太大：{width}x{height}")
        with PILImage.open(io.BytesIO(data)) as image:
            image.verify()
        # Verification alone does not decode JPEG/WebP pixel data.
        with PILImage.open(io.BytesIO(data)) as image:
            image.load()
    except ValueError:
        raise
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


def normalize_orientation(data: bytes) -> bytes:
    """Bake the EXIF orientation into the pixels and drop the tag.

    Phone cameras store "rotate me 90°" in EXIF instead of rotating pixels.
    Anything that re-encodes the image (thumbnails, collages, edits) would
    otherwise show it sideways, so we normalize once at ingest.
    """
    try:
        from PIL import Image as PILImage
        from PIL import ImageOps

        with PILImage.open(io.BytesIO(data)) as image:
            exif = image.getexif()
            orientation = int(exif.get(0x0112, 1) or 1) if exif else 1
            if orientation in (0, 1):
                return data
            fixed = ImageOps.exif_transpose(image)
            if fixed is None:
                return data
            if fixed.mode not in ("RGB", "RGBA", "L", "LA"):
                fixed = fixed.convert("RGB")
            output = io.BytesIO()
            fmt = "PNG" if (image.format or "").upper() == "PNG" else "JPEG"
            save_args: dict = {"format": fmt}
            if fmt == "JPEG":
                save_args.update(quality=95, subsampling=0)
                if fixed.mode not in ("RGB", "L"):
                    fixed = fixed.convert("RGB")
            fixed.save(output, **save_args)
            return output.getvalue()
    except Exception:
        return data


def content_digest(data: bytes) -> str:
    """Stable fingerprint for duplicate detection."""
    return hashlib.sha256(data or b"").hexdigest()


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
        "%s输出尺寸与请求不一致: requested=%s got=%s",
        prefix,
        want,
        got,
    )


def thumbnail_webp(
    data: bytes, size: int = 480, quality: int = 82, fmt: str = "webp"
) -> bytes:
    """Downscale image bytes to a small preview; return the input on failure.

    `fmt` may be "webp" or "jpeg"; the filename suffix should match.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            image.thumbnail((size, size))
            output = io.BytesIO()
            if str(fmt).lower() == "jpeg":
                image.convert("RGB").save(output, format="JPEG", quality=quality)
            else:
                image.convert("RGB").save(output, format="WEBP", quality=quality)
            return output.getvalue()
    except Exception:
        return data


def preview_jpeg(
    data: bytes, size: int = 2048, quality: int = 90, fmt: str = "jpeg"
) -> bytes:
    """Downscale image bytes to a viewing preview; return the input on failure.

    `fmt` may be "jpeg" or "webp"; the filename suffix should match.
    """
    try:
        from PIL import Image as PILImage

        with PILImage.open(io.BytesIO(data)) as image:
            image.thumbnail((size, size))
            output = io.BytesIO()
            if str(fmt).lower() == "webp":
                image.convert("RGB").save(output, format="WEBP", quality=quality)
            else:
                image.convert("RGB").save(output, format="JPEG", quality=quality)
            return output.getvalue()
    except Exception:
        return data
