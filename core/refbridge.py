"""Explicit reference bridging: provenance, de-duplication and byte budgets.

Every picture a task sends as a reference has to come from somewhere we can
name: the message being answered, a quoted message, the gallery this plugin
filled, or a photo the operator uploaded under ``refs/``. A path that matches
none of those is refused, so a model cannot talk the plugin into reading an
arbitrary file off the host.

Identical bytes are sent once. Two references of the same photo only differ by
the role they play, so the first occurrence wins and a later, more specific
role upgrades it in place instead of dropping the role from the prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_IMAGES = 8
DEFAULT_MAX_EACH_MB = 20
DEFAULT_MAX_TOTAL_MB = 64

SOURCE_MESSAGE = "message"
SOURCE_QUOTED = "quoted"
SOURCE_GALLERY = "gallery"
SOURCE_UPLOAD = "upload"
SOURCE_PERSONA = "persona"

SOURCE_LABELS: dict[str, str] = {
    SOURCE_MESSAGE: "当前消息",
    SOURCE_QUOTED: "引用消息",
    SOURCE_GALLERY: "画廊",
    SOURCE_UPLOAD: "参考图库",
    SOURCE_PERSONA: "人设参考",
}

# "base" is the picture being edited and "persona" the fixed identity photo.
# Both outrank a plain subject reference, which is what a bare attachment is.
_ROLE_RANK: dict[str, int] = {
    "subject": 0,
    "base": 1,
    "persona": 1,
    "clothing": 2,
    "style": 2,
    "pose": 2,
    "background": 2,
}


@dataclass
class ReferenceImage:
    """One input picture, with where it came from and what it costs."""

    path: Path
    role: str
    source: str
    digest: str
    size: int


def _under(path: Path, directory: Path | None) -> bool:
    if directory is None:
        return False
    try:
        return path.parent.resolve() == Path(directory).resolve()
    except OSError:
        return False


def digest_of(path: Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of the file, streamed so a large picture never lands in memory."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def classify(
    path: Path,
    *,
    attachments: list[Path] | None = None,
    quoted: list[Path] | None = None,
    gallery_dir: Path | None = None,
    refs_dir: Path | None = None,
    persona: Path | None = None,
) -> str:
    """Return the provenance of a picture, or "" when it is not ours to read."""
    target = Path(path)
    try:
        resolved = target.resolve()
    except OSError:
        resolved = target
    for candidate in attachments or []:
        if Path(candidate).resolve() == resolved:
            return SOURCE_MESSAGE
    for candidate in quoted or []:
        if Path(candidate).resolve() == resolved:
            return SOURCE_QUOTED
    if persona is not None and Path(persona).resolve() == resolved:
        return SOURCE_PERSONA
    if _under(resolved, gallery_dir):
        return SOURCE_GALLERY
    if _under(resolved, refs_dir):
        return SOURCE_UPLOAD
    return ""


def _limit_bytes(megabytes: float) -> int:
    return int(max(0.0, float(megabytes)) * 1024 * 1024)


def plan_references(
    candidates: list[tuple[Path, str]],
    *,
    attachments: list[Path] | None = None,
    quoted: list[Path] | None = None,
    gallery_dir: Path | None = None,
    refs_dir: Path | None = None,
    persona: Path | None = None,
    max_images: int = DEFAULT_MAX_IMAGES,
    max_each_mb: float = DEFAULT_MAX_EACH_MB,
    max_total_mb: float = DEFAULT_MAX_TOTAL_MB,
) -> tuple[list[ReferenceImage], list[str]]:
    """Validate, de-duplicate and budget reference pictures.

    Returns the kept pictures in send order plus human-readable notes about
    what was dropped. Raises ``ValueError`` when a hard limit is exceeded: a
    silent truncation would change what the prompt describes.
    """
    each_cap = _limit_bytes(max_each_mb)
    total_cap = _limit_bytes(max_total_mb)
    kept: list[ReferenceImage] = []
    notes: list[str] = []
    by_digest: dict[str, ReferenceImage] = {}
    total = 0
    for raw_path, raw_role in candidates:
        path = Path(raw_path)
        role = str(raw_role or "subject").strip().lower() or "subject"
        source = classify(
            path,
            attachments=attachments,
            quoted=quoted,
            gallery_dir=gallery_dir,
            refs_dir=refs_dir,
            persona=persona,
        )
        if not source:
            notes.append(f"跳过来源不明的图片：{path.name}")
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > each_cap:
            raise ValueError(
                f"每张参考图不能超过 {int(max_each_mb)} MB：{path.name} 有 "
                f"{size // (1024 * 1024)} MB"
            )
        digest = digest_of(path)
        seen = by_digest.get(digest)
        if seen is not None:
            # The picture being edited keeps its role: a duplicate that only
            # wants to borrow its style must not relabel the original.
            if seen.role != "base" and _ROLE_RANK.get(role, 2) > _ROLE_RANK.get(
                seen.role, 0
            ):
                notes.append(
                    f"同一张图出现多次，{path.name} 的角色改为 {role}"
                )
                seen.role = role
            else:
                notes.append(f"跳过重复的图片：{path.name}")
            continue
        total += size
        if total > total_cap:
            raise ValueError(
                f"参考图总量不能超过 {int(max_total_mb)} MB，"
                f"现在加上 {path.name} 会到 {total // (1024 * 1024)} MB"
            )
        entry = ReferenceImage(
            path=path, role=role, source=source, digest=digest, size=size
        )
        by_digest[digest] = entry
        kept.append(entry)
        if len(kept) > max_images:
            raise ValueError(
                f"参考图最多 {max_images} 张，现在有 {len(kept)} 张（已去重）："
                "请减少图片，或用 reference_roles 只挑需要的角色。"
            )
    return kept, notes


def describe(references: list[ReferenceImage]) -> str:
    """Compact log line: how many, from where, how many bytes."""
    if not references:
        return "无"
    counts: dict[str, int] = {}
    for item in references:
        label = SOURCE_LABELS.get(item.source, item.source)
        counts[label] = counts.get(label, 0) + 1
    parts = "，".join(f"{label} {count} 张" for label, count in counts.items())
    total = sum(item.size for item in references)
    return f"{len(references)} 张（{parts}，共 {total // 1024} KB）"
