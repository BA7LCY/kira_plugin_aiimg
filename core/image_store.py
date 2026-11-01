"""Gallery storage for generated media.

Persistence lives under the plugin data dir. The sqlite table is metadata
only; image bytes are plain files under `gallery/`. Favorites are excluded
from retention and capacity cleanup. A tiny per-session "last images" ring
supports the edit/selfie convenience of referencing the most recent image.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .image_format import guess_image_mime_and_ext, preview_jpeg, thumbnail_webp
from .logbook import get_logger

logger = get_logger("gallery")


class ImageStore:
    """File + sqlite gallery with retention rules and last-image memory."""

    def __init__(self, data_dir: Path, config: dict):
        self.data_dir = Path(data_dir)
        self.gallery_dir = self.data_dir / "gallery"
        self.gallery_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir = self.data_dir / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir = self.data_dir / "previews"
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        self._config = config or {}
        self._lock = threading.Lock()
        self._db = sqlite3.connect(
            self.data_dir / "gallery.sqlite3",
            check_same_thread=False,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'image',
                created_at REAL NOT NULL,
                favorite INTEGER NOT NULL DEFAULT 0,
                archived INTEGER NOT NULL DEFAULT 0,
                provider TEXT NOT NULL DEFAULT '',
                model TEXT NOT NULL DEFAULT '',
                size TEXT NOT NULL DEFAULT '',
                sid TEXT NOT NULL DEFAULT '',
                task_id TEXT NOT NULL DEFAULT '',
                prompt TEXT NOT NULL DEFAULT '',
                bytes INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        for column, decl in (("prompt", "TEXT NOT NULL DEFAULT ''"), ("bytes", "INTEGER NOT NULL DEFAULT 0")):
            try:
                self._db.execute(f"ALTER TABLE images ADD COLUMN {column} {decl}")
            except sqlite3.OperationalError:
                pass
        for statement in (
            "CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_images_favorite ON images(favorite)",
            "CREATE INDEX IF NOT EXISTS idx_images_kind ON images(kind)",
            "CREATE INDEX IF NOT EXISTS idx_images_archived ON images(archived, created_at)",
        ):
            self._db.execute(statement)
        self._db.commit()
        self._last_images: dict[str, list[str]] = {}

    # -- persistence ---------------------------------------------------------

    def update_config(self, config: dict) -> None:
        """Swap retention/capacity rules after a settings change."""
        self._config = config or {}

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._db.execute(sql, params)
            self._db.commit()
            return cursor

    async def save_image(
        self,
        data: bytes,
        *,
        provider_id: str,
        model: str,
        size: str,
        sid: str,
        task_id: str,
        kind: str = "image",
        prompt: str = "",
    ) -> Path:
        """Persist bytes into the gallery and return the stored path."""
        mime, ext = guess_image_mime_and_ext(data)
        image_id = uuid.uuid4().hex[:16]
        created = time.time()
        filename = f"{int(created)}_{image_id}.{ext}"
        path = self.gallery_dir / filename
        with open(path, "wb") as fh:
            fh.write(data)
        if kind == "image":
            await asyncio.to_thread(self.write_derivatives, image_id, data)
        self._execute(
            """
            INSERT INTO images
                (id, path, kind, created_at, favorite, archived,
                 provider, model, size, sid, task_id, prompt, bytes)
            VALUES (?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                image_id,
                str(path),
                kind,
                created,
                provider_id,
                model,
                size,
                sid,
                task_id,
                str(prompt or "")[:2000],
                len(data),
            ),
        )
        await self.cleanup()
        self._remember(sid, str(path))
        logger.info(
            "[aiimg] 画廊入库 %s：%s %s，%dB，任务 %s",
            filename, provider_id or "-", size or "-", len(data), task_id or "-",
        )
        return path

    async def register_video(
        self, path: Path, *, provider_id: str, model: str, size: str,
        sid: str, task_id: str, prompt: str = "",
    ) -> Path:
        """Register a fully downloaded video under normal gallery retention."""
        path = Path(path).resolve()
        if path.parent != self.gallery_dir.resolve() or not path.is_file():
            raise ValueError("视频文件不在画廊目录中")
        image_id = uuid.uuid4().hex[:16]
        self._execute(
            "INSERT INTO images "
            "(id, path, kind, created_at, provider, model, size, sid, task_id, prompt, bytes) "
            "VALUES (?, ?, 'video', ?, ?, ?, ?, ?, ?, ?, ?)",
            (image_id, str(path), time.time(), provider_id, model, size, sid,
             task_id, str(prompt or "")[:2000], path.stat().st_size),
        )
        await self.cleanup()
        return path

    def get(self, image_id: str) -> dict | None:
        row = self._execute(
            "SELECT * FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_gallery(
        self,
        *,
        include_archived: bool = False,
        offset: int = 0,
        limit: int = 0,
    ) -> list[dict]:
        where = "" if include_archived else " WHERE archived = 0"
        sql = f"SELECT * FROM images{where} ORDER BY created_at DESC"
        params: tuple = ()
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            params = (int(limit), max(0, int(offset)))
        rows = self._execute(sql, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def search_gallery(
        self,
        *,
        offset: int = 0,
        limit: int = 0,
        query: str = "",
        kind: str = "",
        favorite: bool | None = None,
        sort: str = "new",
        include_archived: bool = False,
    ) -> tuple[list[dict], int]:
        """Filtered gallery page plus the matching total.

        `query` matches the prompt with LIKE; wildcards are escaped so a
        literal `%` or `_` does not widen the search.
        """
        where, args = self._search_clauses(
            include_archived=include_archived,
            query=query,
            kind=kind,
            favorite=favorite,
        )
        order = "created_at ASC, rowid ASC" if str(sort).lower() == "old" else (
            "created_at DESC, rowid DESC"
        )
        sql = f"SELECT * FROM images{where} ORDER BY {order}"
        page_args = list(args)
        if limit > 0:
            sql += " LIMIT ? OFFSET ?"
            page_args += [int(limit), max(0, int(offset))]
        rows = self._execute(sql, tuple(page_args)).fetchall()
        counted = self._execute(
            f"SELECT COUNT(*) AS n FROM images{where}", tuple(args)
        ).fetchone()
        total = int(counted["n"]) if counted else 0
        return [self._row_to_dict(row) for row in rows], total

    def _search_clauses(
        self,
        *,
        include_archived: bool,
        query: str,
        kind: str,
        favorite: bool | None,
    ) -> tuple[str, list]:
        clauses: list[str] = []
        args: list = []
        if not include_archived:
            clauses.append("archived = 0")
        name = str(kind or "").strip().lower()
        if name:
            clauses.append("kind = ?")
            args.append(name)
        if favorite is not None:
            clauses.append("favorite = ?")
            args.append(1 if favorite else 0)
        text = str(query or "").strip()
        if text:
            clauses.append("prompt LIKE ? ESCAPE '\\'")
            args.append(f"%{_escape_like(text)}%")
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, args

    def detail(self, image_id: str) -> dict | None:
        """Full metadata for one image, with the on-disk size when present."""
        record = self.get(image_id)
        if record is None:
            return None
        path = Path(str(record.get("path") or ""))
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = int(record.get("bytes") or 0)
        record["bytes"] = size
        record["filename"] = path.name
        record["exists"] = path.is_file()
        return record

    def delete_many(self, ids: list[str]) -> dict:
        """Delete several images in one transaction; favorites are skipped."""
        wanted = _clean_ids(ids)
        if not wanted:
            return {"deleted": 0, "skipped": 0}
        marks = ",".join("?" for _ in wanted)
        with self._lock:
            rows = self._db.execute(
                f"SELECT id, path, favorite FROM images WHERE id IN ({marks})",
                wanted,
            ).fetchall()
            removable = [row for row in rows if not row["favorite"]]
            if removable:
                doomed = [row["id"] for row in removable]
                spots = ",".join("?" for _ in doomed)
                self._db.execute(
                    f"DELETE FROM images WHERE id IN ({spots})", doomed
                )
            self._db.commit()
        for row in removable:
            self._remove_files(row["id"], row["path"])
        return {"deleted": len(removable), "skipped": len(rows) - len(removable)}

    def favorite_many(self, ids: list[str], favorite: bool) -> int:
        """Set the favorite flag on several images in one transaction."""
        wanted = _clean_ids(ids)
        if not wanted:
            return 0
        marks = ",".join("?" for _ in wanted)
        cursor = self._execute(
            f"UPDATE images SET favorite = ? WHERE id IN ({marks})",
            (1 if favorite else 0, *wanted),
        )
        return cursor.rowcount

    def thumb_path(self, image_id: str) -> Path:
        safe = "".join(ch for ch in str(image_id) if ch.isalnum() or ch in "-_")
        return self.thumb_dir / f"{safe or 'image'}.webp"

    def preview_path(self, image_id: str) -> Path:
        safe = "".join(ch for ch in str(image_id) if ch.isalnum() or ch in "-_")
        return self.preview_dir / f"{safe or 'image'}.jpg"

    def write_derivatives(self, image_id: str, data: bytes) -> None:
        """Write the small grid thumb and the mid-size preview beside the id."""
        try:
            thumb = thumbnail_webp(data)
            if thumb is not data and thumb:
                self._atomic_write(self.thumb_path(image_id), thumb)
        except Exception:
            pass
        try:
            preview = preview_jpeg(data)
            if preview is not data and preview:
                self._atomic_write(self.preview_path(image_id), preview)
        except Exception:
            pass

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        temp = path.with_name(f".{uuid.uuid4().hex}.tmp")
        try:
            temp.write_bytes(data)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)

    def stats(self) -> dict:
        row = self._execute(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN favorite = 1 THEN 1 ELSE 0 END) AS favorites, "
            "COALESCE(SUM(bytes), 0) AS bytes "
            "FROM images WHERE archived = 0"
        ).fetchone()
        total = int(row["total"]) if row else 0
        favorites = int(row["favorites"]) if row and row["favorites"] is not None else 0
        total_bytes = int(row["bytes"]) if row and row["bytes"] is not None else 0
        return {"count": total, "favorites": favorites, "bytes": total_bytes}

    def set_favorite(self, image_id: str, favorite: bool) -> bool:
        cursor = self._execute(
            "UPDATE images SET favorite = ? WHERE id = ?",
            (1 if favorite else 0, image_id),
        )
        return cursor.rowcount > 0

    def _delete_row(self, image_id: str, path: str) -> None:
        self._execute("DELETE FROM images WHERE id = ?", (image_id,))
        self._remove_files(image_id, path)

    def _remove_files(self, image_id: str, path: str) -> None:
        for target in (path, self.thumb_path(image_id), self.preview_path(image_id)):
            try:
                os.remove(target)
            except OSError:
                pass

    def delete(self, image_id: str) -> bool:
        row = self._execute(
            "SELECT path, favorite FROM images WHERE id = ?", (image_id,)
        ).fetchone()
        if not row:
            return False
        path, favorite = row
        if favorite:
            return False
        self._delete_row(image_id, path)
        return True

    async def cleanup(self) -> None:
        """Apply the time rule first, then the capacity rule, per bucket.

        Images, selfies and videos each have their own budget, so a selfie
        never displaces a normal image and a video never counts as one.
        Favorites are excluded from both rules.
        """
        gallery = self._config.get("gallery", {}) if isinstance(
            self._config.get("gallery"), dict
        ) else {}
        buckets = (
            ("图片", ("image", "frame"), "retention_days", "max_count"),
            ("自拍", ("selfie",), "selfie_retention_days", "selfie_max_count"),
            ("视频", ("video",), "video_retention_days", "video_max_count"),
        )
        for label, kinds, days_key, count_key in buckets:
            retention_days = _as_int(gallery.get(days_key), 0)
            max_count = _as_int(gallery.get(count_key), 0)
            if retention_days <= 0 and max_count <= 0:
                continue
            placeholders = ", ".join("?" for _ in kinds)
            if retention_days > 0:
                cutoff = time.time() - retention_days * 86400
                rows = self._execute(
                    f"SELECT id, path FROM images WHERE favorite = 0 "
                    f"AND kind IN ({placeholders}) AND created_at < ?",
                    (*kinds, cutoff),
                ).fetchall()
                for image_id, path in rows:
                    self._delete_row(image_id, path)
                if rows:
                    logger.info(
                        "[aiimg] 画廊清理：%d 个%s超过 %d 天，已删除",
                        len(rows), label, retention_days,
                    )
            if max_count > 0:
                rows = self._execute(
                    f"SELECT id, path FROM images WHERE favorite = 0 "
                    f"AND kind IN ({placeholders}) "
                    f"ORDER BY created_at DESC, rowid DESC",
                    kinds,
                ).fetchall()
                overflow = rows[max_count:]
                for image_id, path in overflow:
                    self._delete_row(image_id, path)
                if overflow:
                    logger.info(
                        "[aiimg] 画廊清理：%s超出 %d 个上限，删掉最早 %d 个",
                        label, max_count, len(overflow),
                    )

    # -- last-image memory ---------------------------------------------------

    def _remember(self, sid: str, path: str) -> None:
        if not sid:
            return
        ring = self._last_images.setdefault(sid, [])
        if path in ring:
            ring.remove(path)
        ring.insert(0, path)
        del ring[16:]

    def recent_images(self, sid: str) -> list[str]:
        return list(self._last_images.get(sid, []))

    @staticmethod
    def _row_to_dict(row: sqlite3.Row | tuple) -> dict:
        names = (
            "id",
            "path",
            "kind",
            "created_at",
            "favorite",
            "archived",
            "provider",
            "model",
            "size",
            "sid",
            "task_id",
            "prompt",
            "bytes",
        )
        return dict(zip(names, row))

    async def close(self) -> None:
        with self._lock:
            self._db.close()


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _escape_like(text: str) -> str:
    """Escape LIKE wildcards so user input matches literally."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _clean_ids(ids) -> list[str]:
    """Deduplicate ids and keep the IN() clause bounded."""
    cleaned: list[str] = []
    for item in ids if isinstance(ids, (list, tuple)) else []:
        text = str(item).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned[:200]
