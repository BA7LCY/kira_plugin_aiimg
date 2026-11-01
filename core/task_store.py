"""Durable task record store (SQLite).

Task records back the queue and the WebUI task list. They also hold the full
prompt audit trail (prompt_layers + final_prompt) so every generation can be
inspected, which is the point of the no-hidden-prompts policy.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _now() -> int:
    return int(time.time())


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text[:8000]


@dataclass
class TaskRecord:
    task_id: str
    kind: str  # generate_image | edit_image | generate_selfie | generate_image_batch | generate_video
    sid: str = ""
    session_type: str = ""
    user_id: str = ""
    prompt: str = ""
    provider: str = ""
    status: str = "queued"  # queued | running | succeeded | failed | cancelled
    result: str = ""  # JSON: media path/url list + caption
    error: str = ""
    created_at: int = field(default_factory=_now)
    started_at: int = 0
    finished_at: int = 0
    chain: list[str] = field(default_factory=list)
    prompt_layers: list[dict] = field(default_factory=list)
    final_prompt: str = ""
    size: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "sid": self.sid,
            "session_type": self.session_type,
            "user_id": self.user_id,
            "prompt": self.prompt,
            "provider": self.provider,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "chain": self.chain,
            "prompt_layers": self.prompt_layers,
            "final_prompt": self.final_prompt,
            "size": self.size,
            "extra": self.extra,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TaskRecord":
        return cls(
            task_id=row["task_id"],
            kind=row["kind"],
            sid=row["sid"],
            session_type=row["session_type"],
            user_id=row["user_id"],
            prompt=row["prompt"],
            provider=row["provider"],
            status=row["status"],
            result=row["result"],
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            chain=json.loads(row["chain"] or "[]"),
            prompt_layers=json.loads(row["prompt_layers"] or "[]"),
            final_prompt=row["final_prompt"],
            size=row["size"],
            extra=json.loads(row["extra"] or "{}"),
        )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# Task page filters map onto the statuses stored by the queue.
_STATE_ALIASES: dict[str, tuple[str, ...]] = {
    "running": ("queued", "running"),
    "active": ("queued", "running"),
    "succeeded": ("succeeded",),
    "done": ("succeeded",),
    "failed": ("failed",),
    "cancelled": ("cancelled",),
}


class TaskStore:
    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "tasks.sqlite3"
        self._db = sqlite3.connect(str(self._path))
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                sid TEXT NOT NULL DEFAULT '',
                session_type TEXT NOT NULL DEFAULT '',
                user_id TEXT NOT NULL DEFAULT '',
                prompt TEXT NOT NULL DEFAULT '',
                provider TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                result TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL,
                started_at INTEGER NOT NULL DEFAULT 0,
                finished_at INTEGER NOT NULL DEFAULT 0,
                chain TEXT NOT NULL DEFAULT '[]',
                prompt_layers TEXT NOT NULL DEFAULT '[]',
                final_prompt TEXT NOT NULL DEFAULT '',
                size TEXT NOT NULL DEFAULT '',
                extra TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_kind ON tasks(kind)"
        )
        self._db.commit()

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass

    def create(
        self,
        *,
        kind: str,
        sid: str = "",
        session_type: str = "",
        user_id: str = "",
        prompt: str = "",
        provider: str = "",
        chain: list[str] | None = None,
        size: str = "",
        extra: dict | None = None,
    ) -> TaskRecord:
        task_id = uuid.uuid4().hex[:16]
        record = TaskRecord(
            task_id=task_id,
            kind=kind,
            sid=_clean_text(sid),
            session_type=_clean_text(session_type),
            user_id=_clean_text(user_id),
            prompt=_clean_text(prompt),
            provider=_clean_text(provider),
            chain=list(chain or []),
            size=_clean_text(size),
            extra=dict(extra or {}),
        )
        self.upsert(record)
        return record

    def upsert(self, record: TaskRecord) -> None:
        self._db.execute(
            """
            INSERT INTO tasks (
                task_id, kind, sid, session_type, user_id, prompt, provider,
                status, result, error, created_at, started_at, finished_at,
                chain, prompt_layers, final_prompt, size, extra
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                kind=excluded.kind, sid=excluded.sid,
                session_type=excluded.session_type, user_id=excluded.user_id,
                prompt=excluded.prompt, provider=excluded.provider,
                status=excluded.status, result=excluded.result,
                error=excluded.error, started_at=excluded.started_at,
                finished_at=excluded.finished_at, chain=excluded.chain,
                prompt_layers=excluded.prompt_layers,
                final_prompt=excluded.final_prompt, size=excluded.size,
                extra=excluded.extra
            """,
            (
                record.task_id,
                record.kind,
                record.sid,
                record.session_type,
                record.user_id,
                record.prompt,
                record.provider,
                record.status,
                record.result,
                record.error,
                record.created_at,
                record.started_at,
                record.finished_at,
                _json_dumps(record.chain),
                _json_dumps(record.prompt_layers),
                _clean_text(record.final_prompt),
                record.size,
                _json_dumps(record.extra),
            ),
        )
        self._db.commit()

    def get(self, task_id: str) -> TaskRecord | None:
        row = self._db.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return TaskRecord.from_row(row) if row else None

    def list_recent(
        self,
        *,
        kind: str = "",
        status: str = "",
        limit: int = 100,
    ) -> list[TaskRecord]:
        clauses: list[str] = []
        args: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if status:
            clauses.append("status = ?")
            args.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM tasks {where} ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (*args, int(limit)),
        ).fetchall()
        return [TaskRecord.from_row(row) for row in rows]

    def search_tasks(
        self,
        *,
        state: str = "",
        kind: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TaskRecord], int]:
        """Filtered task list plus the matching total.

        `state` accepts UI-friendly aliases: running covers queued+running.
        """
        clauses: list[str] = []
        args: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        wanted = str(state or "").strip().lower()
        if wanted and wanted != "all":
            states = _STATE_ALIASES.get(wanted, (wanted,))
            spots = ",".join("?" for _ in states)
            clauses.append(f"status IN ({spots})")
            args.extend(states)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._db.execute(
            f"SELECT * FROM tasks {where} "
            "ORDER BY created_at DESC, rowid DESC LIMIT ? OFFSET ?",
            (*args, max(1, int(limit)), max(0, int(offset))),
        ).fetchall()
        counted = self._db.execute(
            f"SELECT COUNT(*) AS n FROM tasks {where}", tuple(args)
        ).fetchone()
        return [TaskRecord.from_row(row) for row in rows], int(counted["n"])

    def list_active(self) -> list[TaskRecord]:
        rows = self._db.execute(
            "SELECT * FROM tasks WHERE status IN ('queued', 'running') "
            "ORDER BY created_at ASC"
        ).fetchall()
        return [TaskRecord.from_row(row) for row in rows]

    def set_status(
        self,
        task_id: str,
        status: str,
        *,
        provider: str = "",
        started_at: int = 0,
        finished_at: int = 0,
    ) -> None:
        record = self.get(task_id)
        if record is None:
            return
        record.status = status
        if provider:
            record.provider = provider
        if started_at:
            record.started_at = started_at
        elif status == "running" and not record.started_at:
            record.started_at = _now()
        if finished_at:
            record.finished_at = finished_at
        elif status in ("succeeded", "failed", "cancelled") and not record.finished_at:
            record.finished_at = _now()
        self.upsert(record)

    def set_result(self, task_id: str, result: str, *, provider: str = "") -> None:
        record = self.get(task_id)
        if record is None:
            return
        record.result = result
        if provider:
            record.provider = provider
        record.status = "succeeded"
        record.finished_at = _now()
        self.upsert(record)

    def set_error(self, task_id: str, error: str, *, provider: str = "") -> None:
        record = self.get(task_id)
        if record is None:
            return
        record.error = _clean_text(error)
        if provider:
            record.provider = provider
        record.status = "failed"
        record.finished_at = _now()
        self.upsert(record)

    def set_audit(
        self,
        task_id: str,
        *,
        prompt_layers: list[dict] | None = None,
        final_prompt: str = "",
    ) -> None:
        record = self.get(task_id)
        if record is None:
            return
        if prompt_layers is not None:
            record.prompt_layers = list(prompt_layers)
        if final_prompt:
            record.final_prompt = final_prompt
        self.upsert(record)

    def cancel(self, task_id: str) -> bool:
        record = self.get(task_id)
        if record is None or record.status not in ("queued", "running"):
            return False
        record.status = "cancelled"
        record.finished_at = _now()
        self.upsert(record)
        return True

    def count_active_by_user(
        self, user_id: str, kind: str = "", exclude_task_id: str = ""
    ) -> int:
        clauses = ["user_id = ?", "status IN ('queued', 'running')"]
        args: list[Any] = [user_id]
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if exclude_task_id:
            clauses.append("task_id != ?")
            args.append(exclude_task_id)
        row = self._db.execute(
            f"SELECT COUNT(*) AS n FROM tasks WHERE {' AND '.join(clauses)}",
            tuple(args),
        ).fetchone()
        return int(row["n"] if row else 0)

    def count_active_by_kind(self, kind: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE kind = ? "
            "AND status IN ('queued', 'running')",
            (kind,),
        ).fetchone()
        return int(row["n"] if row else 0)

    def count_active(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(row["n"] if row else 0)

    def prune_old(self, retention_hours: int) -> int:
        """Delete finished records older than retention_hours (0 = keep all)."""
        if not retention_hours:
            return 0
        cutoff = _now() - int(retention_hours) * 3600
        cur = self._db.execute(
            "DELETE FROM tasks WHERE status != 'queued' AND status != 'running' "
            "AND finished_at != 0 AND finished_at < ?",
            (cutoff,),
        )
        self._db.commit()
        return cur.rowcount
