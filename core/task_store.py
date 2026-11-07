"""Durable task record store (SQLite).

Task records back the queue and the WebUI task list. They also hold the full
prompt audit trail (prompt_layers + final_prompt) so every generation can be
inspected, which is the point of the no-hidden-prompts policy.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .logbook import clip, get_logger

logger = get_logger("task_store")


def _now() -> int:
    return int(time.time())


# -- task state machine -----------------------------------------------------

# queued -> running -> succeeded | failed | cancelled, plus queued -> cancelled.
# Finished records are final: nothing may move them back to an active state.
_STATUS_LABELS: dict[str, str] = {
    "queued": "排队中",
    "running": "运行中",
    "succeeded": "已成功",
    "partial": "部分完成",
    "failed": "已失败",
    "cancelled": "已取消",
}
_TERMINAL_STATES = frozenset({"succeeded", "partial", "failed", "cancelled"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "succeeded", "failed", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "cancelled"}),
    "succeeded": frozenset({"partial"}),
    "partial": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


def _validate_transition(current: str, target: str) -> None:
    """Raise ValueError when a status change is not allowed."""
    if target not in _STATUS_LABELS:
        raise ValueError(f"未知任务状态：{target}")
    if target == current:
        return
    allowed = _TRANSITIONS.get(current)
    if allowed is None:
        # Unknown stored value: do not trap rows forever.
        return
    if target in allowed:
        return
    label = _STATUS_LABELS[target]
    if current in _TERMINAL_STATES:
        raise ValueError(f"任务已结束，不能再改成{label}")
    raise ValueError(f"任务{_STATUS_LABELS.get(current, current)}，不能改成{label}")


def new_owner_id() -> str:
    """Return an instance id: ``aiimg-<pid>-<8 hex chars>``."""
    return f"aiimg-{os.getpid()}-{secrets.token_hex(4)}"


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
    owner: str = ""  # instance id currently executing the task
    lease_until: float = 0.0  # lease deadline, epoch seconds
    delivery_state: str = ""  # "" | sent | partial | failed | unknown
    delivery_attempts: int = 0
    attempts: int = 0  # how often the task has been claimed

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
            "owner": self.owner,
            "lease_until": self.lease_until,
            "attempts": self.attempts,
            "delivery_state": self.delivery_state,
            "delivery_attempts": self.delivery_attempts,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TaskRecord":
        keys = row.keys()
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
            owner=row["owner"] if "owner" in keys else "",
            lease_until=float(row["lease_until"]) if "lease_until" in keys else 0.0,
            attempts=int(row["attempts"]) if "attempts" in keys else 0,
            delivery_state=row["delivery_state"] if "delivery_state" in keys else "",
            delivery_attempts=(
                int(row["delivery_attempts"]) if "delivery_attempts" in keys else 0
            ),
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


def _serialized(method):
    """把同步的数据库读写串行化到一把线程锁上。

    后台清理走 asyncio.to_thread，会在另一个线程里用同一个连接，
    没有串行化就可能撞车；这里用可重入锁，同一线程里
    create→upsert、set_status→get+upsert 这类嵌套调用不会把自己锁死。
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._db_lock:
            return method(self, *args, **kwargs)

    return wrapper


class TaskStore:
    def __init__(self, data_dir: Path):
        self._path = Path(data_dir) / "tasks.sqlite3"
        # 线程锁要在连接之前建好：下面初始化 schema 时就会用到。
        self._db_lock = threading.RLock()
        # check_same_thread=False：后台清理由 asyncio.to_thread 在别的线程里
        # 调用 prune_old，默认的线程检查会直接抛 ProgrammingError；
        # 并发安全由上面的 _db_lock 串行化保证。
        self._db = sqlite3.connect(
            str(self._path), timeout=5.0, check_same_thread=False
        )
        self._db.row_factory = sqlite3.Row
        # Autocommit: every statement commits on its own, so a second process
        # opening the same file never waits on a transaction we forgot to close.
        # Explicit commit() calls stay valid (they become no-ops).
        self._db.isolation_level = None
        self._lock: asyncio.Lock | None = None
        self._configure()
        self._create_schema()
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_kind ON tasks(kind)"
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_tasks_lease ON tasks(status, lease_until)"
        )
        self._db.commit()
        logger.info("任务库已就绪：%s，未结束任务 %d 个", self._path, self.count_active())

    # -- setup --------------------------------------------------------------

    def _configure(self) -> None:
        """Make the file safe to share between several plugin instances."""
        # WAL: readers never block the writer and vice versa. Some exotic
        # filesystems refuse it; the store still works, just with more locking.
        try:
            self._db.execute("PRAGMA journal_mode=WAL").fetchone()
        except sqlite3.Error:
            pass
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA synchronous=NORMAL")

    def _create_schema(self) -> None:
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
                extra TEXT NOT NULL DEFAULT '{}',
                owner TEXT NOT NULL DEFAULT '',
                lease_until REAL NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0,
                delivery_state TEXT NOT NULL DEFAULT '',
                delivery_attempts INTEGER NOT NULL DEFAULT 0
            );
            -- 批量任务的每一张单独记一行，重试和查进度都按这一行来。
            CREATE TABLE IF NOT EXISTS task_items (
                task_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                prompt TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'queued',
                error TEXT NOT NULL DEFAULT '',
                image_ref TEXT NOT NULL DEFAULT '',
                updated_at INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (task_id, idx)
            )
            """
        )

    def _alock(self) -> asyncio.Lock:
        """Lazily create the asyncio lock (bound to the running loop)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @_serialized
    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass
        logger.debug("任务库已关闭：%s", self._path)

    @_serialized
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
        logger.info(
            "新建任务 %s：类型 %s，用户 %s，提供方 %s",
            task_id,
            record.kind,
            record.user_id or "-",
            record.provider or "-",
        )
        return record

    @_serialized
    def upsert(self, record: TaskRecord) -> None:
        self._db.execute(
            """
            INSERT INTO tasks (
                task_id, kind, sid, session_type, user_id, prompt, provider,
                status, result, error, created_at, started_at, finished_at,
                chain, prompt_layers, final_prompt, size, extra,
                owner, lease_until, attempts,
                delivery_state, delivery_attempts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                kind=excluded.kind, sid=excluded.sid,
                session_type=excluded.session_type, user_id=excluded.user_id,
                prompt=excluded.prompt, provider=excluded.provider,
                status=excluded.status, result=excluded.result,
                error=excluded.error, started_at=excluded.started_at,
                finished_at=excluded.finished_at, chain=excluded.chain,
                prompt_layers=excluded.prompt_layers,
                final_prompt=excluded.final_prompt, size=excluded.size,
                extra=excluded.extra, owner=excluded.owner,
                lease_until=excluded.lease_until, attempts=excluded.attempts,
                delivery_state=excluded.delivery_state,
                delivery_attempts=excluded.delivery_attempts
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
                _clean_text(record.owner),
                float(record.lease_until or 0.0),
                int(record.attempts or 0),
                str(record.delivery_state or ""),
                int(record.delivery_attempts or 0),
            ),
        )
        self._db.commit()
        # 每条状态都会走 upsert，这一步只留 debug，免得一行状态改出两条日志。
        logger.debug(
            "任务 %s 写库：状态 %s，尝试 %d 次，持有者 %s",
            record.task_id,
            record.status,
            int(record.attempts or 0),
            record.owner or "-",
        )

    @_serialized
    def get(self, task_id: str) -> TaskRecord | None:
        row = self._db.execute(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        ).fetchone()
        return TaskRecord.from_row(row) if row else None

    @_serialized
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

    @_serialized
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

    @_serialized
    def list_active(self) -> list[TaskRecord]:
        rows = self._db.execute(
            "SELECT * FROM tasks WHERE status IN ('queued', 'running') "
            "ORDER BY created_at ASC"
        ).fetchall()
        return [TaskRecord.from_row(row) for row in rows]

    @_serialized
    def set_status(
        self,
        task_id: str,
        status: str,
        *,
        provider: str = "",
        started_at: int = 0,
        finished_at: int = 0,
    ) -> None:
        """Persist a status change, refusing illegal transitions.

        Raises ValueError when the record is already finished, e.g.
        ``任务已结束，不能再改成运行中``.
        """
        record = self.get(task_id)
        if record is None:
            logger.debug("改状态跳过：任务 %s 不存在", task_id)
            return
        previous = record.status
        _validate_transition(previous, status)
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
        if status != previous:
            logger.info("任务 %s 状态改动：%s → %s", task_id, previous, status)

    @_serialized
    def set_result(self, task_id: str, result: str, *, provider: str = "") -> None:
        record = self.get(task_id)
        if record is None:
            logger.debug("写结果跳过：任务 %s 不存在", task_id)
            return
        record.result = result
        if provider:
            record.provider = provider
        record.status = "succeeded"
        record.finished_at = _now()
        self.upsert(record)
        logger.info("任务 %s 已成功，结果 %d 字符", task_id, len(record.result))

    @_serialized
    def set_error(self, task_id: str, error: str, *, provider: str = "") -> None:
        record = self.get(task_id)
        if record is None:
            logger.debug("写错误跳过：任务 %s 不存在", task_id)
            return
        record.error = _clean_text(error)
        if provider:
            record.provider = provider
        record.status = "failed"
        record.finished_at = _now()
        self.upsert(record)
        logger.info("任务 %s 已失败：%s", task_id, clip(record.error, 120) or "-")

    @_serialized
    def set_audit(
        self,
        task_id: str,
        *,
        prompt_layers: list[dict] | None = None,
        final_prompt: str = "",
    ) -> None:
        record = self.get(task_id)
        if record is None:
            logger.debug("写审计信息跳过：任务 %s 不存在", task_id)
            return
        if prompt_layers is not None:
            record.prompt_layers = list(prompt_layers)
        if final_prompt:
            record.final_prompt = final_prompt
        self.upsert(record)
        # 提示词正文不进日志，只记层数，避免日志变成第二份提示词存档。
        logger.debug(
            "任务 %s 已记录提示词分层 %d 层", task_id, len(record.prompt_layers)
        )

    @_serialized
    def set_delivery(self, task_id: str, state: str, attempts: int = 0) -> None:
        """Record how sending the result to the chat went.

        Kept out of ``status`` on purpose: generating an image and getting it
        into the chat are two different things, and the task page shows both.
        """
        self._db.execute(
            "UPDATE tasks SET delivery_state=?, delivery_attempts=? WHERE task_id=?",
            (str(state or ""), max(0, int(attempts or 0)), task_id),
        )
        self._db.commit()
        logger.debug(
            "任务 %s 投递结果：%s，尝试 %d 次", task_id, state or "-", int(attempts or 0)
        )

    # -- batch items --------------------------------------------------------

    @_serialized
    def seed_items(self, task_id: str, prompts: list[tuple[int, str]]) -> None:
        """Write one row per batch prompt so每张都有自己的状态和编号。"""
        rows = [
            (str(task_id), int(index), str(text or ""), "queued", "", "", _now())
            for index, text in prompts
            if str(text or "").strip()
        ]
        self._db.execute("DELETE FROM task_items WHERE task_id=?", (str(task_id),))
        self._db.executemany(
            "INSERT INTO task_items (task_id, idx, prompt, state, error, image_ref, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._db.commit()
        logger.info("任务 %s 登记批量子项 %d 张", task_id, len(rows))

    @_serialized
    def set_item(
        self,
        task_id: str,
        index: int,
        state: str,
        *,
        error: str = "",
        image_ref: str = "",
    ) -> None:
        self._db.execute(
            "UPDATE task_items SET state=?, error=?, image_ref=?, updated_at=?"
            " WHERE task_id=? AND idx=?",
            (
                str(state or ""),
                _clean_text(error),
                str(image_ref or ""),
                _now(),
                str(task_id),
                int(index),
            ),
        )
        self._db.commit()
        logger.debug(
            "任务 %s 第 %d 张：%s%s",
            task_id,
            int(index) + 1,
            state or "-",
            f"（{clip(error, 80)}）" if error else "",
        )

    @_serialized
    def list_items(self, task_id: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT idx, prompt, state, error, image_ref FROM task_items"
            " WHERE task_id=? ORDER BY idx",
            (str(task_id),),
        ).fetchall()
        return [
            {
                "index": int(row["idx"]),
                "prompt": str(row["prompt"] or ""),
                "state": str(row["state"] or ""),
                "error": str(row["error"] or ""),
                "image_ref": str(row["image_ref"] or ""),
            }
            for row in rows
        ]

    @_serialized
    def prune_items(self) -> int:
        """Drop item rows whose parent task is gone."""
        cursor = self._db.execute(
            "DELETE FROM task_items WHERE task_id NOT IN (SELECT task_id FROM tasks)"
        )
        self._db.commit()
        return int(cursor.rowcount or 0)

    @_serialized
    def cancel(self, task_id: str) -> bool:
        record = self.get(task_id)
        if record is None or record.status not in ("queued", "running"):
            logger.debug("取消未生效：任务 %s 不存在或已结束", task_id)
            return False
        record.status = "cancelled"
        record.finished_at = _now()
        self.upsert(record)
        logger.info("任务 %s 已取消", task_id)
        return True

    # -- owner leases -------------------------------------------------------
    #
    # A lease says "instance X is executing this task until epoch T". It lets
    # several plugin instances share one database: only the lease holder may
    # run a task, and a task whose lease expired is up for grabs again.
    # All timestamps go through time.time() so they survive a process restart.

    async def claim(self, task_id: str, owner: str, ttl_seconds: float) -> bool:
        """Atomically take ownership of a queued or abandoned task.

        Succeeds only when the task is queued, or running with an expired
        lease / no owner. On success the owner and lease are written, the
        attempt counter is bumped and the status becomes running.
        """
        if not owner:
            raise ValueError("owner 不能为空")
        async with self._alock():
            # 线程锁只包住同步的数据库操作，持锁期间不 await
            with self._db_lock:
                now = time.time()
                deadline = now + max(0.0, float(ttl_seconds))
                cursor = self._db.execute(
                    """
                    UPDATE tasks SET
                        owner = ?,
                        lease_until = ?,
                        attempts = attempts + 1,
                        status = CASE WHEN status = 'queued' THEN 'running'
                                      ELSE status END,
                        started_at = CASE WHEN started_at = 0 THEN ? ELSE started_at END
                    WHERE task_id = ?
                      AND (status = 'queued'
                           OR (status = 'running'
                               AND (owner = '' OR lease_until <= ?)))
                    """,
                    (owner, deadline, int(now), task_id, now),
                )
                self._db.commit()
                won = cursor.rowcount > 0
        if won:
            logger.info(
                "任务 %s 租约已获取：持有者 %s，租约 %d 秒",
                task_id,
                owner,
                int(ttl_seconds),
            )
        else:
            logger.debug("任务 %s 没抢到租约：已被其他实例持有或已结束", task_id)
        return won

    async def heartbeat(
        self, task_id: str, owner: str, ttl_seconds: float
    ) -> bool:
        """Renew the lease of the current holder; False if not the holder."""
        if not owner:
            return False
        async with self._alock():
            with self._db_lock:
                deadline = time.time() + max(0.0, float(ttl_seconds))
                cursor = self._db.execute(
                    "UPDATE tasks SET lease_until = ? "
                    "WHERE task_id = ? AND owner = ? "
                    "AND status IN ('queued', 'running')",
                    (deadline, task_id, owner),
                )
                self._db.commit()
                held = cursor.rowcount > 0
        # 续租每隔十几秒一次，只留 debug；队列那边会在第一次续租时说一句。
        logger.debug(
            "任务 %s 续租%s：租约 %d 秒",
            task_id,
            "成功" if held else "未生效",
            int(ttl_seconds),
        )
        return held

    async def release(self, task_id: str, owner: str) -> None:
        """Drop the lease; a non-holder is a silent no-op."""
        if not owner:
            return
        async with self._alock():
            with self._db_lock:
                cursor = self._db.execute(
                    "UPDATE tasks SET owner = '', lease_until = 0 "
                    "WHERE task_id = ? AND owner = ?",
                    (task_id, owner),
                )
                self._db.commit()
                dropped = cursor.rowcount > 0
        if dropped:
            logger.info("任务 %s 的租约已释放", task_id)
        else:
            logger.debug("任务 %s 的租约不在 %s 手上，无需释放", task_id, owner)

    async def stale_running(self, now: float | None = None) -> list[TaskRecord]:
        """Running tasks whose lease expired, oldest first."""
        moment = time.time() if now is None else float(now)
        async with self._alock():
            with self._db_lock:
                rows = self._db.execute(
                    "SELECT * FROM tasks WHERE status = 'running' "
                    "AND lease_until <= ? "
                    "ORDER BY created_at ASC, rowid ASC",
                    (moment,),
                ).fetchall()
        records = [TaskRecord.from_row(row) for row in rows]
        if records:
            logger.info("发现 %d 个租约已过期的运行中任务", len(records))
        return records

    async def bump_attempts(self, task_id: str) -> int:
        """Increment attempts and return the new value (0 for unknown tasks)."""
        async with self._alock():
            with self._db_lock:
                cursor = self._db.execute(
                    "UPDATE tasks SET attempts = attempts + 1 WHERE task_id = ?",
                    (task_id,),
                )
                if cursor.rowcount <= 0:
                    return 0
                row = self._db.execute(
                    "SELECT attempts FROM tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                attempts = int(row["attempts"]) if row else 0
        logger.debug("任务 %s 尝试次数 +1，现为 %d 次", task_id, attempts)
        return attempts

    @_serialized
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

    @_serialized
    def count_active_by_kind(self, kind: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE kind = ? "
            "AND status IN ('queued', 'running')",
            (kind,),
        ).fetchone()
        return int(row["n"] if row else 0)

    @_serialized
    def count_active(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE status IN ('queued', 'running')"
        ).fetchone()
        return int(row["n"] if row else 0)

    @_serialized
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
        removed = cur.rowcount
        # 清理是定时动作，只有真的删了东西才留一行。
        if removed:
            logger.info("清理过期任务记录 %d 条（保留 %d 小时）", removed, retention_hours)
        return removed
