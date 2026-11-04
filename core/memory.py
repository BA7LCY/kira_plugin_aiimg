"""图片记忆库：为生成过的图片登记短引用，按会话隔离并支持检索复用。

设计要点：
1. 独立的 ``image_refs`` 表，可以和画廊库共用同一个数据库文件，互不影响。
2. ``ref`` 是给提示词/模型看的短编号（``img-`` + 6 位小写十六进制），
   同一路径只保留一条记录，重复登记时复用原 ref，避免记忆库膨胀。
3. 每次写操作只开一次事务、只提交一次，不逐条提交。
4. 查询默认按 ``sid`` 会话隔离：只看得到本会话登记的图片；
   跨会话检索必须显式传 ``scoped=False``。
5. 文件失效（记录还在、磁盘上没了）只在解析时告警，不删记录，
   把删除权留给调用方或 ``prune``，避免误删仍被引用的行。
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

logger = logging.getLogger(__name__)

__all__ = ["ImageMemory", "REF_PATTERN", "DEFAULT_DB_NAME"]

# 调用方只给数据目录时，落在这个文件名上
DEFAULT_DB_NAME = "image_memory.sqlite3"
# ref 形态：img- 加上 6 位小写十六进制
REF_PATTERN = re.compile(r"^img-[0-9a-f]{6}$")
# ref 撞车后的重试次数；6 位十六进制有 1600 万个组合，正常一次就够
_REF_RETRIES = 16
# 单次查询返回条数上限，防止调用方一句 limit=0 之外又传个离谱大数
_MAX_LIMIT = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_refs (
    ref TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'image',
    sid TEXT NOT NULL DEFAULT '',
    task_id TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    uses INTEGER NOT NULL DEFAULT 0
)
"""

_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_image_refs_path ON image_refs(path)",
    "CREATE INDEX IF NOT EXISTS idx_image_refs_sid ON image_refs(sid, created_at)",
)


def _random_ref() -> str:
    """生成一个候选 ref：``img-`` 加 6 位小写十六进制。"""
    return f"img-{secrets.token_hex(3)}"


def _normalize_path(value: str | Path) -> str:
    """把路径规范成绝对、无多余分隔符的形式，保证同一文件只入库一次。"""
    text = os.fspath(value) if value is not None else ""
    if not text:
        return ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(text)))


def _resolve_db_path(value: str | Path | None) -> Path:
    """确定数据库文件路径：没有就给默认文件名，给的是目录就补默认文件名。"""
    if value is None or str(value).strip() == "":
        value = DEFAULT_DB_NAME
    path = Path(os.path.expanduser(os.fspath(value)))
    if path.exists() and path.is_dir():
        path = path / DEFAULT_DB_NAME
    return path.absolute()


def _clamp_limit(value: int, default: int) -> int:
    """把 limit 收进 0.._MAX_LIMIT；非法值退回默认值。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(number, _MAX_LIMIT))


def _like_pattern(text: str) -> str:
    """构造 LIKE 模式，转义 ``%``、``_`` 和反斜杠，让它们只按字面匹配。"""
    escaped = (
        text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    return f"%{escaped}%"


class ImageMemory:
    """图片记忆库：登记图片路径、按 ref 解析、按会话检索与清理。"""

    def __init__(self, db_path: str | Path = DEFAULT_DB_NAME) -> None:
        self.db_path = _resolve_db_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # 插件里读写可能来自不同线程，连接开 check_same_thread=False，
        # 再用 RLock 串行化所有访问，避免同一个连接被并发使用
        self._lock = threading.RLock()
        self._db: sqlite3.Connection | None = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # 关掉隐式事务，改由 _transaction 显式控制
            timeout=5.0,
        )
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA busy_timeout = 5000")
        with self._transaction() as db:
            db.execute(_SCHEMA)
            for statement in _INDEXES:
                db.execute(statement)
        logger.info("[aiimg] 图片记忆：库已就绪，数据库 %s", self.db_path)

    # -- 基础设施 -----------------------------------------------------------

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """一次事务一次提交：正常结束提交，中途出错整体回滚。"""
        with self._lock:
            db = self._require_db()
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                try:
                    db.rollback()
                except sqlite3.Error:
                    logger.warning(
                        "[aiimg] 图片记忆：事务回滚失败，连接可能已不可用", exc_info=True
                    )
                raise
            else:
                db.commit()

    def _require_db(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("图片记忆库已关闭，无法继续使用")
        return self._db

    def close(self) -> None:
        """关闭连接；重复调用无副作用。"""
        with self._lock:
            db, self._db = self._db, None
        if db is None:
            return
        try:
            db.rollback()  # 正常路径下没有残留事务，这里只是兜底
        except sqlite3.Error:
            pass
        try:
            db.close()
        except sqlite3.Error:
            logger.warning("[aiimg] 图片记忆：关闭连接时出错，数据库 %s", self.db_path)
        logger.debug("[aiimg] 图片记忆：连接已关闭，数据库 %s", self.db_path)

    def __enter__(self) -> "ImageMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- 登记 ---------------------------------------------------------------

    def remember(
        self,
        paths: Sequence[str | Path],
        *,
        sid: str = "",
        kind: str = "image",
        task_id: str = "",
        prompt: str = "",
    ) -> list[str]:
        """登记一批图片路径，返回与入参一一对应的 ref 列表。

        同一路径重复登记不会新增行，而是复用原 ref，并用本次传入的非空
        字段覆盖旧值（``sid``/``kind``/``task_id``/``prompt``）；传空串表示
        "这一项不改"，这样补登记时不会把已有的提示词冲掉。
        """
        if paths is None:
            return []
        if isinstance(paths, (str, Path)):
            # 容错：调用方误传单个路径时当成只有一条，不至于逐字符拆开
            paths = [paths]
        cleaned = [path for path in (_normalize_path(item) for item in paths) if path]
        if not cleaned:
            return []
        wanted_sid = str(sid or "")
        wanted_kind = str(kind or "").strip() or "image"
        wanted_task = str(task_id or "")
        wanted_prompt = str(prompt or "")

        refs: list[str] = []
        created = 0
        reused = 0
        with self._transaction() as db:
            for path in cleaned:
                row = db.execute(
                    "SELECT ref, kind, sid, task_id, prompt FROM image_refs"
                    " WHERE path = ? ORDER BY rowid LIMIT 1",
                    (path,),
                ).fetchone()
                if row is not None:
                    # 老记录：只更新这次确实传了值的字段
                    updates: dict[str, str] = {}
                    for column, value in (
                        ("sid", wanted_sid),
                        ("kind", wanted_kind),
                        ("task_id", wanted_task),
                        ("prompt", wanted_prompt),
                    ):
                        if value and value != str(row[column]):
                            updates[column] = value
                    if updates:
                        assignments = ", ".join(f"{name} = ?" for name in updates)
                        db.execute(
                            f"UPDATE image_refs SET {assignments} WHERE ref = ?",
                            (*updates.values(), row["ref"]),
                        )
                    refs.append(str(row["ref"]))
                    reused += 1
                    continue
                refs.append(
                    self._insert_new(
                        db,
                        path,
                        kind=wanted_kind,
                        sid=wanted_sid,
                        task_id=wanted_task,
                        prompt=wanted_prompt,
                    )
                )
                created += 1
        logger.info(
            "[aiimg] 图片记忆：登记 %d 张，会话 sid=%s（新增 %d，复用 %d）",
            len(refs),
            wanted_sid or "-",
            created,
            reused,
        )
        return refs

    def _insert_new(
        self,
        db: sqlite3.Connection,
        path: str,
        *,
        kind: str,
        sid: str,
        task_id: str,
        prompt: str,
    ) -> str:
        """插入一条新记录；ref 撞车就换一个重试。"""
        created_at = time.time()
        for _ in range(_REF_RETRIES):
            ref = _random_ref()
            try:
                db.execute(
                    "INSERT INTO image_refs"
                    " (ref, path, kind, sid, task_id, prompt, created_at, uses)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
                    (ref, path, kind, sid, task_id, prompt, created_at),
                )
            except sqlite3.IntegrityError:
                # 主键撞车（极小概率），换一个候选继续；语句级失败不影响事务
                logger.debug("[aiimg] 图片记忆：ref %s 已存在，换一个重试", ref)
                continue
            return ref
        raise RuntimeError("图片记忆：多次生成唯一 ref 失败，可能是随机源异常")

    # -- 解析 ---------------------------------------------------------------

    def resolve(self, ref: str, *, sid: str = "", scoped: bool = True) -> str | None:
        """按 ref 取回图片路径；命中后把使用次数加一。

        返回 None 的三种情况：没有这个 ref、ref 不属于该会话（``scoped`` 且
        ``sid`` 非空时）、记录还在但文件已从磁盘消失（只告警，不删记录）。
        """
        key = str(ref or "").strip()
        if not key:
            return None
        with self._lock:
            db = self._require_db()
            row = db.execute(
                "SELECT ref, path, sid FROM image_refs WHERE ref = ?", (key,)
            ).fetchone()
            if row is None:
                logger.debug("[aiimg] 图片记忆：没有这个引用 ref=%s", key)
                return None
            if scoped and sid and str(row["sid"]) != str(sid):
                logger.debug(
                    "[aiimg] 图片记忆：引用 %s 不属于会话 %s，拒绝解析", key, sid
                )
                return None
            path = str(row["path"])
            if not os.path.exists(path):
                logger.warning(
                    "[aiimg] 图片记忆：引用 %s 指向的文件已失效，跳过 path=%s",
                    key,
                    path,
                )
                return None
            with self._transaction() as writer:
                writer.execute(
                    "UPDATE image_refs SET uses = uses + 1 WHERE ref = ?", (key,)
                )
            logger.debug("[aiimg] 图片记忆：命中 %s → %s", key, path)
            return path

    # -- 查询 ---------------------------------------------------------------

    def latest(
        self, *, sid: str = "", scoped: bool = True, limit: int = 1
    ) -> list[dict]:
        """按登记时间倒序取最近几条（同一时刻以 rowid 兜底），返回整行字段。"""
        cap = _clamp_limit(limit, 1)
        if cap == 0:
            return []
        sql = "SELECT * FROM image_refs"
        clauses: list[str] = []
        params: list[object] = []
        if scoped and sid:
            clauses.append("sid = ?")
            params.append(str(sid))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(cap)
        with self._lock:
            rows = self._require_db().execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def find(
        self,
        *,
        query: str = "",
        kind: str = "",
        sid: str = "",
        scoped: bool = True,
        limit: int = 8,
    ) -> list[dict]:
        """按提示词关键词和类型检索，返回精简字段。

        ``query`` 走 LIKE，其中 ``%``、``_`` 按字面匹配，不做通配符解释，
        避免用户搜 "100%" 时把整库都捞出来。
        """
        cap = _clamp_limit(limit, 8)
        if cap == 0:
            return []
        sql = "SELECT ref, path, kind, prompt, created_at FROM image_refs"
        clauses: list[str] = []
        params: list[object] = []
        text = str(query or "").strip()
        if text:
            clauses.append("prompt LIKE ? ESCAPE '\\'")
            params.append(_like_pattern(text))
        kind_text = str(kind or "").strip()
        if kind_text:
            clauses.append("kind = ?")
            params.append(kind_text)
        if scoped and sid:
            clauses.append("sid = ?")
            params.append(str(sid))
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, rowid DESC LIMIT ?"
        params.append(cap)
        with self._lock:
            rows = self._require_db().execute(sql, params).fetchall()
        return [
            {
                "ref": row["ref"],
                "path": row["path"],
                "kind": row["kind"],
                "prompt": row["prompt"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    # -- 清理 ---------------------------------------------------------------

    def prune(self, *, keep: int = 500) -> int:
        """只保留最新的 ``keep`` 条，返回删掉的条数；没超量就一条不删。"""
        try:
            limit = int(keep)
        except (TypeError, ValueError):
            limit = 500
        limit = max(0, limit)
        removed = 0
        total = 0
        with self._transaction() as db:
            total = int(db.execute("SELECT COUNT(*) FROM image_refs").fetchone()[0])
            excess = total - limit
            if excess <= 0:
                return 0
            cursor = db.execute(
                "DELETE FROM image_refs WHERE ref IN ("
                " SELECT ref FROM image_refs"
                " ORDER BY created_at ASC, rowid ASC LIMIT ?)",
                (excess,),
            )
            removed = max(int(cursor.rowcount or 0), 0)
        logger.info(
            "[aiimg] 图片记忆：超出上限 %d，清理最旧的 %d 条，剩余 %d 条",
            limit,
            removed,
            total - removed,
        )
        return removed
