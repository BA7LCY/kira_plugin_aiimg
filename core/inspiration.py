"""灵感库：把群友的灵感句子持久化成一个小型 JSON 库。

灵感库是生图插件的轻量素材池：每条记录只有一句灵感、若干标签和来源会话。
之所以不用数据库，是因为条目量小（默认上限 200 条），JSON 文件方便用户直接
备份、查看和手工修改。所有写操作先落到同目录的 .tmp 文件再原子替换，进程
中途退出也不会留下写了一半的 JSON。
"""

from __future__ import annotations

import json
import math
import os
import random as _random
import re
import secrets
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .logbook import get_logger

logger = get_logger("inspiration")

# 条目 ID 固定为 ins- 加 6 位小写十六进制，方便在聊天里简短引用。
_ID_PATTERN = re.compile(r"^ins-[0-9a-f]{6}$")
_DEFAULT_LIMIT = 200
_DEFAULT_PATH = "inspiration.json"


def _now() -> float:
    """返回当前时间戳，统一用 float，便于和 JSON 里的数字互转。"""
    return time.time()


def _clean_tags(tags: Sequence[str] | str | None) -> list[str]:
    """去掉空白标签和重复标签，保留调用方给出的顺序。"""
    if tags is None:
        return []
    if isinstance(tags, str):
        # 调用方偶尔会把单个标签直接传成字符串，按一个标签处理而不是拆字符。
        tags = [tags]
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in tags:
        tag = str(value).strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        cleaned.append(tag)
    return cleaned


def _copy_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """复制一条记录，避免调用方改动返回值时污染库内状态。"""
    return {
        "id": entry["id"],
        "text": entry["text"],
        "tags": list(entry["tags"]),
        "sid": entry["sid"],
        "user_id": entry["user_id"],
        "created_at": entry["created_at"],
        "uses": entry["uses"],
        "promoted": bool(entry.get("promoted")),
    }


def _safe_float(value: Any, fallback: float) -> float:
    """把文件里的数字转成有限 float，坏值回退到默认值。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number) or number < 0:
        return fallback
    return number


def _safe_uses(value: Any) -> int:
    """使用次数必须是非负整数，坏值按 0 处理。"""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, number)


class InspirationStore:
    """灵感条的读写入口，线程安全，每次变更立即落盘。

    文件内容是一个 JSON 数组，元素结构见 ``add`` 的返回值。读取时会尽量
    容忍单条脏数据（跳过并记警告）；只有整个文件无法解析或顶层结构不对时
    才视为损坏，备份成 ``*.broken-<时间戳>`` 后从空库开始。
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH, *, limit: int = _DEFAULT_LIMIT):
        self._path = Path(path)
        self._limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._load()

    # -- 对外属性 ---------------------------------------------------------

    @property
    def path(self) -> Path:
        """灵感库文件路径，排查问题时方便打日志。"""
        return self._path

    @property
    def limit(self) -> int:
        """条目上限。"""
        return self._limit

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- 读取与持久化 -----------------------------------------------------

    def _load(self) -> None:
        """启动时读文件；任何异常都不能让插件起不来。"""
        if not self._path.exists():
            logger.debug("[aiimg] 灵感库：文件 %s 不存在，从空库开始", self._path)
            return
        try:
            raw = self._path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            self._backup_broken(f"文件不是 UTF-8 文本：{exc}")
            return
        except OSError as exc:
            logger.warning(
                "[aiimg] 灵感库：读取 %s 失败（%s），先按空库运行",
                self._path,
                exc,
            )
            return
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._backup_broken(f"JSON 解析失败：{exc}")
            return

        # 兼容将来可能加外壳的写法；当前写出的永远是顶层数组。
        if isinstance(data, dict) and isinstance(data.get("entries"), list):
            data = data["entries"]
        if not isinstance(data, list):
            self._backup_broken("顶层结构不是数组")
            return

        entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        skipped = 0
        for index, item in enumerate(data):
            entry = self._parse_entry(item, index, seen_ids)
            if entry is None:
                skipped += 1
                continue
            entries.append(entry)
            seen_ids.add(entry["id"])
        self._entries = entries
        if skipped:
            logger.warning(
                "[aiimg] 灵感库：载入时跳过 %d 条格式不对的记录（文件 %s）",
                skipped,
                self._path,
            )
        if entries:
            logger.info(
                "[aiimg] 灵感库：从 %s 载入 %d 条（上限 %d）",
                self._path.name,
                len(entries),
                self._limit,
            )
        else:
            logger.debug("[aiimg] 灵感库：%s 内容为空数组，从空库开始", self._path)

    def _parse_entry(
        self, item: Any, index: int, seen_ids: set[str]
    ) -> dict[str, Any] | None:
        """把一条 JSON 记录整理成标准结构；无法修复时返回 None。"""
        if not isinstance(item, dict):
            logger.debug("[aiimg] 灵感库：第 %d 条不是对象，已跳过", index + 1)
            return None
        text = item.get("text")
        text = str(text).strip() if text is not None else ""
        if not text:
            logger.debug("[aiimg] 灵感库：第 %d 条内容为空，已跳过", index + 1)
            return None

        entry_id = item.get("id")
        if not isinstance(entry_id, str) or not _ID_PATTERN.fullmatch(entry_id):
            entry_id = self._new_id(seen_ids)
            logger.warning(
                "[aiimg] 灵感库：第 %d 条 ID 不合法，已改为 %s",
                index + 1,
                entry_id,
            )
        elif entry_id in seen_ids:
            old_id = entry_id
            entry_id = self._new_id(seen_ids)
            logger.warning(
                "[aiimg] 灵感库：第 %d 条 ID %s 重复，已改为 %s",
                index + 1,
                old_id,
                entry_id,
            )

        tags = item.get("tags")
        if not isinstance(tags, list):
            tags = []
        created_at = _safe_float(item.get("created_at"), _now())
        return {
            "id": entry_id,
            "text": text,
            "tags": _clean_tags(tags),
            "sid": str(item.get("sid") or "").strip(),
            "user_id": str(item.get("user_id") or "").strip(),
            "created_at": created_at,
            "uses": _safe_uses(item.get("uses")),
            "promoted": bool(item.get("promoted")),
        }

    def _backup_broken(self, reason: str) -> None:
        """把损坏文件挪到 .broken-<时间戳>，然后按空库继续。"""
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        target = self._path.with_name(f"{self._path.name}.broken-{stamp}")
        suffix = 1
        while target.exists():
            target = self._path.with_name(f"{self._path.name}.broken-{stamp}-{suffix}")
            suffix += 1
        try:
            os.replace(self._path, target)
        except OSError as exc:
            logger.error(
                "[aiimg] 灵感库：文件损坏（%s），备份到 %s 失败（%s），先按空库运行",
                reason,
                target,
                exc,
            )
            return
        logger.warning(
            "[aiimg] 灵感库：文件损坏（%s），已备份为 %s，从空库开始",
            reason,
            target.name,
        )

    def _write(self, entries: list[dict[str, Any]]) -> None:
        """先写同目录 .tmp 再原子替换，失败时保证原文件完好。"""
        payload = [_copy_entry(entry) for entry in entries]
        tmp = self._path.with_name(f"{self._path.name}.tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)
        except OSError as exc:
            # 失败时清掉半成品，原文件此时还没有被 os.replace 碰过。
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.error("[aiimg] 灵感库：写入 %s 失败（%s）", self._path, exc)
            raise

    def _commit(self, entries: list[dict[str, Any]]) -> None:
        """只有磁盘写成功后才替换内存状态，写失败时库内保持原样。"""
        self._write(entries)
        self._entries = entries

    def _new_id(self, used: set[str]) -> str:
        """生成未占用的 ins-xxxxxx。"""
        for _ in range(100):
            candidate = f"ins-{secrets.token_hex(3)}"
            if candidate not in used:
                return candidate
        # 6 位十六进制有 1600 万种组合，正常使用下不可能走到这里。
        raise RuntimeError("灵感库：无法生成唯一的条目 ID")

    # -- 查询过滤 ---------------------------------------------------------

    @staticmethod
    def _visible(entry: dict[str, Any], sid: str, scoped: bool) -> bool:
        """scoped 为真时只让同一会话的条目可见，避免跨群串灵感。"""
        if not scoped:
            return True
        return entry["sid"] == sid

    @staticmethod
    def _matches(entry: dict[str, Any], needle: str) -> bool:
        """query 对正文和标签做不区分大小写的子串匹配。"""
        if not needle:
            return True
        if needle in entry["text"].lower():
            return True
        return any(needle in tag.lower() for tag in entry["tags"])

    # -- 对外接口 ---------------------------------------------------------

    def add(
        self,
        text: str,
        *,
        tags: Sequence[str] | None = None,
        sid: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        """新增一条灵感，返回写入后的记录副本。

        插入前如果已经满库，会先按“使用次数最少、其次最老”的顺序淘汰旧条目，
        给新条目腾位置；新加的这条不会在同一次调用里被淘汰。
        """
        clean_text = str(text if text is not None else "").strip()
        if not clean_text:
            raise ValueError("灵感内容为空")
        clean_tags = _clean_tags(tags)
        clean_sid = str(sid or "").strip()
        clean_user = str(user_id or "").strip()

        with self._lock:
            entries = [_copy_entry(entry) for entry in self._entries]
            evicted: list[dict[str, Any]] = []
            while len(entries) + 1 > self._limit:
                victim = min(
                    range(len(entries)),
                    key=lambda i: (
                        entries[i]["uses"],
                        entries[i]["created_at"],
                        i,
                    ),
                )
                evicted.append(entries.pop(victim))
            used = {entry["id"] for entry in entries}
            entry = {
                "id": self._new_id(used),
                "text": clean_text,
                "tags": clean_tags,
                "sid": clean_sid,
                "user_id": clean_user,
                "created_at": _now(),
                "uses": 0,
                "promoted": False,
            }
            entries.append(entry)
            self._commit(entries)
            total = len(entries)

        # 只有落盘成功后才记录淘汰结果，避免写失败时日志说删了其实没删。
        if evicted:
            logger.info(
                "[aiimg] 灵感库：超过 %d 条上限，删掉最久没用的 %d 条（%s）",
                self._limit,
                len(evicted),
                "、".join(entry["id"] for entry in evicted),
            )
        logger.info(
            "[aiimg] 灵感库：新增第 %d 条，sid=%s，id=%s",
            total,
            clean_sid or "-",
            entry["id"],
        )
        return _copy_entry(entry)

    def list(
        self,
        *,
        query: str = "",
        sid: str = "",
        scoped: bool = True,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        """按关键字和会话范围查询，最新的排在前面。"""
        clean_query = str(query or "").strip().lower()
        clean_sid = str(sid or "").strip()
        max_items = max(0, int(limit))
        with self._lock:
            rows = [
                (position, entry)
                for position, entry in enumerate(self._entries)
                if self._visible(entry, clean_sid, scoped)
                and self._matches(entry, clean_query)
            ]
            # 时间戳相同时按下标兜底，保证后插入的始终排在前面。
            rows.sort(
                key=lambda item: (item[1]["created_at"], item[0]),
                reverse=True,
            )
            if max_items:
                rows = rows[:max_items]
            else:
                rows = []
            result = [_copy_entry(entry) for _, entry in rows]
        logger.debug(
            "[aiimg] 灵感库：查询 %r，sid=%s，scoped=%s，返回 %d 条",
            clean_query,
            clean_sid or "-",
            scoped,
            len(result),
        )
        return result

    def random(
        self,
        *,
        count: int = 1,
        sid: str = "",
        scoped: bool = True,
    ) -> list[dict[str, Any]]:
        """随机抽取若干条并把 uses 加一，抽中结果立即落盘。"""
        wanted = max(0, int(count))
        if wanted == 0:
            return []
        clean_sid = str(sid or "").strip()
        with self._lock:
            pool = [
                entry
                for entry in self._entries
                if self._visible(entry, clean_sid, scoped)
            ]
            if not pool:
                logger.debug(
                    "[aiimg] 灵感库：随机抽取时没有可用条目（sid=%s）",
                    clean_sid or "-",
                )
                return []
            picked = _random.sample(pool, min(wanted, len(pool)))
            picked_ids = {entry["id"] for entry in picked}

            # 先在副本上累加 uses，写盘成功后才替换内存，失败不产生副作用。
            entries = []
            updated: dict[str, dict[str, Any]] = {}
            for entry in self._entries:
                copy = _copy_entry(entry)
                if copy["id"] in picked_ids:
                    copy["uses"] += 1
                    updated[copy["id"]] = copy
                entries.append(copy)
            self._commit(entries)
            result = [updated[entry["id"]] for entry in picked]

        logger.info(
            "[aiimg] 灵感库：随机抽取 %d 条（sid=%s，候选 %d 条）",
            len(result),
            clean_sid or "-",
            len(pool),
        )
        return [_copy_entry(entry) for entry in result]

    def mark_promoted(self, entry_id: str) -> bool:
        """把一条标成「已收进预设」；原件保留，只改标记。

        找不到时返回 False。重复标记不算错，直接返回 True。
        """
        target = str(entry_id or "")
        with self._lock:
            entries = [_copy_entry(entry) for entry in self._entries]
            for entry in entries:
                if entry["id"] != target:
                    continue
                if entry.get("promoted"):
                    return True
                entry["promoted"] = True
                self._commit(entries)
                break
            else:
                logger.debug("[aiimg] 灵感库：标记收录时没找到 %s", target or "-")
                return False
        logger.info("[aiimg] 灵感库：%s 已收进预设", target)
        return True

    def remove(self, entry_id: str) -> bool:
        """按 ID 删除一条；不存在时返回 False。"""
        target = str(entry_id or "")
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry["id"] != target:
                    continue
                entries = [
                    _copy_entry(row)
                    for position, row in enumerate(self._entries)
                    if position != index
                ]
                self._commit(entries)
                total = len(entries)
                break
            else:
                logger.debug("[aiimg] 灵感库：删除时没找到 %s", target or "-")
                return False

        logger.info(
            "[aiimg] 灵感库：删除 %s，剩余 %d 条",
            target,
            total,
        )
        return True
