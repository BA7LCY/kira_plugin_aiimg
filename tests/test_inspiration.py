"""灵感库 InspirationStore 的持久化、查询、淘汰和并发测试。"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core import inspiration as inspiration_module
from core.inspiration import InspirationStore

_ID_RE = re.compile(r"^ins-[0-9a-f]{6}$")


def _texts(rows):
    return {row["text"] for row in rows}


def _read_file(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite(path: Path, mutate) -> None:
    """直接改盘上的 JSON，用来构造指定 uses / created_at 的场景。"""
    data = _read_file(path)
    mutate(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_add_and_list_basic(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)

    entry = store.add(
        "  夕阳下的海边  ",
        tags=["风景", "治愈", "风景"],
        sid="s1",
        user_id="u1",
    )
    assert _ID_RE.fullmatch(entry["id"])
    assert entry["text"] == "夕阳下的海边"
    assert entry["tags"] == ["风景", "治愈"]
    assert entry["sid"] == "s1"
    assert entry["user_id"] == "u1"
    assert entry["uses"] == 0
    assert isinstance(entry["created_at"], float)

    store.add("雨夜便利店", tags=["城市"])
    rows = store.list(scoped=False, limit=10)
    assert len(rows) == 2
    # 默认按时间倒序，最近添加的排在前面。
    assert rows[0]["text"] == "雨夜便利店"
    assert rows[1]["text"] == "夕阳下的海边"

    # 返回值是副本，改它不能影响库内数据。
    entry["text"] = "被篡改"
    assert "夕阳下的海边" in _texts(store.list(scoped=False, limit=10))

    # 文件是顶层数组，字段和各条记录一致。
    data = _read_file(path)
    assert isinstance(data, list)
    assert len(data) == 2
    assert set(data[0]) == {
        "id", "text", "tags", "sid", "user_id", "created_at", "uses", "promoted",
    }

    reloaded = InspirationStore(path)
    assert len(reloaded) == 2
    assert _texts(reloaded.list(scoped=False, limit=10)) == {
        "夕阳下的海边", "雨夜便利店",
    }


def test_mark_promoted_keeps_the_entry_and_survives_reload(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)
    entry = store.add("黄昏的海边，长焦压缩感")

    assert entry["promoted"] is False
    assert store.mark_promoted(entry["id"]) is True
    assert store.mark_promoted(entry["id"]) is True

    rows = store.list(scoped=False, limit=5)
    assert len(rows) == 1
    assert rows[0]["promoted"] is True
    assert rows[0]["text"] == entry["text"]

    reloaded = InspirationStore(path)
    assert reloaded.list(scoped=False, limit=5)[0]["promoted"] is True
    assert store.mark_promoted("ins-000000") is False


def test_empty_text_rejected_and_single_tag_string(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)

    with pytest.raises(ValueError, match="灵感内容为空"):
        store.add("")
    with pytest.raises(ValueError, match="灵感内容为空"):
        store.add("   ")
    assert len(store) == 0
    assert not path.exists()

    entry = store.add("一句灵感", tags="  单独标签  ")
    assert entry["tags"] == ["单独标签"]


def test_query_matches_text_and_tags_case_insensitively(tmp_path):
    store = InspirationStore(tmp_path / "inspiration.json")
    store.add("Sunset Beach", tags=["Travel", "photo"])
    store.add("夜空星星", tags=["星空"])
    store.add("海边日落", tags=["旅行"])

    assert [row["text"] for row in store.list(query="sun", scoped=False)] == [
        "Sunset Beach"
    ]
    assert [row["text"] for row in store.list(query="TRAVEL", scoped=False)] == [
        "Sunset Beach"
    ]
    assert [row["text"] for row in store.list(query="photo", scoped=False)] == [
        "Sunset Beach"
    ]
    # “星”同时命中正文“星星”和标签“星空”，但只应返回同一条。
    assert len(store.list(query="星", scoped=False)) == 1
    assert len(store.list(query="旅", scoped=False)) == 1
    assert store.list(query="不存在的关键字", scoped=False) == []
    assert len(store.list(limit=2, scoped=False)) == 2


def test_scoped_isolates_sessions(tmp_path):
    store = InspirationStore(tmp_path / "inspiration.json")
    store.add("公共灵感", sid="")
    store.add("一号的灵感", sid="s1")
    store.add("二号的灵感", sid="s2")

    assert _texts(store.list(sid="s1", scoped=True)) == {"一号的灵感"}
    assert _texts(store.list(sid="s2", scoped=True)) == {"二号的灵感"}
    assert _texts(store.list(sid="", scoped=True)) == {"公共灵感"}
    assert _texts(store.list(sid="s1", scoped=False)) == {
        "公共灵感", "一号的灵感", "二号的灵感",
    }
    # 随机抽取也遵守同样的会话隔断。
    assert [row["text"] for row in store.random(count=5, sid="s2")] == ["二号的灵感"]


def test_eviction_prefers_least_used_then_oldest(tmp_path, caplog):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path, limit=3)
    store.add("A")
    store.add("B")
    store.add("C")

    # B 使用最少（0 次），虽然它不是最老的，也应先被淘汰。
    def mark_uses(rows):
        table = {"A": (5, 100.0), "B": (0, 200.0), "C": (2, 300.0)}
        for row in rows:
            row["uses"], row["created_at"] = table[row["text"]]

    _rewrite(path, mark_uses)
    store = InspirationStore(path, limit=3)
    with caplog.at_level(logging.INFO, logger="kira_plugin_aiimg.inspiration"):
        store.add("D")

    assert _texts(store.list(scoped=False, limit=10)) == {"A", "C", "D"}
    assert len(_read_file(path)) == 3
    assert any(
        "超过 3 条上限" in record.message and "删掉最久没用的 1 条" in record.message
        for record in caplog.records
    )

    # uses 全相同后，只能比谁更老：A 的 created_at 最小，被淘汰。
    def mark_all_used(rows):
        for row in rows:
            row["uses"] = 4

    _rewrite(path, mark_all_used)
    store = InspirationStore(path, limit=3)
    store.add("E")
    assert _texts(store.list(scoped=False, limit=10)) == {"C", "D", "E"}


def test_default_limit_is_200_and_evicts_on_overflow(tmp_path):
    path = tmp_path / "inspiration.json"
    seeded = []
    for index in range(200):
        seeded.append({
            "id": f"ins-{index:06x}",
            "text": f"旧灵感-{index}",
            "tags": [],
            "sid": "",
            "user_id": "",
            "created_at": float(index),
            # 第 0 条用得最少，应成为唯一的淘汰对象。
            "uses": 0 if index == 0 else 1,
        })
    path.write_text(json.dumps(seeded, ensure_ascii=False), encoding="utf-8")

    store = InspirationStore(path)
    assert len(store) == 200
    new_entry = store.add("新的灵感")

    rows = store.list(scoped=False, limit=500)
    assert len(rows) == 200
    assert "旧灵感-0" not in _texts(rows)
    assert "新的灵感" in _texts(rows)
    assert new_entry["id"] in {row["id"] for row in rows}
    assert len(_read_file(path)) == 200


def test_corrupt_json_is_backed_up_then_starts_empty(tmp_path, caplog):
    path = tmp_path / "inspiration.json"
    broken = "{ 这不是合法的 JSON"
    path.write_text(broken, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="kira_plugin_aiimg.inspiration"):
        store = InspirationStore(path)

    assert len(store) == 0
    backups = list(tmp_path.glob("inspiration.json.broken-*"))
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == broken
    assert not path.exists()
    assert any(
        "文件损坏" in record.message and "已备份" in record.message
        for record in caplog.records
    )

    # 备份之后仍能正常写入，插件不会因为坏文件起不来。
    store.add("损坏之后的新灵感")
    assert path.exists()
    assert _read_file(path)[0]["text"] == "损坏之后的新灵感"


def test_non_utf8_file_is_backed_up(tmp_path):
    path = tmp_path / "inspiration.json"
    path.write_bytes(b"\xff\xfe\x00broken")

    store = InspirationStore(path)

    assert len(store) == 0
    assert list(tmp_path.glob("inspiration.json.broken-*"))
    assert not path.exists()


def test_atomic_write_failure_keeps_original_file_and_memory(tmp_path, monkeypatch):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)
    first = store.add("第一条")
    original = path.read_bytes()

    def fail_replace(src, dst):
        raise OSError(28, "磁盘空间不足")

    monkeypatch.setattr(inspiration_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        store.add("第二条")

    # 原文件逐字节未变，内存里也没有偷偷多出一条。
    assert path.read_bytes() == original
    rows = store.list(scoped=False, limit=10)
    assert [row["text"] for row in rows] == ["第一条"]
    assert rows[0]["id"] == first["id"]
    assert not path.with_name("inspiration.json.tmp").exists()
    assert [row["text"] for row in _read_file(path)] == ["第一条"]


def test_random_increments_uses_and_persists(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)
    store.add("灵光一", sid="s1")
    store.add("灵光二", sid="s1")
    store.add("灵光三", sid="s2")

    picked = store.random(count=2, sid="s1")
    assert len(picked) == 2
    assert {row["sid"] for row in picked} == {"s1"}
    assert len({row["id"] for row in picked}) == 2
    assert {row["uses"] for row in picked} == {1}

    reloaded = InspirationStore(path)
    s1_rows = reloaded.list(sid="s1", scoped=True)
    assert sorted(row["uses"] for row in s1_rows) == [1, 1]
    # 没被抽中的会话不受影响。
    assert reloaded.list(sid="s2", scoped=True)[0]["uses"] == 0

    # count 超过库存时返回全部且不重复，uses 继续累加并落盘。
    every = reloaded.random(count=99, scoped=False)
    assert len(every) == 3
    assert len({row["id"] for row in every}) == 3
    assert _texts(every) == {"灵光一", "灵光二", "灵光三"}
    persisted = _read_file(path)
    assert sorted(row["uses"] for row in persisted) == [1, 2, 2]

    before = path.read_bytes()
    assert reloaded.random(count=0) == []
    assert path.read_bytes() == before


def test_random_on_empty_library_returns_empty(tmp_path):
    store = InspirationStore(tmp_path / "inspiration.json")
    assert store.random() == []
    assert store.list() == []
    assert not (tmp_path / "inspiration.json").exists()


def test_remove(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path)
    keep = store.add("留下")
    drop = store.add("删掉")

    assert store.remove(drop["id"]) is True
    assert store.remove(drop["id"]) is False
    assert store.remove("ins-000000") is False
    assert [row["id"] for row in store.list(scoped=False, limit=10)] == [keep["id"]]

    reloaded = InspirationStore(path)
    assert [row["text"] for row in reloaded.list(scoped=False, limit=10)] == ["留下"]


def test_concurrent_add_does_not_lose_entries(tmp_path):
    path = tmp_path / "inspiration.json"
    store = InspirationStore(path, limit=200)

    def work(index: int) -> str:
        return store.add(f"并发灵感-{index}", sid="s1")["id"]

    with ThreadPoolExecutor(max_workers=20) as executor:
        ids = list(executor.map(work, range(20)))

    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert len(store) == 20

    data = _read_file(path)
    assert len(data) == 20
    assert len({row["id"] for row in data}) == 20
    assert {row["text"] for row in data} == {f"并发灵感-{i}" for i in range(20)}
    assert _texts(store.list(sid="s1", scoped=True, limit=100)) == {
        f"并发灵感-{i}" for i in range(20)
    }
