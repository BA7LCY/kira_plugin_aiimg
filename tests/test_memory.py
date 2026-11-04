"""图片记忆库测试：登记去重、ref 解析、会话隔离、LIKE 转义、清理与重开。"""

import logging
import sqlite3

import pytest

from core import memory as memory_module
from core.memory import ImageMemory


def _touch(tmp_path, name="a.png"):
    """造一个真实存在的小文件，resolve 会检查磁盘。"""
    path = tmp_path / name
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    return path


def _memory(tmp_path, name="memory.sqlite3") -> ImageMemory:
    return ImageMemory(tmp_path / name)


# -- ref 形态与唯一性 -------------------------------------------------------


def test_ref_format_and_uniqueness(tmp_path):
    mem = _memory(tmp_path)
    paths = [_touch(tmp_path, f"{index}.png") for index in range(30)]
    refs = mem.remember([str(path) for path in paths], sid="s1", prompt="猫")
    assert len(refs) == 30
    assert len(set(refs)) == 30
    for ref in refs:
        assert memory_module.REF_PATTERN.match(ref), ref
        assert len(ref) == 10
    mem.close()


def test_ref_collision_retries(tmp_path, monkeypatch):
    """随机源撞出重复 ref 时必须换一个，而不是报错。"""
    mem = _memory(tmp_path)
    monkeypatch.setattr(memory_module, "_random_ref", lambda: "img-000001")
    first = mem.remember([str(_touch(tmp_path, "a.png"))])[0]
    candidates = iter(["img-000001", "img-000001", "img-000002"])
    monkeypatch.setattr(memory_module, "_random_ref", lambda: next(candidates))
    second = mem.remember([str(_touch(tmp_path, "b.png"))])[0]
    assert (first, second) == ("img-000001", "img-000002")
    assert len(mem.find(scoped=False, limit=100)) == 2
    mem.close()


def test_ref_exhaustion_rolls_back(tmp_path, monkeypatch):
    """一直撞 ref 时抛错，且整批登记回滚，不留半截数据。"""
    mem = _memory(tmp_path)
    monkeypatch.setattr(memory_module, "_random_ref", lambda: "img-000001")
    mem.remember([str(_touch(tmp_path, "a.png"))])
    with pytest.raises(RuntimeError):
        mem.remember([str(_touch(tmp_path, "b.png"))])
    rows = mem.find(scoped=False, limit=100)
    assert len(rows) == 1
    assert rows[0]["path"].endswith("a.png")
    mem.close()


# -- 同一路径去重 -----------------------------------------------------------


def test_same_path_reuses_ref_and_updates_non_empty_fields(tmp_path):
    mem = _memory(tmp_path)
    path = _touch(tmp_path)
    first = mem.remember(
        [str(path)], sid="s1", kind="image", task_id="t1", prompt="海边日落"
    )
    second = mem.remember([str(path)], sid="s2", task_id="t2")
    assert first == second
    rows = mem.find(scoped=False, limit=10)
    assert len(rows) == 1
    # 只更新传了值的字段：prompt 传空串，旧值保留
    row = mem.latest(sid="s2")[0]
    assert row["task_id"] == "t2"
    assert row["prompt"] == "海边日落"
    assert row["kind"] == "image"
    mem.close()


def test_batch_with_duplicate_paths(tmp_path):
    mem = _memory(tmp_path)
    path = _touch(tmp_path)
    refs = mem.remember([str(path), str(path), str(path)], sid="s1")
    assert len(refs) == 3
    assert len(set(refs)) == 1
    assert len(mem.find(scoped=False, limit=10)) == 1
    mem.close()


# -- resolve ----------------------------------------------------------------


def test_resolve_hit_bumps_uses(tmp_path):
    mem = _memory(tmp_path)
    path = _touch(tmp_path)
    ref = mem.remember([str(path)], sid="s1")[0]
    assert mem.resolve(ref, sid="s1") == str(path)
    assert mem.resolve(ref, sid="s1") == str(path)
    assert mem.latest(sid="s1")[0]["uses"] == 2
    # 不校验会话时同样命中，次数继续累加
    assert mem.resolve(ref, scoped=False) == str(path)
    assert mem.latest(scoped=False)[0]["uses"] == 3
    mem.close()


def test_resolve_missing_ref_returns_none(tmp_path):
    mem = _memory(tmp_path)
    assert mem.resolve("img-000000") is None
    assert mem.resolve("") is None
    mem.close()


def test_resolve_stale_path_warns_and_keeps_row(tmp_path, caplog):
    mem = _memory(tmp_path)
    path = _touch(tmp_path, "gone.png")
    ref = mem.remember([str(path)])[0]
    path.unlink()
    with caplog.at_level(logging.WARNING):
        assert mem.resolve(ref, scoped=False) is None
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "失效" in text and ref in text
    # 只告警不删行：记录仍然在
    rows = mem.latest(scoped=False)
    assert rows[0]["ref"] == ref
    assert rows[0]["uses"] == 0
    mem.close()


def test_use_after_close_raises(tmp_path):
    mem = _memory(tmp_path)
    mem.remember([str(_touch(tmp_path))])
    mem.close()
    with pytest.raises(RuntimeError):
        mem.remember([str(_touch(tmp_path, "b.png"))])
    with pytest.raises(RuntimeError):
        mem.resolve("img-000000")


# -- 会话隔离 ---------------------------------------------------------------


def test_scoped_isolation_between_sessions(tmp_path):
    mem = _memory(tmp_path)
    red = _touch(tmp_path, "red.png")
    blue = _touch(tmp_path, "blue.png")
    ref_red = mem.remember([str(red)], sid="s1", prompt="红裙")[0]
    ref_blue = mem.remember([str(blue)], sid="s2", prompt="红裙")[0]

    assert mem.resolve(ref_red, sid="s1") == str(red)
    assert mem.resolve(ref_red, sid="s2") is None
    assert mem.resolve(ref_red, sid="s2", scoped=False) == str(red)
    # 不带 sid 时不做隔离
    assert mem.resolve(ref_red) == str(red)

    assert [row["ref"] for row in mem.latest(sid="s1")] == [ref_red]
    assert [row["ref"] for row in mem.latest(sid="s2")] == [ref_blue]
    scoped_hits = mem.find(query="红裙", sid="s1", limit=10)
    assert [row["ref"] for row in scoped_hits] == [ref_red]
    assert len(mem.find(query="红裙", scoped=False, limit=10)) == 2
    mem.close()


# -- 检索 -------------------------------------------------------------------


def test_find_escapes_like_wildcards(tmp_path):
    mem = _memory(tmp_path)
    mem.remember([str(_touch(tmp_path, "a.png"))], sid="s1", prompt="100%纯色")
    mem.remember([str(_touch(tmp_path, "b.png"))], sid="s1", prompt="带_下划线")
    mem.remember([str(_touch(tmp_path, "c.png"))], sid="s1", prompt="普通描述")

    assert [row["prompt"] for row in mem.find(query="%", sid="s1")] == ["100%纯色"]
    assert [row["prompt"] for row in mem.find(query="_", sid="s1")] == ["带_下划线"]
    assert [row["prompt"] for row in mem.find(query="%纯", sid="s1")] == ["100%纯色"]
    # 中文关键词照常模糊匹配
    assert [row["prompt"] for row in mem.find(query="纯", sid="s1")] == ["100%纯色"]
    assert len(mem.find(query="", sid="s1", limit=10)) == 3
    mem.close()


def test_find_filters_by_kind_and_returns_five_fields(tmp_path):
    mem = _memory(tmp_path)
    mem.remember([str(_touch(tmp_path, "a.png"))], sid="s1", kind="image", prompt="普通")
    mem.remember([str(_touch(tmp_path, "b.png"))], sid="s1", kind="image", prompt="普通")
    mem.remember([str(_touch(tmp_path, "c.mp4"))], sid="s1", kind="video", prompt="普通视频")

    videos = mem.find(kind="video", sid="s1")
    assert [row["kind"] for row in videos] == ["video"]
    assert len(mem.find(query="普通", sid="s1")) == 3
    assert len(mem.find(query="普通", kind="video", sid="s1")) == 1
    assert set(videos[0]) == {"ref", "path", "kind", "prompt", "created_at"}
    assert mem.find(limit=0) == []
    mem.close()


# -- 顺序 -------------------------------------------------------------------


def test_latest_orders_newest_first(tmp_path):
    mem = _memory(tmp_path)
    refs = []
    for index in range(5):
        path = _touch(tmp_path, f"{index}.png")
        refs.append(mem.remember([str(path)], sid="s1", prompt=f"第{index}张")[0])

    newest_three = mem.latest(sid="s1", limit=3)
    assert [row["ref"] for row in newest_three] == list(reversed(refs))[:3]
    assert mem.latest(sid="s1")[0]["ref"] == refs[-1]
    assert len(mem.latest(sid="s1", limit=100)) == 5
    assert mem.latest(sid="s1", limit=0) == []
    mem.close()


# -- 清理 -------------------------------------------------------------------


def test_prune_keeps_newest_and_logs(tmp_path, caplog):
    mem = _memory(tmp_path)
    refs = []
    for index in range(10):
        path = _touch(tmp_path, f"{index}.png")
        refs.append(mem.remember([str(path)], sid="s1", prompt=f"第{index}张")[0])

    with caplog.at_level(logging.INFO):
        assert mem.prune(keep=4) == 6
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "清理最旧的 6 条" in text

    left = mem.latest(scoped=False, limit=100)
    assert [row["ref"] for row in left] == list(reversed(refs))[:4]
    assert mem.prune(keep=4) == 0
    assert mem.prune(keep=0) == 4
    assert mem.latest(scoped=False) == []
    mem.close()


# -- 重开 -------------------------------------------------------------------


def test_close_and_reopen_keeps_data(tmp_path):
    db_path = tmp_path / "memory.sqlite3"
    mem = ImageMemory(db_path)
    path = _touch(tmp_path)
    ref = mem.remember([str(path)], sid="s1", prompt="留声机")[0]
    assert mem.resolve(ref, sid="s1") == str(path)
    mem.close()
    mem.close()  # 重复关闭不报错

    again = ImageMemory(db_path)
    assert again.find(query="留声机", sid="s1")[0]["ref"] == ref
    assert again.latest(sid="s1")[0]["uses"] == 1
    again.close()


def test_directory_db_path_uses_default_file(tmp_path):
    mem = ImageMemory(tmp_path)
    assert mem.db_path == tmp_path / memory_module.DEFAULT_DB_NAME
    assert mem.db_path.exists()
    mem.close()


def test_schema_matches_spec(tmp_path):
    mem = _memory(tmp_path)
    conn = sqlite3.connect(mem.db_path)
    try:
        columns = {
            row[1]: (row[2], row[3], row[4], row[5])
            for row in conn.execute("PRAGMA table_info('image_refs')")
        }
        assert set(columns) == {
            "ref",
            "path",
            "kind",
            "sid",
            "task_id",
            "prompt",
            "created_at",
            "uses",
        }
        assert columns["ref"][3] == 1  # 主键
        assert columns["path"][1] == 1  # NOT NULL
        assert columns["kind"][2] == "'image'"
        assert columns["uses"][2] == "0"
        assert columns["uses"][1] == 1  # NOT NULL
        indexes = {row[1] for row in conn.execute("PRAGMA index_list('image_refs')")}
        assert "idx_image_refs_path" in indexes
    finally:
        conn.close()
    mem.close()


def test_remember_logs_in_chinese_with_counts(tmp_path, caplog):
    mem = _memory(tmp_path)
    first = _touch(tmp_path, "a.png")
    second = _touch(tmp_path, "b.png")
    with caplog.at_level(logging.INFO):
        mem.remember([str(first), str(second)], sid="s9")
        mem.remember([str(first)], sid="s9")
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "[aiimg]" in text
    assert "图片记忆：登记 2 张，会话 sid=s9（新增 2，复用 0）" in text
    assert "图片记忆：登记 1 张，会话 sid=s9（新增 0，复用 1）" in text
    mem.close()
