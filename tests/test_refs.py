"""Ordered, role-tagged index for uploaded selfie reference photos."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core import refs


def _photo(data_dir: Path, name: str, mtime: float | None = None) -> Path:
    path = data_dir / "refs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_appended_uploads_keep_their_order_and_default_role(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "a.png")
    _photo(data_dir, "b.png")

    refs.append_entry(data_dir, "refs/a.png")
    refs.append_entry(data_dir, "refs/b.png")

    entries = refs.list_entries(data_dir)
    assert [entry["file"] for entry in entries] == ["refs/a.png", "refs/b.png"]
    assert [entry["role"] for entry in entries] == ["subject", "subject"]

    saved = json.loads((data_dir / "refs" / "index.json").read_text(encoding="utf-8"))
    assert isinstance(saved, list)
    assert isinstance(saved[0]["added_at"], int)
    assert not (data_dir / "refs" / "index.json.tmp").exists()


def test_unindexed_files_append_after_indexed_entries_by_mtime(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "indexed.png")
    refs.save_entries(
        data_dir,
        [{"file": "refs/indexed.png", "role": "clothing", "added_at": 1}],
    )
    _photo(data_dir, "later.png", mtime=300)
    _photo(data_dir, "earlier.png", mtime=100)

    entries = refs.list_entries(data_dir)

    assert [entry["file"] for entry in entries] == [
        "refs/indexed.png",
        "refs/earlier.png",
        "refs/later.png",
    ]
    assert [entry["role"] for entry in entries] == ["clothing", "subject", "subject"]


def test_a_missing_indexed_file_is_skipped(tmp_path):
    data_dir = tmp_path / "data"
    refs.save_entries(
        data_dir,
        [{"file": "refs/gone.png", "role": "subject", "added_at": 1}],
    )

    assert refs.list_entries(data_dir) == []


def test_set_role_persists_and_validates(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "a.png")
    refs.append_entry(data_dir, "refs/a.png")

    refs.set_role(data_dir, "refs/a.png", "style")

    assert refs.read_index(data_dir)[0]["role"] == "style"
    with pytest.raises(ValueError):
        refs.set_role(data_dir, "refs/a.png", "face")
    with pytest.raises(KeyError):
        refs.set_role(data_dir, "refs/missing.png", "pose")


def test_reorder_requires_the_full_list(tmp_path):
    data_dir = tmp_path / "data"
    for name in ("a.png", "b.png", "c.png"):
        _photo(data_dir, name)
        refs.append_entry(data_dir, f"refs/{name}")

    refs.reorder(data_dir, ["refs/c.png", "refs/a.png", "refs/b.png"])
    assert [entry["file"] for entry in refs.list_entries(data_dir)] == [
        "refs/c.png",
        "refs/a.png",
        "refs/b.png",
    ]

    with pytest.raises(ValueError):
        refs.reorder(data_dir, ["refs/a.png"])
    with pytest.raises(ValueError):
        refs.reorder(data_dir, ["refs/a.png", "refs/a.png", "refs/b.png"])
    with pytest.raises(ValueError):
        refs.reorder(data_dir, ["refs/a.png", "refs/b.png", "refs/nope.png"])
    with pytest.raises(ValueError):
        refs.reorder(data_dir, [])


def test_remove_entry_only_drops_the_deleted_photo(tmp_path):
    data_dir = tmp_path / "data"
    first = _photo(data_dir, "a.png")
    _photo(data_dir, "b.png")
    refs.append_entry(data_dir, "refs/a.png")
    refs.append_entry(data_dir, "refs/b.png")

    first.unlink()
    refs.remove_entry(data_dir, "refs/a.png")

    assert [entry["file"] for entry in refs.list_entries(data_dir)] == ["refs/b.png"]


def test_index_file_is_never_treated_as_a_photo(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "a.png")
    refs.append_entry(data_dir, "refs/a.png")

    assert [entry["file"] for entry in refs.list_entries(data_dir)] == ["refs/a.png"]


def test_corrupt_index_falls_back_to_scan(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "a.png")
    (data_dir / "refs" / "index.json").write_text("not json", encoding="utf-8")

    entries = refs.list_entries(data_dir)

    assert [(entry["file"], entry["role"]) for entry in entries] == [
        ("refs/a.png", "subject")
    ]


def test_normalize_file_strips_the_prefix_and_blocks_traversal(tmp_path):
    assert refs.normalize_file("a.png") == "refs/a.png"
    assert refs.normalize_file("refs/a.png") == "refs/a.png"
    assert refs.normalize_file("refs\\a.png") == "refs/a.png"
    for bad in ("", "refs/", "../evil.png", "sub/a.png", ".hidden"):
        assert refs.normalize_file(bad) == ""
    with pytest.raises(ValueError):
        refs.append_entry(tmp_path, "../evil.png")
    with pytest.raises(ValueError):
        refs.append_entry(tmp_path, "refs/a.png", role="face")


def test_roles_are_validated_on_read(tmp_path):
    data_dir = tmp_path / "data"
    _photo(data_dir, "a.png")
    (data_dir / "refs" / "index.json").write_text(
        json.dumps([{"file": "refs/a.png", "role": "unknown", "added_at": "x"}]),
        encoding="utf-8",
    )

    entry = refs.list_entries(data_dir)[0]

    assert entry["role"] == refs.DEFAULT_ROLE
    assert entry["added_at"] == 0
