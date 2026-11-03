"""Reference bridging: provenance checks, de-duplication and byte budgets."""

from __future__ import annotations

from pathlib import Path

import pytest

from core import refbridge


def _photo(path: Path, payload: bytes = b"") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload or (b"\x89PNG\r\n\x1a\n" + path.name.encode("utf-8")))
    return path


def test_classify_names_every_source(tmp_path):
    attachments = [_photo(tmp_path / "msg.png")]
    quoted = [_photo(tmp_path / "quoted.png")]
    gallery = _photo(tmp_path / "gallery" / "g.png")
    upload = _photo(tmp_path / "refs" / "u.png")
    persona = _photo(tmp_path / "persona.png")

    assert (
        refbridge.classify(
            attachments[0], attachments=attachments, gallery_dir=tmp_path / "gallery"
        )
        == refbridge.SOURCE_MESSAGE
    )
    assert (
        refbridge.classify(quoted[0], quoted=quoted, gallery_dir=tmp_path / "gallery")
        == refbridge.SOURCE_QUOTED
    )
    assert (
        refbridge.classify(gallery, gallery_dir=tmp_path / "gallery")
        == refbridge.SOURCE_GALLERY
    )
    assert (
        refbridge.classify(upload, refs_dir=tmp_path / "refs")
        == refbridge.SOURCE_UPLOAD
    )
    assert refbridge.classify(persona, persona=persona) == refbridge.SOURCE_PERSONA


def test_classify_refuses_a_path_from_nowhere(tmp_path):
    stranger = _photo(tmp_path / "elsewhere" / "secret.png")

    assert (
        refbridge.classify(
            stranger,
            attachments=[],
            gallery_dir=tmp_path / "gallery",
            refs_dir=tmp_path / "refs",
        )
        == ""
    )


def test_unknown_source_is_skipped_with_a_note(tmp_path):
    stranger = _photo(tmp_path / "elsewhere" / "secret.png")

    kept, notes = refbridge.plan_references(
        [(stranger, "base")], gallery_dir=tmp_path / "gallery"
    )

    assert kept == []
    assert notes == ["跳过来源不明的图片：secret.png"]


def test_identical_bytes_are_sent_once(tmp_path):
    payload = b"\x89PNG\r\n\x1a\nsame-bytes"
    first = _photo(tmp_path / "refs" / "a.png", payload)
    second = _photo(tmp_path / "refs" / "b.png", payload)

    kept, notes = refbridge.plan_references(
        [(first, "subject"), (second, "subject")], refs_dir=tmp_path / "refs"
    )

    assert [item.path for item in kept] == [first]
    assert notes == ["跳过重复的图片：b.png"]


def test_a_more_specific_role_upgrades_the_duplicate(tmp_path):
    payload = b"\x89PNG\r\n\x1a\nsame-bytes"
    first = _photo(tmp_path / "refs" / "a.png", payload)
    second = _photo(tmp_path / "refs" / "b.png", payload)

    kept, notes = refbridge.plan_references(
        [(first, "subject"), (second, "style")], refs_dir=tmp_path / "refs"
    )

    assert len(kept) == 1
    assert kept[0].role == "style"
    assert notes == ["同一张图出现多次，b.png 的角色改为 style"]


def test_the_edited_picture_keeps_its_role(tmp_path):
    payload = b"\x89PNG\r\n\x1a\nsame-bytes"
    base = _photo(tmp_path / "refs" / "base.png", payload)
    again = _photo(tmp_path / "refs" / "again.png", payload)

    kept, notes = refbridge.plan_references(
        [(base, "base"), (again, "style")], refs_dir=tmp_path / "refs"
    )

    assert kept[0].role == "base"
    assert notes == ["跳过重复的图片：again.png"]


def test_a_single_picture_over_the_limit_is_refused(tmp_path):
    big = tmp_path / "refs" / "big.png"
    big.parent.mkdir(parents=True, exist_ok=True)
    big.write_bytes(b"x" * (2 * 1024 * 1024))

    with pytest.raises(ValueError, match="每张参考图不能超过 1 MB"):
        refbridge.plan_references(
            [(big, "subject")], refs_dir=tmp_path / "refs", max_each_mb=1
        )


def test_the_total_budget_is_refused_before_it_overflows(tmp_path):
    first = tmp_path / "refs" / "a.png"
    second = tmp_path / "refs" / "b.png"
    for index, path in enumerate((first, second)):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Distinct payloads: identical bytes would be de-duplicated first and
        # never reach the total budget.
        path.write_bytes(bytes([index + 1]) * (700 * 1024))

    with pytest.raises(ValueError, match="参考图总量不能超过 1 MB"):
        refbridge.plan_references(
            [(first, "subject"), (second, "subject")],
            refs_dir=tmp_path / "refs",
            max_total_mb=1,
        )


def test_too_many_pictures_is_an_error_not_a_silent_cut(tmp_path):
    pairs = [
        (_photo(tmp_path / "refs" / f"{index}.png"), "subject") for index in range(3)
    ]

    with pytest.raises(ValueError, match="参考图最多 2 张，现在有 3 张"):
        refbridge.plan_references(pairs, refs_dir=tmp_path / "refs", max_images=2)


def test_describe_counts_sources_and_bytes(tmp_path):
    quoted = _photo(tmp_path / "quoted.png")
    upload = _photo(tmp_path / "refs" / "u.png")
    kept, _notes = refbridge.plan_references(
        [(quoted, "base"), (upload, "style")],
        quoted=[quoted],
        refs_dir=tmp_path / "refs",
    )

    text = refbridge.describe(kept)

    assert text.startswith("2 张（引用消息 1 张，参考图库 1 张，共 ")
    assert text.endswith(" KB）")
    assert refbridge.describe([]) == "无"


def test_digest_is_stable_and_content_based(tmp_path):
    payload = b"\x89PNG\r\n\x1a\ncontent"
    first = _photo(tmp_path / "a.png", payload)
    second = _photo(tmp_path / "b.png", payload)

    assert refbridge.digest_of(first) == refbridge.digest_of(second)
    assert len(refbridge.digest_of(first)) == 64
