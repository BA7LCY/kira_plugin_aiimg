"""视频发送方式的纯规划测试：三档模式、回退顺序、空值兜底。"""

import logging
from pathlib import Path

import pytest

from core.delivery import (
    VIDEO_SEND_MODES,
    log_video_attempts,
    normalize_send_mode,
    plan_video_attempts,
)

VIDEO = "https://cdn.example.test/v/out.mp4"
LOCAL = "/data/videos/out.mp4"


def test_modes_tuple_is_the_contract():
    assert VIDEO_SEND_MODES == ("auto", "url", "file")


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("auto", "auto"),
        ("url", "url"),
        ("file", "file"),
        (" AUTO ", "auto"),
        ("Url", "url"),
        ("FILE", "file"),
    ],
)
def test_normalize_keeps_known_modes(raw, expected):
    assert normalize_send_mode(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", None, "base64", "stream", "字面量", 0])
def test_normalize_falls_back_to_auto(raw):
    assert normalize_send_mode(raw) == "auto"


def test_file_mode_uses_only_the_local_copy():
    assert plan_video_attempts("file", LOCAL, VIDEO) == [("file", LOCAL)]


def test_file_mode_without_a_path_plans_nothing():
    # 明确要发文件时不该偷偷退回链接，否则「只要文件」的语义就没了。
    assert plan_video_attempts("file", "", VIDEO) == []
    assert plan_video_attempts("file", "   ", VIDEO) == []
    assert plan_video_attempts("file", None, VIDEO) == []


def test_url_mode_uses_only_the_link():
    assert plan_video_attempts("url", LOCAL, VIDEO) == [("url", VIDEO)]


def test_url_mode_without_a_link_plans_nothing():
    assert plan_video_attempts("url", LOCAL, "") == []
    assert plan_video_attempts("url", LOCAL, "  ") == []
    assert plan_video_attempts("url", LOCAL, None) == []


def test_auto_prefers_the_local_file():
    assert plan_video_attempts("auto", LOCAL, VIDEO) == [
        ("file", LOCAL),
        ("url", VIDEO),
    ]


def test_auto_falls_back_to_whichever_exists():
    assert plan_video_attempts("auto", LOCAL, "") == [("file", LOCAL)]
    assert plan_video_attempts("auto", "", VIDEO) == [("url", VIDEO)]


def test_auto_with_nothing_plans_nothing():
    assert plan_video_attempts("auto", "", "") == []
    assert plan_video_attempts("auto", None, None) == []
    assert plan_video_attempts("auto", "  ", "  ") == []


def test_auto_never_sends_the_same_value_twice():
    # 旧任务只有链接时，本地路径和链接是同一个值，只该发一次。
    assert plan_video_attempts("auto", VIDEO, VIDEO) == [("file", VIDEO)]


def test_unknown_mode_behaves_like_auto():
    assert plan_video_attempts("nonsense", LOCAL, VIDEO) == [
        ("file", LOCAL),
        ("url", VIDEO),
    ]
    assert plan_video_attempts("", LOCAL, VIDEO) == [
        ("file", LOCAL),
        ("url", VIDEO),
    ]


def test_attempt_payload_is_the_value_itself():
    attempts = plan_video_attempts("auto", LOCAL, VIDEO)
    assert [kind for kind, _value in attempts] == ["file", "url"]
    assert attempts[0][1] == LOCAL
    assert attempts[1][1] == VIDEO


def test_path_objects_are_accepted():
    # pathlib 会按平台改写分隔符，这里跟着它走，别把平台差异当成逻辑差异。
    assert plan_video_attempts("file", Path(LOCAL), "") == [("file", str(Path(LOCAL)))]


def test_log_reports_count_and_kinds_without_leaking_the_link(caplog):
    plan = plan_video_attempts("auto", LOCAL, VIDEO)
    with caplog.at_level(logging.INFO, logger="kira_plugin_aiimg.delivery"):
        log_video_attempts("task-42", "auto", plan)
    assert "task-42" in caplog.text
    assert "方式=auto" in caplog.text
    assert "计划 2 次" in caplog.text
    assert "file、url" in caplog.text
    # 链接和本地路径都不该出现在日志里。
    assert VIDEO not in caplog.text
    assert LOCAL not in caplog.text


def test_log_of_an_empty_plan_is_still_one_line(caplog):
    with caplog.at_level(logging.INFO, logger="kira_plugin_aiimg.delivery"):
        log_video_attempts("task-43", "file", plan_video_attempts("file", "", VIDEO))
    assert "计划 0 次" in caplog.text
    assert "无" in caplog.text
    assert VIDEO not in caplog.text
