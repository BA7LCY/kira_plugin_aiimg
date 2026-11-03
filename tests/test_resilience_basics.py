"""Coverage for the resilience features: key pool cooldown, placeholder
detection, output-size warnings and the duplicate-request debouncer."""

import io
import logging
import time

import pytest
from PIL import Image as PILImage

from core.debounce import Debouncer
from core.image_format import (
    image_dimensions,
    is_placeholder_image,
    log_size_mismatch,
)
from core.providers.base import KeyPoolMixin, ProviderSettings


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class _Pool(KeyPoolMixin):
    def __init__(self, keys, cooldown=300):
        self.keys = list(keys)
        self.log = []
        self._init_key_pool(
            ProviderSettings(id="test", key_cooldown_seconds=cooldown)
        )

    def _key_pool(self):
        return self.keys

    def _key_logger(self):
        return _RecordingLogger(self.log)


class _RecordingLogger:
    def __init__(self, sink):
        self.sink = sink

    def warning(self, message, *args, **kwargs):
        self.sink.append(message % args if args else message)

    def info(self, message, *args, **kwargs):
        self.sink.append(message % args if args else message)


def test_pool_rotates_through_every_key():
    pool = _Pool(["a", "b", "c"])

    picks = [pool._pick_key() for _ in range(4)]

    assert picks == ["a", "b", "c", "a"]


def test_rejected_key_is_skipped_while_parked():
    pool = _Pool(["a", "b"])

    pool._handle_key_status("a", 401)

    assert [pool._pick_key() for _ in range(3)] == ["b", "b", "b"]


def test_parked_key_returns_after_the_cooldown():
    pool = _Pool(["a", "b"], cooldown=300)
    pool._handle_key_status("a", 403)
    pool._key_cooldown_until["a"] = time.monotonic() - 1

    assert sorted({pool._pick_key() for _ in range(4)}) == ["a", "b"]


def test_success_status_does_not_park_the_key():
    pool = _Pool(["a", "b"])

    pool._handle_key_status("a", 200)
    pool._handle_key_status("a", 500)

    assert pool._key_cooldown_until == {}


def test_cooldown_zero_disables_parking():
    pool = _Pool(["a", "b"], cooldown=0)

    pool._handle_key_status("a", 401)

    assert pool._key_cooldown_until == {}
    assert [pool._pick_key() for _ in range(2)] == ["a", "b"]


def test_all_keys_parked_still_returns_one():
    pool = _Pool(["a", "b"])
    pool._handle_key_status("a", 401)
    pool._handle_key_status("b", 401)

    assert pool._pick_key() in {"a", "b"}


def test_parking_logs_the_remaining_count():
    pool = _Pool(["a", "b"])

    pool._handle_key_status("a", 429)

    assert pool.log == ["密钥被服务商拒绝，暂时停用 300 秒（剩余可用 2 个）"]


def test_empty_pool_raises():
    pool = _Pool([])

    with pytest.raises(RuntimeError, match="no API key"):
        pool._pick_key()


def test_dimensions_read_from_the_header():
    assert image_dimensions(_png(37, 91)) == (37, 91)
    assert image_dimensions(b"not an image") is None


def test_placeholder_covers_tiny_payloads_and_one_pixel_images():
    assert is_placeholder_image(b"")
    assert is_placeholder_image(b"\x89PNG" + b"0" * 100)
    assert is_placeholder_image(_png(1, 1))
    assert not is_placeholder_image(_png(64, 64))


def test_size_mismatch_is_reported_once(caplog):
    logger = logging.getLogger("test.size_mismatch")

    with caplog.at_level(logging.WARNING, logger="test.size_mismatch"):
        log_size_mismatch(logger, _png(935, 1683), requested_size="854x1842", source="OpenAICompat")
        log_size_mismatch(logger, _png(854, 1842), requested_size="854x1842", source="OpenAICompat")
        log_size_mismatch(logger, _png(64, 64), requested_size="", source="OpenAICompat")

    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "[OpenAICompat] 输出尺寸与请求不一致: requested=854x1842 got=935x1683"
    ]


def test_debouncer_reports_the_first_task_for_a_repeat():
    debounce = Debouncer(3.0)
    key = debounce.make_key("u1", "g1", "generate_image", "一只猫")

    assert debounce.duplicate_of(key) == ""
    debounce.remember(key, "task-1")

    assert debounce.duplicate_of(key) == "task-1"


def test_debouncer_ignores_a_different_prompt():
    debounce = Debouncer(3.0)
    first = debounce.make_key("u1", "g1", "generate_image", "一只猫")
    second = debounce.make_key("u1", "g1", "generate_image", "一只狗")
    debounce.remember(first, "task-1")

    assert debounce.duplicate_of(second) == ""


def test_debouncer_expires_after_the_window():
    debounce = Debouncer(0.05)
    key = debounce.make_key("u1", "一只猫")
    debounce.remember(key, "task-1")
    time.sleep(0.08)

    assert debounce.duplicate_of(key) == ""


def test_debouncer_with_zero_seconds_is_off():
    debounce = Debouncer(0.0)
    key = debounce.make_key("u1", "一只猫")
    debounce.remember(key, "task-1")

    assert debounce.enabled is False
    assert debounce.duplicate_of(key) == ""


def test_debouncer_prunes_old_records():
    debounce = Debouncer(1.0, ttl=1.0)
    for index in range(70):
        debounce.remember(debounce.make_key("u", index), f"task-{index}")
    debounce._records[next(iter(debounce._records))] = (time.monotonic() - 500, "old")

    debounce.remember(debounce.make_key("u", "fresh"), "task-fresh")

    assert len(debounce._records) < 71
