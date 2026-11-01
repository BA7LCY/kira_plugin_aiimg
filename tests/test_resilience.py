"""Provider resilience: backoff, circuit breaker, per-provider concurrency."""

import asyncio
import io
import json
from types import SimpleNamespace

from PIL import Image


def _png():
    output = io.BytesIO()
    Image.new("RGB", (2, 2), "red").save(output, "PNG")
    return output.getvalue()

import pytest

from core.http import backoff_delay, retry_after_seconds
from core.provider_stats import ProviderStats
from core.providers.base import ImageResult
from core.task_store import TaskRecord
from test_plugin_lifecycle import _Ctx, _config, plugin_module  # noqa: F401


def test_backoff_delay_grows_then_caps():
    low = [backoff_delay(0, base=0.5) for _ in range(20)]
    high = [backoff_delay(5, base=0.5, cap=8.0) for _ in range(20)]

    assert all(0.4 <= value <= 0.6 for value in low)
    assert all(6.8 <= value <= 8.0 for value in high)
    assert len(set(round(value, 3) for value in low)) > 1


def test_retry_after_seconds_only_accepts_numbers():
    assert retry_after_seconds("7") == 7.0
    assert retry_after_seconds(" 0.5 ") == 0.5
    assert retry_after_seconds("999") == 300.0
    assert retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT") is None
    assert retry_after_seconds(None) is None


def test_stats_snapshot_reports_latency_and_success_rate():
    stats = ProviderStats(None, threshold=2, cooldown=60)

    asyncio.run(stats.record_success("p1", 100))
    asyncio.run(stats.record_success("p1", 300))
    asyncio.run(stats.record_failure("p1", 500, "HTTP 500"))

    row = stats.snapshot()[0]

    assert row["attempts"] == 3
    assert row["success_rate"] == pytest.approx(0.667, abs=0.001)
    assert row["p50_ms"] == 300.0
    assert row["p95_ms"] == 500.0
    assert row["last_error"] == "HTTP 500"


def test_stats_cool_down_then_recover(tmp_path):
    stats = ProviderStats(tmp_path, threshold=2, cooldown=60)

    asyncio.run(stats.record_failure("p1", 10, "boom"))
    assert stats.cooling_seconds("p1") == 0

    asyncio.run(stats.record_failure("p1", 10, "boom"))
    assert stats.cooling_seconds("p1") > 0

    reloaded = ProviderStats(tmp_path, threshold=2, cooldown=60)
    assert reloaded.cooling_seconds("p1") > 0

    asyncio.run(reloaded.record_success("p1", 10))
    assert reloaded.cooling_seconds("p1") == 0


class _Gallery:
    async def register_video(self, path, **kwargs):
        return path

    async def save_image(self, data, **kwargs):
        return "media/fake.png"


class _Store:
    def set_audit(self, *args, **kwargs):
        return None

    def upsert(self, record):
        return None


class _ProviderSettings:
    def __init__(self, max_concurrent=0):
        self.max_concurrent = max_concurrent
        self.default_size = "1024x1024"


class _Backend:
    supports_edit = True

    def __init__(self, *, fail=False, delay=0.0, ok=True):
        self.fail = fail
        self.delay = delay
        self.ok = ok
        self.calls = 0
        self.inflight = 0
        self.max_inflight = 0

    async def generate(self, prompt, *, intent=None, size_hint=""):
        self.calls += 1
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail:
                raise RuntimeError("HTTP 500: boom")
            return [
                ImageResult(data=_png(), mime="image/png", ext="png", size="1024x1024", model="fake")
            ]
        finally:
            self.inflight -= 1

    async def edit(self, prompt, images, *, intent=None, size_hint=""):
        return await self.generate(prompt)


class _Registry:
    def __init__(self, backends, settings=None):
        self._backends = backends
        self._settings = settings or {
            provider_id: _ProviderSettings() for provider_id in backends
        }

    def get_backend(self, provider_id):
        if provider_id not in self._backends:
            raise KeyError(provider_id)
        return self._backends[provider_id]

    def get_settings(self, provider_id):
        if provider_id not in self._settings:
            raise KeyError(provider_id)
        return self._settings[provider_id]


def _prepare(plugin_module, tmp_path, backends, *, threshold=3, settings=None, config=None):
    cfg = config or _config()
    cfg.setdefault("section_queue", {})
    cfg["section_queue"]["provider_failure_threshold"] = threshold
    cfg["section_queue"]["provider_cooldown_seconds"] = 60
    plugin = plugin_module.AIImagePlugin(_Ctx(tmp_path), cfg)
    plugin._data_dir = tmp_path
    plugin._registry = _Registry(backends, settings)
    plugin._stats = ProviderStats(tmp_path / "stats", threshold=threshold, cooldown=60)
    plugin._gallery = _Gallery()
    plugin._store = _Store()
    return plugin


def _record(chain):
    return TaskRecord(
        task_id="t1",
        kind="generate_image",
        prompt="a cat",
        chain=list(chain),
        extra={"chain_items": [{"provider": p, "output": ""} for p in chain]},
    )


def test_chain_skips_provider_in_cooldown(plugin_module, tmp_path):
    bad = _Backend(fail=True)
    good = _Backend()
    plugin = _prepare(plugin_module, tmp_path, {"bad": bad, "good": good}, threshold=2)

    for _ in range(2):
        result = asyncio.run(plugin._run_image_chain(_record(["bad", "good"]), kind="generate"))
        assert result["provider"] == "good"

    assert bad.calls == 2
    assert plugin._stats.cooling_seconds("bad") > 0

    result = asyncio.run(plugin._run_image_chain(_record(["bad", "good"]), kind="generate"))

    assert result["provider"] == "good"
    assert bad.calls == 2


def test_chain_still_uses_cooling_provider_when_it_is_the_only_option(plugin_module, tmp_path):
    bad = _Backend(fail=True)
    plugin = _prepare(plugin_module, tmp_path, {"bad": bad}, threshold=1)

    with pytest.raises(RuntimeError):
        asyncio.run(plugin._run_image_chain(_record(["bad"]), kind="generate"))
    assert plugin._stats.cooling_seconds("bad") > 0

    with pytest.raises(RuntimeError):
        asyncio.run(plugin._run_image_chain(_record(["bad"]), kind="generate"))

    assert bad.calls == 2


def test_per_provider_concurrency_limits_inflight_calls(plugin_module, tmp_path):
    backend = _Backend(delay=0.05)
    settings = {"p1": _ProviderSettings(max_concurrent=1)}
    plugin = _prepare(plugin_module, tmp_path, {"p1": backend}, settings=settings)

    async def run_twice():
        await asyncio.gather(
            plugin._run_image_chain(_record(["p1"]), kind="generate"),
            plugin._run_image_chain(_record(["p1"]), kind="generate"),
        )

    asyncio.run(run_twice())

    assert backend.calls == 2
    assert backend.max_inflight == 1


def test_unlimited_provider_runs_in_parallel(plugin_module, tmp_path):
    backend = _Backend(delay=0.05)
    plugin = _prepare(plugin_module, tmp_path, {"p1": backend})

    async def run_twice():
        await asyncio.gather(
            plugin._run_image_chain(_record(["p1"]), kind="generate"),
            plugin._run_image_chain(_record(["p1"]), kind="generate"),
        )

    asyncio.run(run_twice())

    assert backend.max_inflight == 2


def test_metrics_endpoint_lists_provider_health(plugin_module, tmp_path):
    plugin = _prepare(plugin_module, tmp_path, {"p1": _Backend()})
    asyncio.run(plugin._stats.record_success("p1", 42))

    payload = asyncio.run(plugin.api_metrics())

    assert payload["providers"][0]["id"] == "p1"
    assert payload["providers"][0]["success_rate"] == 1.0
    assert payload["generated_at"] > 0


class _EmptyBackend(_Backend):
    async def generate(self, prompt, *, intent=None, size_hint=""):
        self.calls += 1
        return []


class _GarbageBackend(_Backend):
    async def generate(self, prompt, *, intent=None, size_hint=""):
        self.calls += 1
        return [
            ImageResult(
                data=b"<html>not an image</html>",
                mime="image/png",
                ext="png",
                size="1024x1024",
                model="fake",
            )
        ]


def test_http_200_without_an_image_is_a_failure(plugin_module, tmp_path):
    backend = _EmptyBackend()
    plugin = _prepare(plugin_module, tmp_path, {"p1": backend})

    with pytest.raises(RuntimeError, match="可用的图片"):
        asyncio.run(plugin._run_image_chain(_record(["p1"]), kind="generate"))

    row = plugin._stats.snapshot()[0]
    assert row["failures"] == 1
    assert row["successes"] == 0


def test_bytes_that_are_not_an_image_do_not_count_as_success(plugin_module, tmp_path):
    backend = _GarbageBackend()
    plugin = _prepare(plugin_module, tmp_path, {"p1": backend})

    with pytest.raises(RuntimeError, match="可用的图片"):
        asyncio.run(plugin._run_image_chain(_record(["p1"]), kind="generate"))

    assert plugin._stats.snapshot()[0]["successes"] == 0


def test_video_url_must_be_fetchable(plugin_module, tmp_path):
    plugin = _prepare(plugin_module, tmp_path, {"p1": _Backend()})
    with pytest.raises(ValueError):
        asyncio.run(plugin._download_video(""))
    with pytest.raises(ValueError):
        asyncio.run(plugin._download_video("http://127.0.0.1:9/nope.mp4"))


@pytest.mark.parametrize("data", [b"\x89PNG", b"\x89PNG\r\n\x1a\n", b"GIF89a", b"RIFF0000WEBP"])
def test_truncated_image_signatures_are_rejected(plugin_module, data):
    assert not plugin_module._looks_like_image(data)


def test_valid_image_is_decodable(plugin_module):
    assert plugin_module._looks_like_image(_png())
    assert not plugin_module._looks_like_image(_png()[:40])


def test_gallery_failure_is_not_also_a_success(plugin_module, tmp_path):
    plugin = _prepare(plugin_module, tmp_path, {"p1": _Backend()})

    async def fail_save(*args, **kwargs):
        raise OSError("disk full")

    plugin._gallery.save_image = fail_save
    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(plugin._run_image_chain(_record(["p1"]), kind="generate"))
    row = plugin._stats.snapshot()[0]
    assert (row["attempts"], row["successes"], row["failures"]) == (1, 0, 1)
    assert plugin._consecutive_failures("p1") == 1


@pytest.mark.parametrize("download_ok", [True, False])
def test_video_success_requires_saved_media(plugin_module, tmp_path, download_ok):
    plugin = _prepare(plugin_module, tmp_path, {"p1": _Backend()})

    async def generate(*args, **kwargs):
        return SimpleNamespace(url="https://example.com/clip.mp4", model="m", size="720p", duration=5)

    async def download(url):
        if not download_ok:
            raise ValueError("invalid video")
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"saved")
        return path

    plugin._registry.get_video_backend = lambda pid: SimpleNamespace(generate_video=generate)
    plugin._download_video = download
    if download_ok:
        result = asyncio.run(plugin._run_video(_record(["p1"])))
        assert result["media"] == [str(tmp_path / "clip.mp4")]
        assert not result["url"]
    else:
        with pytest.raises(RuntimeError, match="invalid video"):
            asyncio.run(plugin._run_video(_record(["p1"])))
    row = plugin._stats.snapshot()[0]
    assert row["successes"] == int(download_ok)
    assert row["failures"] == int(not download_ok)


@pytest.mark.parametrize("broken", [{"attempts": "bad"}, {"latencies": 5},
                                    {"latencies": ["NaN"]}, {"cooling_until": "Infinity"}])
def test_corrupt_stats_row_does_not_break_initialization(tmp_path, broken):
    (tmp_path / "provider_stats.json").write_text(json.dumps({"providers": {
        "broken": broken, "good": {"attempts": 1, "successes": 1}
    }}), encoding="utf-8")
    stats = ProviderStats(tmp_path)
    assert [row["id"] for row in stats.snapshot()] == ["good"]


@pytest.mark.parametrize("payload", [{"media": ["clip.mp4"]}, {"url": "https://example.com/old.mp4"}])
def test_video_notification_matches_framework_constructor(plugin_module, tmp_path, payload):
    plugin = _prepare(plugin_module, tmp_path, {"p1": _Backend()})
    plugin._ready = True
    sent = []

    async def send(sid, chain):
        sent.extend(chain)

    async def publish(*args, **kwargs):
        return True

    plugin._safe_send = send
    plugin._publish_notification = publish
    record = _record(["p1"])
    record.status = "succeeded"
    record.result = json.dumps(payload)
    asyncio.run(plugin._on_task_finished(record))
    assert len(sent) == 1
    assert sent[0].kw["file"] == (payload.get("url") or payload["media"][0])
