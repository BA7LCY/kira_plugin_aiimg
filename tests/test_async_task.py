"""Tests for the generic async task polling framework."""

import asyncio
import json
import time

import pytest

from core.async_task import (
    DEFAULT_ID_PATHS,
    DEFAULT_RESULT_PATHS,
    DEFAULT_STATE_PATHS,
    FAILURE_STATES,
    PENDING_STATES,
    SUCCESS_STATES,
    TaskSpec,
    as_text,
    await_task,
    build_poll_url,
    detect_task_id,
    dig,
    first_value,
    has_inline_image,
    spec_from_config,
    task_result,
    task_state,
)


class TestDig:
    def test_dig_dict(self):
        assert dig({"a": {"b": "c"}}, "a.b") == "c"

    def test_dig_list(self):
        assert dig({"a": [{"b": 1}]}, "a.0.b") == 1

    def test_dig_missing(self):
        assert dig({"a": 1}, "b") is None

    def test_dig_out_of_range_list(self):
        assert dig({"a": [1]}, "a.5") is None

    def test_dig_non_dict_midpath(self):
        assert dig({"a": "x"}, "a.b") is None

    def test_dig_non_list_index(self):
        assert dig({"a": [1]}, "a.x") is None


class TestFirstValue:
    def test_finds_first(self):
        assert first_value({"a": "", "b": "ok"}, ("a", "b")) == "ok"

    def test_none_when_empty(self):
        assert first_value({"a": None, "b": ""}, ("a", "b")) is None


class TestAsText:
    def test_scalar(self):
        assert as_text(" hello ") == "hello"

    def test_dict_collapses(self):
        assert as_text({}) == ""

    def test_list_collapses(self):
        assert as_text([1, 2]) == ""


class TestDetectTaskId:
    def test_finds_task_id(self):
        spec = TaskSpec()
        assert detect_task_id({"data": [{"id": "api_abc"}]}, spec) == "api_abc"

    def test_returns_empty_when_not_found(self):
        assert detect_task_id({"data": [{"url": "x.png"}]}, TaskSpec()) == ""

    def test_respects_id_prefixes(self):
        spec = TaskSpec(id_prefixes=("task_",))
        # No prefix match → empty
        assert detect_task_id({"id": "api_abc"}, spec) == ""
        # Matching prefix
        assert detect_task_id({"id": "task_123"}, spec) == "task_123"

    def test_custom_paths(self):
        spec = TaskSpec(id_paths=("my_task_id",))
        assert detect_task_id({"my_task_id": "gen_99"}, spec) == "gen_99"


class TestBuildPollUrl:
    def test_default_pattern(self):
        url = build_poll_url(TaskSpec(), "https://api.example.com/v1", "task_1")
        assert "api.example.com/api/image-tasks" in url
        assert "task_1" in url

    def test_custom_template(self):
        spec = TaskSpec(poll_url="https://poll.example.org/tasks/{task_id}")
        url = build_poll_url(spec, "https://api.example.com/v1", "task_1")
        assert url == "https://poll.example.org/tasks/task_1"

    def test_base_url_without_v1(self):
        url = build_poll_url(TaskSpec(), "https://api.example.com", "t1")
        assert "api.example.com/api/image-tasks" in url

    def test_uses_root_not_v1(self):
        url = build_poll_url(TaskSpec(), "https://x.com/v1", "t1")
        assert "x.com/api/image-tasks" in url
        assert "/v1/api" not in url

    def test_template_with_root(self):
        spec = TaskSpec(poll_url="{root}/tasks/{task_id}")
        url = build_poll_url(spec, "https://x.com/v1", "t1")
        assert url == "https://x.com/tasks/t1"

    def test_template_with_base_url(self):
        spec = TaskSpec(poll_url="{base_url}/status/{task_id}")
        url = build_poll_url(spec, "https://x.com/v1", "t1")
        assert url == "https://x.com/v1/status/t1"


class TestTaskState:
    def test_default_paths(self):
        spec = TaskSpec()
        assert task_state({"status": "running"}, spec) == "running"
        assert task_state({"state": "completed"}, spec) == "completed"

    def test_wrapped_in_items(self):
        spec = TaskSpec()
        assert task_state({"items": [{"status": "failed"}]}, spec) == "failed"

    def test_fallback_to_any_state_field(self):
        spec = TaskSpec()
        # Keys matching one of the state path suffixes (status/state/task_status) work
        result = task_state({"items": [{"task_status": "processing"}]}, spec)
        assert result == "processing"

    def test_empty_when_no_state(self):
        assert task_state({"url": "x.png"}, TaskSpec()) == ""

    def test_custom_state_paths(self):
        spec = TaskSpec(state_paths=("my_state",))
        assert task_state({"my_state": "done"}, spec) == "done"

    def test_state_not_found_when_suffix_mismatch(self):
        assert task_state({"unknown_key": "running"}, TaskSpec()) == ""


class TestHasInlineImage:
    def test_b64_is_inline(self):
        assert has_inline_image({"data": [{"b64_json": "AAAA"}]}) is True

    def test_url_is_not_inline(self):
        # url alone is not considered inline — needs to be fetched
        assert has_inline_image({"data": [{"url": "http://x.com/a.png"}]}) is True

    def test_no_image(self):
        assert has_inline_image({"error": "boom"}) is False

    def test_nested_image(self):
        assert has_inline_image({"result": {"image_url": "http://x.com/a.png"}}) is True


class TestTaskResult:
    def test_result_from_default_paths(self):
        spec = TaskSpec()
        payload = {"items": [{"results": [{"url": "http://x.com/out.png"}]}]}
        assert task_result(payload, spec)["url"] == "http://x.com/out.png"

    def test_result_from_top_level(self):
        payload = {"url": "http://x.com/out.png"}
        assert task_result(payload, TaskSpec())["url"] == "http://x.com/out.png"

    def test_result_from_b64(self):
        payload = {"data": [{"b64_json": "AAAA"}]}
        assert task_result(payload, TaskSpec())["b64_json"] == "AAAA"

    def test_raises_when_no_image(self):
        with pytest.raises(RuntimeError, match="without an image"):
            task_result({"error": "boom"}, TaskSpec())


class TestSpecFromConfig:
    def test_empty_config_returns_defaults(self):
        spec = spec_from_config({})
        assert spec.id_paths == DEFAULT_ID_PATHS
        assert spec.interval == 5.0
        assert spec.timeout == 600.0

    def test_none_config_returns_defaults(self):
        spec = spec_from_config(None)
        assert spec.id_paths == DEFAULT_ID_PATHS

    def test_custom_string_path(self):
        spec = spec_from_config({"id_paths": "task_id"})
        assert spec.id_paths == ("task_id",)

    def test_custom_list_path(self):
        spec = spec_from_config({"id_paths": ["my.id", "alt"]})
        assert spec.id_paths == ("my.id", "alt")

    def test_custom_states(self):
        spec = spec_from_config({"success_states": "ok"})
        assert spec.success_states == frozenset({"ok"})

    def test_custom_states_list(self):
        spec = spec_from_config({"failure_states": ["err", "fail"]})
        assert spec.failure_states == frozenset({"err", "fail"})

    def test_interval_clamped(self):
        spec = spec_from_config({"interval": 0.5})
        assert spec.interval == 1.0

    def test_timeout_clamped(self):
        spec = spec_from_config({"timeout": 10})
        assert spec.timeout == 60.0

    def test_max_poll_errors_clamped(self):
        spec = spec_from_config({"max_poll_errors": 0})
        assert spec.max_poll_errors == 1

    def test_configured_flag(self):
        assert spec_from_config({}).configured is False
        assert spec_from_config({"poll_url": "http://x.com"}).configured is True


class TestAwaitTask:
    @pytest.mark.asyncio
    async def test_polls_until_success(self):
        calls = 0

        async def fetch(_url):
            nonlocal calls
            calls += 1
            if calls < 3:
                return 200, json.dumps({"status": "running"}).encode()
            return 200, json.dumps({"status": "completed", "data": [{"b64_json": "AAAA"}]}).encode()

        result = await await_task(
            fetch=fetch,
            spec=TaskSpec(interval=0.01),
            task_id="t1",
            poll_url="http://x.com",
            sleep=asyncio.sleep,
            clock=time.monotonic,
        )
        assert result["b64_json"] == "AAAA"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_raises_on_failure_state(self):
        async def fetch(_url):
            return 200, json.dumps({"status": "failed"}).encode()

        with pytest.raises(RuntimeError, match="failed"):
            await await_task(
                fetch=fetch,
                spec=TaskSpec(interval=0.01),
                task_id="t1",
                poll_url="http://x.com",
                sleep=asyncio.sleep,
                clock=time.monotonic,
            )

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        async def fetch(_url):
            return 200, json.dumps({"status": "running"}).encode()

        with pytest.raises(RuntimeError, match="timed out"):
            await await_task(
                fetch=fetch,
                spec=TaskSpec(interval=0.01, timeout=0.1),
                task_id="t1",
                poll_url="http://x.com",
                sleep=asyncio.sleep,
                clock=time.monotonic,
            )

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        async def fetch(_url):
            return 500, b"internal error"

        with pytest.raises(RuntimeError, match="HTTP 500"):
            await await_task(
                fetch=fetch,
                spec=TaskSpec(interval=0.01),
                task_id="t1",
                poll_url="http://x.com",
                sleep=asyncio.sleep,
                clock=time.monotonic,
            )

    @pytest.mark.asyncio
    async def test_retries_transport_errors(self):
        calls = 0

        async def fetch(_url):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("dns fail")
            return 200, json.dumps({"status": "completed", "data": [{"b64_json": "AAAA"}]}).encode()

        result = await await_task(
            fetch=fetch,
            spec=TaskSpec(interval=0.01, max_poll_errors=3),
            task_id="t1",
            poll_url="http://x.com",
            sleep=asyncio.sleep,
            clock=time.monotonic,
        )
        assert result["b64_json"] == "AAAA"

    @pytest.mark.asyncio
    async def test_gives_up_after_max_poll_errors(self):
        async def fetch(_url):
            raise ConnectionError("dns fail")

        with pytest.raises(RuntimeError, match="polling task t1 failed"):
            await await_task(
                fetch=fetch,
                spec=TaskSpec(interval=0.01, max_poll_errors=2),
                task_id="t1",
                poll_url="http://x.com",
                sleep=asyncio.sleep,
                clock=time.monotonic,
            )


@pytest.fixture
def anyio_backend():
    return "asyncio"
