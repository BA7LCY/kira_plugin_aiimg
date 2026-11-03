"""Generic submit → poll support for providers that queue image jobs.

Some gateways answer ``POST /images/generations`` with a task id instead of the
picture and expect the caller to poll a second endpoint. The shape of that
handshake differs per service: where the id sits, which URL to poll, what the
state is called and where the finished image hides.

Rather than hard-coding one dialect, a ``TaskSpec`` describes it: a list of
candidate paths for each field, and defaults that match the most common
OpenAI-ish layout. Providers override any of it through the ``async_task``
config field, so a new gateway does not need new code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .sizes import normalize_size_text  # noqa: F401  (kept for symmetry imports)

DEFAULT_ID_PATHS: tuple[str, ...] = (
    "task_id",
    "taskId",
    "id",
    "data.task_id",
    "data.id",
    "data.0.task_id",
    "data.0.id",
    "result.task_id",
    "result.id",
)

DEFAULT_STATE_PATHS: tuple[str, ...] = (
    "status",
    "state",
    "task_status",
    "items.0.status",
    "items.0.state",
    "tasks.0.status",
    "tasks.0.state",
    "data.status",
    "data.state",
    "data.0.status",
    "data.0.state",
    "result.status",
    "result.state",
)

DEFAULT_RESULT_PATHS: tuple[str, ...] = (
    "items.0.results.0.url",
    "items.0.results.0.b64_json",
    "items.0.url",
    "items.0.b64_json",
    "tasks.0.results.0.url",
    "tasks.0.results.0.b64_json",
    "data.0.url",
    "data.0.b64_json",
    "data.url",
    "data.b64_json",
    "results.0.url",
    "results.0.b64_json",
    "images.0.url",
    "images.0.b64_json",
    "outputs.0.url",
    "output.url",
    "output.0.url",
    "url",
    "b64_json",
    "image_url",
)

SUCCESS_STATES = frozenset({"success", "succeeded", "completed", "complete", "done", "finished"})
FAILURE_STATES = frozenset({"failed", "failure", "error", "cancelled", "canceled", "rejected"})
PENDING_STATES = frozenset({"pending", "queued", "running", "processing", "in_progress", "submitted", "created"})


@dataclass(frozen=True)
class TaskSpec:
    """How one provider reports a queued image job."""

    id_paths: tuple[str, ...] = DEFAULT_ID_PATHS
    id_prefixes: tuple[str, ...] = ()
    poll_url: str = ""
    state_paths: tuple[str, ...] = DEFAULT_STATE_PATHS
    result_paths: tuple[str, ...] = DEFAULT_RESULT_PATHS
    success_states: frozenset[str] = SUCCESS_STATES
    failure_states: frozenset[str] = FAILURE_STATES
    interval: float = 5.0
    timeout: float = 600.0
    max_poll_errors: int = 3
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def configured(self) -> bool:
        """True when the operator described a dialect by hand."""
        return bool(self.extra)


def _split_path(path: str) -> list[str]:
    return [part for part in str(path).split(".") if part]


def dig(payload: Any, path: str) -> Any:
    """Walk a dotted path through dicts and lists; None when it does not exist."""
    current = payload
    for part in _split_path(path):
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            try:
                index = int(part)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
        else:
            return None
    return current


def first_value(payload: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = dig(payload, path)
        if value not in (None, "", [], {}):
            return value
    return None


def as_text(value: Any) -> str:
    """Scalar as a trimmed string; None and containers collapse to ""."""
    if value is None or isinstance(value, (dict, list)):
        return ""
    return str(value).strip()


def detect_task_id(payload: Any, spec: TaskSpec) -> str:
    """The queued job's id, or "" when the response already holds the picture."""
    value = first_value(payload, spec.id_paths)
    task_id = as_text(value)
    if not task_id:
        return ""
    if spec.id_prefixes and not task_id.startswith(tuple(spec.id_prefixes)):
        return ""
    return task_id


def build_poll_url(spec: TaskSpec, base_url: str, task_id: str) -> str:
    """Fill the poll template; the default matches OpenAI-ish task APIs."""
    template = str(spec.poll_url or "").strip()
    root = str(base_url or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    if not template:
        return f"{root}/api/image-tasks?ids={task_id}"
    return (
        template.replace("{task_id}", task_id)
        .replace("{base_url}", str(base_url or "").rstrip("/"))
        .replace("{root}", root)
    )


def task_state(payload: Any, spec: TaskSpec) -> str:
    value = first_value(payload, spec.state_paths)
    text = as_text(value).lower()
    if text:
        return text
    # Gateways wrap the job in arbitrary containers (items/tasks/data/…), so
    # fall back to the first state-looking field anywhere in the body.
    return _find_state(payload, spec.state_paths)


def _find_state(node: Any, state_paths: tuple[str, ...], depth: int = 0) -> str:
    if depth > 8:
        return ""
    keys = {path.rsplit(".", 1)[-1].lower() for path in state_paths}
    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).lower() in keys:
                text = as_text(value).lower()
                if text:
                    return text
        for value in node.values():
            found = _find_state(value, state_paths, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_state(item, state_paths, depth + 1)
            if found:
                return found
    return ""


def task_result(payload: Any, spec: TaskSpec) -> dict:
    """The finished job's image holder, still in provider shape.

    Providers nest this differently, so a path hit is used when it is a dict;
    otherwise the payload is searched for the first dict carrying an image.
    """
    for path in spec.result_paths:
        value = dig(payload, path)
        if isinstance(value, dict) and (
            value.get("url") or value.get("b64_json") or value.get("image_url")
        ):
            return value
        if isinstance(value, str) and value.startswith(("http://", "https://", "data:image/")):
            key = "b64_json" if value.startswith("data:image/") else "url"
            return {key: value}
    found = _find_image_holder(payload)
    if found is not None:
        return found
    raise RuntimeError("image task finished without an image")


def has_inline_image(payload: Any) -> bool:
    """True when the response already carries the picture, i.e. nothing to poll."""
    return _find_image_holder(payload) is not None


def _find_image_holder(node: Any, depth: int = 0) -> dict | None:
    if depth > 8:
        return None
    if isinstance(node, dict):
        if node.get("url") or node.get("b64_json") or node.get("image_url"):
            return node
        for value in node.values():
            found = _find_image_holder(value, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_image_holder(item, depth + 1)
            if found is not None:
                return found
    return None


def spec_from_config(raw: Any, *, base_interval: float = 5.0, base_timeout: float = 600.0) -> TaskSpec:
    """Build a spec from the ``async_task`` config object.

    Unknown keys are ignored so a config written for a newer version still
    loads, and every list accepts a single string for convenience.
    """
    if not isinstance(raw, dict):
        return TaskSpec(interval=base_interval, timeout=base_timeout)

    def paths(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return (value.strip(),)
        if isinstance(value, list):
            items = tuple(str(item).strip() for item in value if str(item).strip())
            return items or fallback
        return fallback

    def states(key: str, fallback: frozenset[str]) -> frozenset[str]:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return frozenset({value.strip().lower()})
        if isinstance(value, list):
            items = frozenset(str(item).strip().lower() for item in value if str(item).strip())
            return items or fallback
        return fallback

    def number(key: str, fallback: float, minimum: float) -> float:
        try:
            return max(minimum, float(raw.get(key)))
        except (TypeError, ValueError):
            return fallback

    return TaskSpec(
        id_paths=paths("id_paths", DEFAULT_ID_PATHS),
        id_prefixes=tuple(
            str(item).strip()
            for item in (raw.get("id_prefixes") or [])
            if str(item).strip()
        )
        if isinstance(raw.get("id_prefixes"), list)
        else (),
        poll_url=str(raw.get("poll_url") or "").strip(),
        state_paths=paths("state_paths", DEFAULT_STATE_PATHS),
        result_paths=paths("result_paths", DEFAULT_RESULT_PATHS),
        success_states=states("success_states", SUCCESS_STATES),
        failure_states=states("failure_states", FAILURE_STATES),
        interval=number("interval", base_interval, 1.0),
        timeout=number("timeout", base_timeout, 60.0),
        max_poll_errors=max(1, int(number("max_poll_errors", 3, 1))),
        extra=dict(raw),
    )


async def await_task(
    *,
    fetch: Callable[[str], Awaitable[tuple[int, bytes]]],
    spec: TaskSpec,
    task_id: str,
    poll_url: str,
    sleep: Callable[[float], Awaitable[None]],
    clock: Callable[[], float],
    on_status: Callable[[str, int], None] | None = None,
    logger: Any = None,
) -> dict:
    """Poll until the job finishes; return the raw image holder.

    ``fetch`` performs one GET and returns ``(status, body)``, so the caller
    keeps control of the HTTP session, proxy and headers. Transport errors are
    tolerated ``max_poll_errors`` times in a row before giving up.
    """
    deadline = clock() + spec.timeout
    failures = 0
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            raise RuntimeError(f"image task {task_id} timed out")
        await sleep(min(max(1.0, spec.interval), remaining))
        try:
            status, body = await fetch(poll_url)
            if status != 200:
                if on_status is not None:
                    on_status(task_id, status)
                detail = body[:200].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {status}: {detail}")
            payload = json.loads(body)
        except Exception as exc:
            failures += 1
            if failures > spec.max_poll_errors:
                raise RuntimeError(f"polling task {task_id} failed: {exc}") from exc
            if logger is not None:
                logger.warning("查询任务 %s 失败，重试中：%s", task_id, exc)
            continue
        failures = 0
        state = task_state(payload, spec)
        if state in spec.failure_states:
            raise RuntimeError(f"image task {task_id} {state}")
        if state in spec.success_states:
            return task_result(payload, spec)
        if state and state not in PENDING_STATES and logger is not None:
            logger.debug("任务 %s 返回了未知状态 %s，继续等待", task_id, state)
