"""Regression coverage for provider retry indexing and safe diagnostic logs."""

import logging
from types import SimpleNamespace

import pytest

from core import http
from core.providers import openai_chat, openai_compat, openai_full_url
from core.providers.base import ProviderSettings


class Response:
    def __init__(self, status=200, body=b"{}", headers=None):
        self.status = status
        self.body = body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        return self.body


class Session:
    def __init__(self):
        self.responses = []
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


@pytest.fixture(params=["chat", "compat_json", "compat_form", "full_url"])
def request_case(request, monkeypatch):
    settings = ProviderSettings(
        id="regression", base_url="https://example.test/v1", api_keys=["key-secret"],
        model="image-model", max_retries=3,
    )
    kind = request.param
    if kind == "chat":
        module = openai_chat
        backend = module.OpenAIChatImageBackend(settings)
    elif kind == "full_url":
        module = openai_full_url
        backend = module.OpenAIFullURLBackend(settings)
    else:
        module = openai_compat
        backend = module.OpenAICompatBackend(settings)
    session = Session()
    delays = []
    attempts = []

    async def session_get():
        return session

    async def sleep(delay):
        delays.append(delay)

    def backoff(attempt):
        attempts.append(attempt)
        return http.backoff_delay(attempt)

    monkeypatch.setattr(backend, "_session_get", session_get)
    monkeypatch.setattr(module, "backoff_delay", backoff)
    monkeypatch.setattr(http.random, "random", lambda: 0.5)
    if module is openai_compat:
        monkeypatch.setattr(module, "asyncio_sleep", sleep)
    else:
        # Patch the module reference, not the shared asyncio.sleep used by pytest.
        monkeypatch.setattr(module, "asyncio", SimpleNamespace(sleep=sleep))

    async def invoke(retries=3, url="https://example.test/v1/images"):
        backend.max_retries = retries
        if kind == "chat":
            monkeypatch.setattr(backend, "_chat_completions_url", lambda: url)
            return await backend._post_chat(
                key="key-secret", model="image-model", messages=[],
                extra_body={}, stream=False,
            )
        if kind == "full_url":
            return await backend._post_json(url, {})
        if kind == "compat_form":
            return await backend._post_form(
                url, "key-secret", {"prompt": "test"},
                [("image", "input.png", b"image-bytes", "image/png")], retries=retries,
            )
        return await backend._post_json(url, "key-secret", {}, retries=retries)

    return SimpleNamespace(
        session=session, delays=delays, attempts=attempts, invoke=invoke,
        logger=module.logger,
    )


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
async def test_backoff_uses_zero_based_retry_index(request_case, status):
    case = request_case
    case.session.responses = [Response(status) for _ in range(3)] + [Response()]

    await case.invoke()

    assert case.attempts == [0, 1, 2]
    assert case.delays == [0.5, 1.0, 2.0]
    assert len(case.session.requests) == 4


@pytest.mark.parametrize("retry_after, expected", [("0", 0.0), ("7.5", 7.5)])
async def test_retry_after_overrides_backoff_without_resetting_index(request_case, retry_after, expected):
    case = request_case
    case.session.responses = [
        Response(429, headers={"Retry-After": retry_after}),
        Response(503), Response(),
    ]

    await case.invoke()

    assert case.attempts == [1]
    assert case.delays == [expected, 1.0]
    assert len(case.session.requests) == 3


async def test_invalid_retry_after_falls_back_to_exponential_delay(request_case):
    case = request_case
    case.session.responses = [
        Response(503, headers={"Retry-After": "not-a-number"}), Response(),
    ]

    await case.invoke()

    assert case.attempts == [0]
    assert case.delays == [0.5]


@pytest.mark.parametrize("retries", [0, 2])
async def test_exhausted_budget_does_not_add_a_sleep(request_case, retries):
    case = request_case
    case.session.responses = [Response(503, b"upstream busy") for _ in range(retries + 1)]

    with pytest.raises(RuntimeError, match="HTTP 503: upstream busy"):
        await case.invoke(retries=retries)

    assert case.attempts == list(range(retries))
    assert len(case.delays) == retries
    assert len(case.session.requests) == retries + 1


async def test_nonretryable_errors_keep_original_detail_without_sleep(request_case):
    case = request_case
    case.session.responses = [Response(400, b"unsupported size")]

    with pytest.raises(RuntimeError, match="HTTP 400: unsupported size"):
        await case.invoke()

    assert case.attempts == []
    assert case.delays == []
    assert len(case.session.requests) == 1


async def test_retry_and_final_logs_do_not_expose_endpoint_or_response_secrets(request_case, caplog):
    case = request_case
    url = "https://user-secret:password-secret@example.test/path-secret?sig=query-secret#fragment-secret"
    detail = b"Authorization: Bearer key-secret https://cdn.test/image.png?sig=body-secret"
    case.session.responses = [Response(503, detail), Response(503, detail)]

    with caplog.at_level(logging.WARNING, logger=case.logger.name):
        with pytest.raises(RuntimeError, match="HTTP 503"):
            await case.invoke(retries=1, url=url)

    messages = [record.getMessage() for record in caplog.records if record.name == case.logger.name]
    assert len(messages) == 2
    for message in messages:
        assert "503" in message
        assert "secret" not in message
        assert "example.test" not in message
        assert "cdn.test" not in message
    assert "1/1" in messages[0]
    assert len(case.session.requests) == 2
    # Redaction is log-only: transport still receives the original endpoint/key.
    assert case.session.requests[0][0] == url
    assert case.session.requests[0][1]["headers"]["Authorization"] == "Bearer key-secret"
