"""Short-window duplicate suppression for generation requests.

The chat model occasionally calls the same tool twice for one message (a
retry inside its own loop, or a batch that repeats an entry). A duplicate is
only recognised when the same person asks for the same kind, prompt and
output in a very short window, so normal rapid-fire requests still go
through.
"""

from __future__ import annotations

import time
from typing import Any


class Debouncer:
    """Remembers recent requests and reports the task id of a duplicate."""

    def __init__(self, seconds: float = 3.0, *, ttl: float = 300.0) -> None:
        self._seconds = max(0.0, float(seconds or 0))
        self._ttl = max(self._seconds, float(ttl or 300.0))
        self._records: dict[str, tuple[float, str]] = {}

    @property
    def enabled(self) -> bool:
        return self._seconds > 0

    def configure(self, seconds: float) -> None:
        self._seconds = max(0.0, float(seconds or 0))

    @staticmethod
    def make_key(*parts: Any) -> str:
        return "\x1f".join(str(part or "") for part in parts)

    def duplicate_of(self, key: str) -> str:
        """Return the task id of a recent identical request, or ""."""
        if not self.enabled or not key:
            return ""
        now = time.monotonic()
        self._prune(now)
        entry = self._records.get(key)
        if entry is None:
            return ""
        at, task_id = entry
        if now - at <= self._seconds:
            return task_id
        return ""

    def remember(self, key: str, task_id: str) -> None:
        if not self.enabled or not key:
            return
        now = time.monotonic()
        self._prune(now)
        self._records[key] = (now, str(task_id or ""))

    def clear(self) -> None:
        self._records.clear()

    def _prune(self, now: float) -> None:
        if len(self._records) < 64:
            return
        expired = [key for key, (at, _task) in self._records.items() if now - at > self._ttl]
        for key in expired:
            self._records.pop(key, None)
