"""Per-provider health, latency and circuit-breaker state.

The numbers are deliberately small: a rolling window of the last calls per
provider is enough to tell a healthy endpoint from one that is degrading, and
it keeps the state file cheap to rewrite on every task.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from pathlib import Path
from typing import Any

WINDOW = 50
DEFAULT_THRESHOLD = 3
DEFAULT_COOLDOWN = 120.0


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return round(ordered[index], 1)


class ProviderStats:
    """Rolling stats plus the cooldown window used by the chain runner."""

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        threshold: int = DEFAULT_THRESHOLD,
        cooldown: float = DEFAULT_COOLDOWN,
        window: int = WINDOW,
    ):
        self._path = Path(data_dir) / "provider_stats.json" if data_dir else None
        self._window = max(5, int(window))
        self._threshold = max(1, int(threshold))
        self._cooldown = max(1.0, float(cooldown))
        self._rows: dict[str, dict[str, Any]] = {}
        self._cool_until: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._load()

    # -- persistence -----------------------------------------------------

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        rows = raw.get("providers") if isinstance(raw, dict) else None
        if not isinstance(rows, dict):
            return
        for provider_id, row in rows.items():
            if not isinstance(provider_id, str) or not isinstance(row, dict):
                continue
            try:
                parsed = {
                    "attempts": int(row.get("attempts") or 0),
                    "successes": int(row.get("successes") or 0),
                    "failures": int(row.get("failures") or 0),
                    "consecutive_failures": int(row.get("consecutive_failures") or 0),
                    "latencies": [float(v) for v in (row.get("latencies") or [])][-self._window :],
                    "last_error": str(row.get("last_error") or ""),
                    "last_seen": float(row.get("last_seen") or 0),
                }
                cooldown_until = float(row.get("cooling_until") or 0)
                numbers = [parsed[key] for key in (
                    "attempts", "successes", "failures", "consecutive_failures", "last_seen"
                )] + parsed["latencies"] + [cooldown_until]
                if any(not math.isfinite(value) or value < 0 for value in numbers):
                    continue
            except (TypeError, ValueError, OverflowError):
                # A corrupt optional metrics row must not prevent plugin startup.
                continue
            self._rows[provider_id] = parsed
            if cooldown_until > time.time():
                self._cool_until[provider_id] = cooldown_until

    def _write(self) -> None:
        if self._path is None:
            return
        rows: dict[str, dict[str, Any]] = {}
        for provider_id, row in self._rows.items():
            rows[provider_id] = {
                **row,
                "cooling_until": self._cool_until.get(provider_id, 0),
            }
        payload = json.dumps({"providers": rows}, ensure_ascii=False, indent=2)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass

    # -- recording -------------------------------------------------------

    def configure(self, *, threshold: int | None = None, cooldown: float | None = None) -> None:
        if threshold is not None:
            self._threshold = max(1, int(threshold))
        if cooldown is not None:
            self._cooldown = max(1.0, float(cooldown))

    def _row(self, provider_id: str) -> dict[str, Any]:
        row = self._rows.get(provider_id)
        if row is None:
            row = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "consecutive_failures": 0,
                "latencies": [],
                "last_error": "",
                "last_seen": 0.0,
            }
            self._rows[provider_id] = row
        return row

    async def record_success(self, provider_id: str, elapsed_ms: float) -> None:
        async with self._lock:
            row = self._row(provider_id)
            row["attempts"] += 1
            row["successes"] += 1
            row["consecutive_failures"] = 0
            row["latencies"] = (row["latencies"] + [round(elapsed_ms, 1)])[-self._window :]
            row["last_seen"] = time.time()
            self._cool_until.pop(provider_id, None)
            self._write()

    async def record_failure(self, provider_id: str, elapsed_ms: float, error: str = "") -> None:
        async with self._lock:
            row = self._row(provider_id)
            row["attempts"] += 1
            row["failures"] += 1
            row["consecutive_failures"] += 1
            row["latencies"] = (row["latencies"] + [round(elapsed_ms, 1)])[-self._window :]
            row["last_error"] = str(error or "")[:400]
            row["last_seen"] = time.time()
            if row["consecutive_failures"] >= self._threshold:
                self._cool_until[provider_id] = time.time() + self._cooldown
            self._write()

    # -- queries ---------------------------------------------------------

    def cooling_seconds(self, provider_id: str) -> float:
        until = self._cool_until.get(provider_id)
        if not until:
            return 0.0
        remaining = until - time.time()
        if remaining <= 0:
            self._cool_until.pop(provider_id, None)
            return 0.0
        return remaining

    def snapshot(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for provider_id, row in sorted(self._rows.items()):
            attempts = int(row["attempts"])
            successes = int(row["successes"])
            latencies = list(row["latencies"])
            rows.append(
                {
                    "id": provider_id,
                    "attempts": attempts,
                    "successes": successes,
                    "failures": int(row["failures"]),
                    "success_rate": round(successes / attempts, 3) if attempts else None,
                    "p50_ms": percentile(latencies, 0.5) if latencies else None,
                    "p95_ms": percentile(latencies, 0.95) if latencies else None,
                    "consecutive_failures": int(row["consecutive_failures"]),
                    "cooling_seconds": round(self.cooling_seconds(provider_id), 1),
                    "last_error": row["last_error"],
                    "last_seen": row["last_seen"],
                }
            )
        return rows
