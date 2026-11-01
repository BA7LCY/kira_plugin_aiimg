"""Logging helpers shared by the plugin.

Everything logs under ``kira_plugin_aiimg`` so one logger level covers the whole
plugin, while each module appends its own name so the console shows who spoke.
Logs stay verbose on purpose: task lifecycles, provider attempts and API calls
are all worth reading when something goes wrong.
"""

from __future__ import annotations

import logging
import time

ROOT = "kira_plugin_aiimg"


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}" if name else ROOT)


def clip(value: object, limit: int = 160) -> str:
    """Collapse whitespace and truncate, for prompt and URL previews."""
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…(+{len(text) - limit} 字)"


def mask(secret: object) -> str:
    """Show just enough of an API key to recognise it in a log line."""
    text = str(secret or "")
    if not text:
        return ""
    if len(text) <= 8:
        return "***"
    return f"{text[:3]}…{text[-3:]}"


def elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
