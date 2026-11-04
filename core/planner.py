"""Split one short brief into several distinct image prompts.

The conversation model already knows what the user wants; this module only
turns that single sentence into a set of prompts that differ in framing, light
and style, and gives each one a different aspect ratio so a batch does not come
back as eight variations of the same square picture.

Everything here is pure: build a request, parse the reply, spread the ratios.
The network call lives in the plugin.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

DEFAULT_RATIOS: tuple[str, ...] = ("1:1", "3:4", "4:3", "16:9", "9:16")

DEFAULT_PLANNER_TEMPLATE = (
    "把下面这句需求拆成 {{count}} 条互不相同的出图提示词。\n"
    "要求：\n"
    "1. 每条直接描述画面，不要编号，不要解释，不要写「提示词」「画面」这类前缀。\n"
    "2. 主体保持一致，但角度、景别、光线、构图、风格至少有三项互不相同。\n"
    "3. 每条指定一个画幅，只能从 {{ratios}} 里挑，至少用两种不同画幅。\n"
    "只输出 JSON，不要别的文字：\n"
    '{"prompts": [{"prompt": "...", "aspect_ratio": "3:4"}]}\n'
    "需求：{{brief}}"
)

_PROMPT_KEYS = ("prompt", "text", "description", "caption", "title", "content")
_RATIO_KEYS = ("aspect_ratio", "ratio", "aspect", "size", "shape")
_LIST_KEYS = ("prompts", "items", "images", "results", "data", "outputs")
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_RATIO_RE = re.compile(r"^\s*(\d{1,2})\s*[:：xX×/]\s*(\d{1,2})\s*$")


@dataclass(frozen=True)
class PlannedPrompt:
    """One prompt from the plan, with the frame it should be drawn in."""

    prompt: str
    aspect_ratio: str = ""


def normalize_ratio(value: object) -> str:
    """'3x4' / '3：4' / ' 3 : 4 ' -> '3:4'; anything else -> ''."""
    match = _RATIO_RE.match(str(value or ""))
    if not match:
        return ""
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return ""
    return f"{width}:{height}"


def ratio_choices(raw: object = None) -> list[str]:
    """Allowed frames, de-duplicated and normalized; falls back to defaults."""
    items = raw if isinstance(raw, (list, tuple)) else str(raw or "").replace(",", " ").split()
    choices: list[str] = []
    for item in items:
        ratio = normalize_ratio(item)
        if ratio and ratio not in choices:
            choices.append(ratio)
    return choices or list(DEFAULT_RATIOS)


def build_request(
    template: str | None,
    *,
    brief: str,
    count: int,
    ratios: list[str],
) -> str:
    """Render the planner template; an empty template means a built-in default."""
    body = str(template or "").strip() or DEFAULT_PLANNER_TEMPLATE
    text = body
    for token, value in (
        ("{{count}}", str(max(1, int(count)))),
        ("{{ratios}}", "、".join(ratios)),
        ("{{brief}}", str(brief or "").strip()),
    ):
        text = text.replace(token, value)
    # A template that forgot the placeholder still has to carry the brief.
    if "{{brief}}" not in body and str(brief or "").strip() not in text:
        text = f"{text}\n需求：{str(brief or '').strip()}"
    return text


def _balanced_json(text: str) -> str:
    """Return the first complete top-level JSON object or array in ``text``."""
    start = -1
    opener = ""
    for index, char in enumerate(text):
        if char in "{[":
            start = index
            opener = char
            break
    if start < 0:
        return ""
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def extract_json(text: str) -> object | None:
    """Best-effort JSON out of a model reply, fenced or wrapped in prose."""
    raw = str(text or "").strip()
    if not raw:
        return None
    for candidate in (raw, *[match.group(1).strip() for match in _FENCE_RE.finditer(raw)]):
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
        # Not valid JSON on its own; fall through to the brace scan below.
    for candidate in (raw, *[match.group(1) for match in _FENCE_RE.finditer(raw)]):
        block = _balanced_json(candidate)
        if not block:
            continue
        try:
            return json.loads(block)
        except (TypeError, ValueError):
            continue
    return None


def _entries(payload: object) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _LIST_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        for key in _PROMPT_KEYS:
            if payload.get(key):
                return [payload]
    return []


def _entry_prompt(entry: object) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in _PROMPT_KEYS:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _entry_ratio(entry: object) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in _RATIO_KEYS:
        value = entry.get(key)
        if value:
            return normalize_ratio(value)
    return ""


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def parse_plan(text: str, *, count: int, ratios: list[str]) -> list[PlannedPrompt]:
    """Parse a planner reply into prompts; raises ValueError when unusable."""
    payload = extract_json(text)
    entries = _entries(payload)
    allowed = set(ratios)
    planned: list[PlannedPrompt] = []
    seen: set[str] = set()
    for entry in entries:
        prompt = _entry_prompt(entry)
        if not prompt:
            continue
        key = _key(prompt)
        if key in seen:
            continue
        seen.add(key)
        ratio = _entry_ratio(entry)
        planned.append(PlannedPrompt(prompt=prompt, aspect_ratio=ratio if ratio in allowed else ""))
        if len(planned) >= max(1, int(count)):
            break
    if not planned:
        raise ValueError("批量规划没有返回可用的提示词")
    return planned


def spread_ratios(items: list[PlannedPrompt], ratios: list[str]) -> list[PlannedPrompt]:
    """Fill missing frames and keep the batch varied.

    A frame the model asked for wins the first time it shows up. When the same
    frame comes back and the pool still has unused entries, the repeat is
    replaced, so a batch never collapses into one shape while there is room to
    differ. Once the pool is used up, repeats cycle in pool order.
    """
    choices = list(ratios) or list(DEFAULT_RATIOS)
    if not items or not choices:
        return list(items)
    result: list[PlannedPrompt] = []
    used: set[str] = set()
    cursor = 0
    for item in items:
        ratio = item.aspect_ratio if item.aspect_ratio in choices else ""
        if ratio and ratio not in used:
            used.add(ratio)
            result.append(PlannedPrompt(prompt=item.prompt, aspect_ratio=ratio))
            continue
        for offset in range(len(choices)):
            candidate = choices[(cursor + offset) % len(choices)]
            if candidate not in used:
                ratio = candidate
                cursor = (cursor + offset + 1) % len(choices)
                break
        else:
            ratio = ratio or choices[cursor % len(choices)]
            cursor = (cursor + 1) % len(choices)
        used.add(ratio)
        result.append(PlannedPrompt(prompt=item.prompt, aspect_ratio=ratio))
    return result


def plan(
    text: str,
    *,
    count: int,
    ratios: list[str],
) -> list[PlannedPrompt]:
    """Parse a reply and make sure the frames vary."""
    return spread_ratios(parse_plan(text, count=count, ratios=ratios), ratios)


def describe(items: list[PlannedPrompt]) -> str:
    """Short audit line: '3 条（1:1, 3:4, 16:9）'."""
    if not items:
        return "0 条"
    frames = "、".join(item.aspect_ratio or "默认" for item in items)
    return f"{len(items)} 条（{frames}）"
