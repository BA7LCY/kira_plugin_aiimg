"""Resolution-independent image output controls."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


_EXACT_RE = re.compile(r"^(\d{2,5})\s*[xX×]\s*(\d{2,5})$")
_RATIO_RE = re.compile(r"^(\d{1,2}):(\d{1,2})$")
_RESOLUTION_RE = re.compile(r"^(?:1K|2K|4K)$", re.IGNORECASE)
_PROMPT_EXACT_RE = re.compile(r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)")
_PROMPT_RATIO_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[:比/]\s*(\d{1,2})(?!\d)")
_PROMPT_RESOLUTION_RE = re.compile(r"(?<!\w)(1K|2K|4K)(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class OutputIntent:
    exact_size: str | None = None
    aspect_ratio: str | None = None
    resolution: str | None = None

    @property
    def is_empty(self) -> bool:
        return not (self.exact_size or self.aspect_ratio or self.resolution)


def _normalize_exact(value: str) -> str | None:
    match = _EXACT_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if not (16 <= width <= 8192 and 16 <= height <= 8192):
        return None
    return f"{width}x{height}"


def _normalize_ratio(value: str) -> str | None:
    match = _RATIO_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    return f"{int(match.group(1))}:{int(match.group(2))}"


def _ratio_from_dims(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def parse_output_intent(value: str | None) -> OutputIntent:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"auto", "default", "默认"}:
        return OutputIntent()
    exact = _normalize_exact(raw)
    if exact:
        return OutputIntent(exact_size=exact)
    ratio = _normalize_ratio(raw)
    if ratio:
        return OutputIntent(aspect_ratio=ratio)
    if _RESOLUTION_RE.fullmatch(raw):
        return OutputIntent(resolution=raw.upper())
    raise ValueError(f"无效输出参数: {value}")


def extract_output_intent_from_prompt(prompt: str | None) -> OutputIntent:
    text = str(prompt or "")
    exact = list(_PROMPT_EXACT_RE.finditer(text))
    if exact:
        match = exact[-1]
        return OutputIntent(exact_size=_normalize_exact(f"{match.group(1)}x{match.group(2)}"))
    ratio_match = list(_PROMPT_RATIO_RE.finditer(text))
    resolution_match = list(_PROMPT_RESOLUTION_RE.finditer(text))
    if not ratio_match and not resolution_match:
        return OutputIntent()
    ratio = _normalize_ratio(
        f"{ratio_match[-1].group(1)}:{ratio_match[-1].group(2)}"
    ) if ratio_match else None
    resolution = resolution_match[-1].group(1).upper() if resolution_match else None
    return OutputIntent(aspect_ratio=ratio, resolution=resolution)


def merge_output_intents(*intents: OutputIntent | None) -> OutputIntent:
    aspect_ratio = resolution = None
    for intent in intents:
        if intent is None or intent.is_empty:
            continue
        if intent.exact_size:
            if not aspect_ratio and not resolution:
                return intent
            width, height = map(int, intent.exact_size.split("x", 1))
            aspect_ratio = aspect_ratio or _ratio_from_dims(width, height)
            resolution = resolution or ("1K" if max(width, height) <= 1024 else "2K" if max(width, height) <= 2048 else "4K")
            continue
        aspect_ratio = aspect_ratio or intent.aspect_ratio
        resolution = resolution or intent.resolution
    return OutputIntent(aspect_ratio=aspect_ratio, resolution=resolution)


def apply_conflict_rule(intent: OutputIntent, *, prefer_resolution: bool = True) -> OutputIntent:
    """Keep one side when a concrete size and a ratio are both present.

    The user picks which side wins in the settings; nothing is filled in for
    the dropped side.
    """
    if not (intent.exact_size and intent.aspect_ratio):
        return intent
    if prefer_resolution:
        return OutputIntent(exact_size=intent.exact_size)
    return OutputIntent(aspect_ratio=intent.aspect_ratio, resolution=intent.resolution)


def resolve_output_intent(
    *,
    prompt: str,
    output: str = "",
    aspect_ratio: str = "",
    resolution: str = "",
) -> OutputIntent:
    structured: list[str] = []
    for value in (aspect_ratio, resolution):
        text = str(value or "").strip()
        if text and text.lower() not in {"auto", "default", "默认"}:
            structured.append(text)
    output_text = str(output or "").strip()
    structured_intent = parse_output_intent(" ".join(structured)) if structured else OutputIntent()
    output_intent = parse_output_intent(output_text) if output_text and output_text.lower() not in {"auto", "default", "默认"} else OutputIntent()
    return merge_output_intents(
        extract_output_intent_from_prompt(prompt),
        structured_intent,
        output_intent,
    )


def output_intent_dict(intent: OutputIntent) -> dict[str, str | None]:
    return {
        "exact_size": intent.exact_size,
        "aspect_ratio": intent.aspect_ratio,
        "resolution": intent.resolution,
    }


def format_output_intent(intent: OutputIntent | None) -> str:
    if intent is None or intent.is_empty:
        return ""
    if intent.exact_size:
        return intent.exact_size
    return " ".join(value for value in (intent.aspect_ratio, intent.resolution) if value)


def aspect_ratio_from_size(size: str | None) -> str | None:
    normalized = _normalize_exact(str(size or ""))
    if not normalized:
        return None
    width, height = map(int, normalized.split("x", 1))
    return _ratio_from_dims(width, height)


def to_request_size_args(intent: OutputIntent) -> dict[str, str]:
    if intent.exact_size:
        return {"size": intent.exact_size}
    if intent.resolution:
        return {"resolution": intent.resolution}
    return {}


def resolution_from_size(size: str | None) -> str | None:
    normalized = _normalize_exact(str(size or ""))
    if not normalized:
        return None
    width, height = map(int, normalized.split("x", 1))
    longest = max(width, height)
    if longest <= 1024:
        return "1K"
    if longest <= 2048:
        return "2K"
    return "4K"


# gpt-image-2 adaptive size table: maps a resolution token plus a ratio to a
# concrete pixel size, so short values like "4K 16:9" resolve to real sizes.
GPT_IMAGE_2_SIZE_MAP: dict[str, dict[str, str]] = {
    "1K": {
        "1:1": "1024x1024",
        "16:9": "1280x720",
        "9:16": "720x1280",
        "5:4": "1040x832",
        "4:5": "832x1040",
        "4:3": "1024x768",
        "3:4": "768x1024",
        "3:2": "1008x672",
        "2:3": "672x1008",
        "21:9": "1344x576",
    },
    "2K": {
        "1:1": "2048x2048",
        "16:9": "2048x1152",
        "9:16": "1152x2048",
        "5:4": "2080x1664",
        "4:5": "1664x2080",
        "4:3": "2048x1536",
        "3:4": "1536x2048",
        "3:2": "2064x1376",
        "2:3": "1376x2064",
        "21:9": "2016x864",
    },
    "4K": {
        "1:1": "2880x2880",
        "16:9": "3840x2160",
        "9:16": "2160x3840",
        "5:4": "3200x2560",
        "4:5": "2560x3200",
        "4:3": "3264x2448",
        "3:4": "2448x3264",
        "3:2": "3504x2336",
        "2:3": "2336x3504",
        "21:9": "3808x1632",
    },
}


def is_gpt_image_2_model(model: str | None) -> bool:
    normalized = re.sub(r"[_\s]+", "-", str(model or "").strip().lower())
    return "gpt-image-2" in normalized


def resolve_gpt_image_2_size(
    intent: OutputIntent,
    *,
    default_size: str | None = None,
    default_resolution: str = "1K",
    default_aspect_ratio: str = "1:1",
) -> str | None:
    """Resolve adaptive controls to a gpt-image-2 compatible size."""
    if intent.exact_size:
        return intent.exact_size

    resolution = str(intent.resolution or "").strip().upper()
    if intent.resolution and resolution not in GPT_IMAGE_2_SIZE_MAP:
        return None
    if resolution not in GPT_IMAGE_2_SIZE_MAP:
        resolution = resolution_from_size(default_size)
    if resolution not in GPT_IMAGE_2_SIZE_MAP:
        resolution = default_resolution.upper()
    if resolution not in GPT_IMAGE_2_SIZE_MAP:
        resolution = "1K"

    aspect_ratio = str(intent.aspect_ratio or "").strip()
    if intent.aspect_ratio and aspect_ratio not in GPT_IMAGE_2_SIZE_MAP[resolution]:
        return None
    if not aspect_ratio:
        aspect_ratio = aspect_ratio_from_size(default_size)
    if not aspect_ratio:
        aspect_ratio = default_aspect_ratio
    if aspect_ratio not in GPT_IMAGE_2_SIZE_MAP[resolution]:
        aspect_ratio = "1:1"

    return GPT_IMAGE_2_SIZE_MAP[resolution][aspect_ratio]


RATIO_SIZE_TABLE = GPT_IMAGE_2_SIZE_MAP


def size_from_ratio_and_resolution(
    aspect_ratio: str | None,
    resolution: str | None = None,
    *,
    default_size: str | None = None,
) -> str | None:
    """Concrete size for a ratio/resolution pair when a provider lists no sizes."""
    ratio = str(aspect_ratio or "").strip()
    if not ratio:
        return None
    tier = str(resolution or "").strip().upper()
    if tier not in RATIO_SIZE_TABLE:
        tier = resolution_from_size(default_size) or ""
    if tier not in RATIO_SIZE_TABLE:
        tier = "1K"
    return RATIO_SIZE_TABLE.get(tier, {}).get(ratio)


def select_allowed_size(
    intent: OutputIntent,
    allowed_sizes: list[str] | tuple[str, ...],
    *,
    default_size: str | None = None,
) -> str | None:
    """Pick a concrete size from a provider's allowed list for an intent."""
    normalized_sizes = [
        normalized
        for item in allowed_sizes
        if (normalized := _normalize_exact(item)) is not None
    ]
    if not normalized_sizes:
        return None
    if intent.exact_size:
        return intent.exact_size

    desired_ratio = intent.aspect_ratio or aspect_ratio_from_size(default_size)
    candidates = normalized_sizes
    if desired_ratio:
        same_ratio = [
            size
            for size in normalized_sizes
            if aspect_ratio_from_size(size) == desired_ratio
        ]
        if same_ratio:
            candidates = same_ratio
        elif intent.aspect_ratio:
            return None

    resolution_targets = {"1K": 1024, "2K": 2048, "4K": 4096}
    target_edge = resolution_targets.get(str(intent.resolution or "").upper())
    if target_edge is None:
        normalized_default = _normalize_exact(default_size)
        if normalized_default:
            width_text, height_text = normalized_default.split("x", 1)
            target_edge = max(int(width_text), int(height_text))
        else:
            target_edge = 1024

    def distance(size: str) -> tuple[int, int]:
        width_text, height_text = size.split("x", 1)
        width = int(width_text)
        height = int(height_text)
        return abs(max(width, height) - target_edge), abs(
            width * height - target_edge**2
        )

    return min(candidates, key=distance)
