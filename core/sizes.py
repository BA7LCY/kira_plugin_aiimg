"""Size and aspect-ratio normalization shared by image providers.

Handles the common size spellings (pixels, K/4K shorthand, ratio + K), keeps
ratio canonicalization in one place, and provides fallback rules when a
provider rejects the requested size.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from math import gcd

from .output_spec import OutputIntent, format_output_intent


def normalize_size_text(size: str | None) -> str:
    return str(size or "").strip().lower().replace("×", "x")


SUPPORTED_RATIOS: dict[str, list[str]] = {
    "1:1": ["256x256", "512x512", "1024x1024", "2048x2048"],
    "4:3": ["1152x896", "2048x1536"],
    "3:4": ["768x1024", "1536x2048"],
    "3:2": ["2048x1360"],
    "2:3": ["1360x2048"],
    "16:9": ["1024x576", "2048x1152"],
    "9:16": ["576x1024", "1152x2048"],
}

_CANONICAL_RATIO_VALUES = {
    ratio: (int(ratio.split(":")[0]) / int(ratio.split(":")[1]))
    for ratio in SUPPORTED_RATIOS
}


def build_supported_sizes() -> list[str]:
    sizes: list[str] = []
    for size_list in SUPPORTED_RATIOS.values():
        for size in size_list:
            s = normalize_size_text(size)
            if s and s not in sizes:
                sizes.append(s)
    return sizes


SUPPORTED_SIZES = build_supported_sizes()


def normalize_ratio_default_sizes(
    raw: dict | None,
    *,
    supported_ratios: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Normalize ratio->size overrides and drop unsupported entries."""
    if not isinstance(raw, dict):
        return {}
    supported = supported_ratios or SUPPORTED_RATIOS
    out: dict[str, str] = {}
    for ratio, size in raw.items():
        r = str(ratio or "").strip()
        s = normalize_size_text(size)
        if not r or not s:
            continue
        sizes = [
            normalize_size_text(item)
            for item in supported.get(r, [])
            if item is not None
        ]
        if sizes and s in sizes:
            out[r] = s
    return out


def resolve_ratio_size(
    ratio: str | None,
    *,
    overrides: dict[str, str] | None = None,
    supported_ratios: dict[str, list[str]] | None = None,
) -> tuple[str, str | None]:
    """Resolve ratio to size with optional overrides.

    Returns (size, warning). warning is None when no fallback was needed.
    """
    ratio_text = str(ratio or "").strip()
    supported = supported_ratios or SUPPORTED_RATIOS
    sizes = [
        normalize_size_text(item)
        for item in supported.get(ratio_text, [])
        if normalize_size_text(item)
    ]
    if not sizes:
        return "", f"unsupported ratio '{ratio_text}'"

    override = normalize_size_text((overrides or {}).get(ratio_text, ""))
    if override:
        if override in sizes:
            return override, None
        return sizes[0], f"ratio_default_sizes[{ratio_text}]='{override}' not supported"

    return sizes[0], None


def _canonicalize_ratio_text(ratio: str | None) -> str | None:
    if not ratio:
        return None
    m = re.fullmatch(r"(\d{1,4}):(\d{1,4})", str(ratio).strip())
    if not m:
        return None
    a = int(m.group(1))
    b = int(m.group(2))
    if a <= 0 or b <= 0:
        return None
    value = a / b

    exact = f"{a}:{b}"
    if exact in _CANONICAL_RATIO_VALUES:
        return exact

    best_key = None
    best_diff = 10.0
    for key, v in _CANONICAL_RATIO_VALUES.items():
        d = abs(v - value)
        if d < best_diff:
            best_key = key
            best_diff = d

    if best_key is not None and best_diff <= 0.03:
        return best_key
    return exact


def size_to_ratio(size: str | None) -> str | None:
    s = normalize_size_text(size)
    if not s:
        return None
    m = re.fullmatch(r"(\d{2,5})x(\d{2,5})", s)
    if not m:
        return None
    w = int(m.group(1))
    h = int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    g = gcd(w, h)
    raw = f"{w // g}:{h // g}"
    return _canonicalize_ratio_text(raw)


def ratio_defaults_from_sizes(sizes: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for size in sizes:
        ratio = size_to_ratio(size)
        if ratio and ratio not in out:
            out[ratio] = size
    return out


def _closest_size(target: str, allowed: list[str]) -> str:
    """Pick the allowed size whose area is nearest to the target area."""
    pattern = re.compile(r"(\d+)x(\d+)")

    def area(size: str) -> int:
        m = pattern.fullmatch(size)
        return (int(m.group(1)) * int(m.group(2))) if m else 0

    wanted = area(target)
    if not wanted:
        return allowed[0]
    return min(allowed, key=lambda size: abs(area(size) - wanted))


def resolve_provider_size(
    intent: OutputIntent,
    *,
    size_hint: str = "",
    default_size: str = "",
    allowed_sizes: list[str] | None = None,
    allowed_ratios: dict[str, list[str]] | None = None,
    ratio_overrides: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Resolve an output intent into a concrete size string for a provider.

    Priority: explicit exact size -> size hint -> provider default -> ratio
    table -> empty (provider decides). Returns (size, warnings).

    Args:
        intent: Parsed output intent (exact size / ratio / resolution).
        size_hint: Provider or caller supplied concrete size, e.g. "1024x1024".
        default_size: Provider configured default, e.g. "4096x4096".
        allowed_sizes: If given, the only sizes the provider accepts.
        allowed_ratios: If given, ratio->size table for this provider.
        ratio_overrides: ratio_default_sizes style overrides.
    """
    warnings: list[str] = []

    if intent.exact_size:
        target = intent.exact_size
        if allowed_sizes:
            if normalize_size_text(target) in [normalize_size_text(s) for s in allowed_sizes]:
                return target, warnings
            fallback = _closest_size(target, allowed_sizes)
            warnings.append(
                f"size '{target}' not supported by provider, using '{fallback}'"
            )
            return fallback, warnings
        return target, warnings

    ratio = intent.aspect_ratio
    resolution = intent.resolution
    hint = normalize_size_text(size_hint)
    if hint and not allowed_sizes:
        return hint, warnings
    if hint and allowed_sizes:
        if hint in [normalize_size_text(s) for s in allowed_sizes]:
            return hint, warnings

    if ratio:
        if allowed_ratios:
            size, warn = resolve_ratio_size(
                ratio,
                overrides=ratio_overrides,
                supported_ratios=allowed_ratios,
            )
            if warn:
                warnings.append(warn)
            if size:
                return size, warnings
        if allowed_sizes:
            ratio_sizes = [
                s for s in allowed_sizes if size_to_ratio(s) == ratio
            ]
            if ratio_sizes:
                return ratio_sizes[-1], warnings
        return "", warnings

    if resolution:
        if allowed_sizes:
            # Pick the largest allowed size whose longer edge fits the tier.
            tiers = {"1K": 1024, "2K": 2048, "4K": 4096}
            cap = tiers.get(resolution.upper())
            if cap:
                fits = [
                    s
                    for s in allowed_sizes
                    if _size_long_edge(s) <= cap
                ]
                if fits:
                    return max(fits, key=lambda s: _size_area(s)), warnings
        return "", warnings

    if default_size:
        return default_size, warnings
    return "", warnings


def _size_long_edge(size: str) -> int:
    m = re.fullmatch(r"(\d+)x(\d+)", normalize_size_text(size))
    return max(int(m.group(1)), int(m.group(2))) if m else 0


def _size_area(size: str) -> int:
    m = re.fullmatch(r"(\d+)x(\d+)", normalize_size_text(size))
    return (int(m.group(1)) * int(m.group(2))) if m else 0


def size_request_text(
    *,
    output: str = "",
    aspect_ratio: str = "",
    resolution: str = "",
) -> str:
    """Combine explicit output controls into the request size text."""
    parts = []
    for value in (output, aspect_ratio, resolution):
        text = str(value or "").strip()
        if text and text.lower() not in {"auto", "default", "默认"}:
            parts.append(text)
    return " ".join(parts) if parts else format_output_intent(OutputIntent())
