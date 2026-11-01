"""Ordered, role-tagged index of the uploaded selfie reference photos.

The photos themselves stay in ``<data_dir>/refs``; ``index.json`` only records
their order and role. Files that predate the index are appended as ``subject``
in mtime order, so upgrading can never hide an existing photo.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .logbook import get_logger

logger = get_logger("refs")

ROLES: tuple[str, ...] = ("subject", "clothing", "style", "pose", "background")
DEFAULT_ROLE = "subject"
INDEX_NAME = "index.json"
_TEMP_NAME = f"{INDEX_NAME}.tmp"


def refs_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "refs"


def normalize_file(value: object) -> str:
    """Accept ``name.png`` or ``refs/name.png`` and return the stored form."""
    text = str(value or "").strip().replace("\\", "/")
    if text.startswith("refs/"):
        text = text[len("refs/"):]
    if not text or text.startswith(".") or "/" in text:
        return ""
    return f"refs/{text}"


def _scan(directory: Path) -> list[tuple[float, str]]:
    """Real files in mtime order, ignoring the index and dotfiles."""
    try:
        children = list(directory.iterdir())
    except OSError:
        return []
    found: list[tuple[float, str]] = []
    for path in children:
        if path.name.startswith(".") or path.name in {INDEX_NAME, _TEMP_NAME}:
            continue
        if not path.is_file():
            continue
        try:
            found.append((path.stat().st_mtime, path.name))
        except OSError:
            continue
    found.sort()
    return found


def read_index(data_dir: Path) -> list[dict]:
    """Parsed index entries; a missing or corrupt index falls back to a scan."""
    try:
        payload = json.loads(
            (refs_dir(data_dir) / INDEX_NAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    if not isinstance(payload, list):
        return []
    entries: list[dict] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        file = normalize_file(item.get("file"))
        if not file:
            continue
        role = str(item.get("role") or DEFAULT_ROLE).strip().lower()
        if role not in ROLES:
            role = DEFAULT_ROLE
        added = item.get("added_at")
        entries.append(
            {
                "file": file,
                "role": role,
                "added_at": int(added) if isinstance(added, (int, float)) else 0,
            }
        )
    return entries


def list_entries(data_dir: Path) -> list[dict]:
    """Index order first, then unindexed files by mtime as subject."""
    if data_dir is None:
        return []
    scan = _scan(refs_dir(data_dir))
    on_disk = {f"refs/{name}" for _mtime, name in scan}
    entries: list[dict] = []
    seen: set[str] = set()
    for item in read_index(data_dir):
        if item["file"] in seen or item["file"] not in on_disk:
            continue
        seen.add(item["file"])
        entries.append(item)
    for mtime, name in scan:
        file = f"refs/{name}"
        if file in seen:
            continue
        entries.append({"file": file, "role": DEFAULT_ROLE, "added_at": int(mtime)})
        seen.add(file)
    return entries


def save_entries(data_dir: Path, entries: list[dict]) -> None:
    """Write the index atomically so a crash cannot leave half a JSON file."""
    directory = refs_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "file": entry["file"],
            "role": entry["role"],
            "added_at": int(entry.get("added_at") or 0),
        }
        for entry in entries
    ]
    target = directory / INDEX_NAME
    temp = directory / _TEMP_NAME
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temp.replace(target)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["role"]] = counts.get(entry["role"], 0) + 1
    logger.info(
        "[aiimg] 参考图索引：共 %d 张（%s）",
        len(entries),
        "，".join(f"{role} {count}" for role, count in sorted(counts.items())) or "空",
    )


def append_entry(
    data_dir: Path, file: object, role: str = DEFAULT_ROLE
) -> list[dict]:
    """Uploads join the end of the list as ``subject`` unless already indexed."""
    name = normalize_file(file)
    if not name:
        raise ValueError("文件名不对")
    if role not in ROLES:
        raise ValueError("角色只能是 subject、clothing、style、pose、background")
    entries = list_entries(data_dir)
    matched = False
    for entry in entries:
        if entry["file"] == name:
            entry["role"] = role
            matched = True
    if not matched:
        entries.append({"file": name, "role": role, "added_at": int(time.time())})
    save_entries(data_dir, entries)
    return entries


def set_role(data_dir: Path, file: object, role: str) -> list[dict]:
    name = normalize_file(file)
    if role not in ROLES:
        raise ValueError("角色只能是 subject、clothing、style、pose、background")
    entries = list_entries(data_dir)
    for entry in entries:
        if entry["file"] == name:
            entry["role"] = role
            save_entries(data_dir, entries)
            return entries
    raise KeyError(name)


def reorder(data_dir: Path, names: list[object]) -> list[dict]:
    """Apply a full new order; a partial list is rejected instead of guessed."""
    entries = list_entries(data_dir)
    current = [entry["file"] for entry in entries]
    wanted = [normalize_file(item) for item in names]
    if not wanted or any(not item for item in wanted) or sorted(wanted) != sorted(current):
        raise ValueError(f"顺序必须包含全部 {len(current)} 张参考图")
    by_file = {entry["file"]: entry for entry in entries}
    ordered = [by_file[name] for name in wanted]
    save_entries(data_dir, ordered)
    return ordered


def remove_entry(data_dir: Path, file: object) -> list[dict]:
    """Deleting a photo also drops its entry, so the index never outlives a file."""
    name = normalize_file(file)
    entries = [entry for entry in list_entries(data_dir) if entry["file"] != name]
    save_entries(data_dir, entries)
    return entries
