"""Add required source/licensing credits to YouTube metadata for hybrid renders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_DESCRIPTION_CHARS = 5000
_CREDITS_MARKER = "FOOTAGE CREDITS AND LICENSES"


def append_footage_credits(metadata: dict[str, Any], video_path: str | Path) -> dict[str, Any]:
    """Return copy of upload metadata with deduplicated clip credits appended.

    Credits are read from the renderer's scene-plan sidecar, not guessed from
    the source manifest. If the license block cannot fit in YouTube's
    description limit, raise instead of silently truncating attribution.
    """
    if not isinstance(metadata, dict):
        raise TypeError("upload metadata must be a dictionary")
    scene_plan_path = Path(video_path).expanduser().with_suffix(".scene_plan.json")
    if not scene_plan_path.is_file():
        return dict(metadata)
    try:
        plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(metadata)

    rows = plan.get("scenes", []) if isinstance(plan, dict) else []
    unique: dict[tuple[str, str], str] = {}
    for row in rows if isinstance(rows, list) else []:
        footage = row.get("footage") if isinstance(row, dict) else None
        if not isinstance(footage, dict):
            continue
        title = str(footage.get("title") or "Untitled source clip").strip()
        creator = str(footage.get("attribution") or footage.get("channel") or "").strip()
        source_url = str(footage.get("source_url") or "").strip()
        license_name = str(footage.get("license") or "").strip()
        license_url = str(footage.get("license_url") or "").strip()
        if not source_url or not license_name:
            raise ValueError(f"Footage attribution is incomplete for source: {title}")
        if license_name.casefold().startswith("cc by") and (not creator or not license_url):
            raise ValueError(f"CC BY attribution is incomplete for source: {title}")
        key = (source_url, license_name + "|" + license_url)
        credit_line = f"- {title}"
        if creator:
            credit_line += f" — {creator}"
        credit_line += f"\n  Source: {source_url}\n  License: {license_name}"
        if license_url:
            credit_line += f" ({license_url})"
        unique.setdefault(key, credit_line)

    if not unique:
        return dict(metadata)

    block = _CREDITS_MARKER + "\n" + "\n".join(unique.values())
    # Reserve a separating newline between the original description and credits.
    block = "\n\n" + block
    if len(block) >= MAX_DESCRIPTION_CHARS:
        raise ValueError(
            "Footage credits alone exceed YouTube's 5,000-character description limit. "
            "Reduce the number of source clips before upload."
        )

    result = dict(metadata)
    original = str(result.get("description") or "").strip()
    if _CREDITS_MARKER in original:
        original = original.split(_CREDITS_MARKER, 1)[0].rstrip()
    available = MAX_DESCRIPTION_CHARS - len(block)
    if len(original) > available:
        if available < 2:
            raise ValueError("No room remains for required footage credits in the video description.")
        original = original[: available - 1].rsplit(" ", 1)[0].rstrip() + "…"
    result["description"] = (original + block).strip()
    return result
