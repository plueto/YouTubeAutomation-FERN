"""Select FERN's visual renderer without changing narration or audio settings.

The chalk chunk engine is the default. Set FERN_RENDERER=clips to explicitly
restore the older stock-clip montage. This module intentionally does not perform
source search, download media, or upload anything.
"""
from __future__ import annotations

import os


def get_renderer_mode() -> str:
    """Return the normalized renderer mode or fail on ambiguous configuration."""
    value = os.environ.get("FERN_RENDERER", "chalk").strip().casefold()
    if value in {"chalk", "chunk", "chunk_engine", "chalkboard"}:
        return "chalk"
    if value in {"clips", "stock", "montage", "legacy"}:
        return "clips"
    raise ValueError(
        "Invalid FERN_RENDERER value. Use 'chalk' (chunk engine) or 'clips' "
        "(legacy stock montage)."
    )
