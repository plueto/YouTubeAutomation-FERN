"""Quick local regression checks for the FERN chalkboard pacing patch."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from PIL import ImageChops

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moduli.chalkboard import (
    _board_background,
    _draw_scene,
    _group_sentences,
    _phrases,
    _scene_heading,
)


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def main() -> int:
    script = (
        "The company promised a revolutionary product, then quietly changed the pricing model. "
        "Customers saw one price on the front page, but the real costs appeared after signup. "
        "That changed how the entire business made money."
    )
    scenes = _group_sentences(script, 9)
    assert len(scenes) == 3, f"Expected 3 natural spoken beats, found {len(scenes)}"
    assert len({normalize(s) for s in scenes}) == len(scenes), "Duplicate scene narration detected"
    assert " ".join(scenes).split() == script.split(), "Scene beats lost or reordered narration words"

    for i, scene in enumerate(scenes):
        phrases = _phrases(scene, 3)
        assert len(phrases) == 3, f"Scene {i+1} does not have three labels: {phrases}"
        assert len({normalize(p) for p in phrases}) == 3, f"Scene {i+1} repeats labels: {phrases}"
        assert _scene_heading(scene, i), "Scene heading is empty"

    board = _board_background(1280, 720)
    early = _draw_scene(board, scenes[0], 0, 0.18)
    late = _draw_scene(board, scenes[0], 0, 0.92)
    assert early.size == (1280, 720) and late.size == (1280, 720), "Design canvas dimensions changed unexpectedly"
    assert ImageChops.difference(early, late).getbbox() is not None, "Drawing does not visibly progress between early and late frames"
    assert late.size == (1280, 720)

    print("PASS: contiguous narration beats, no duplicate scenes or labels, progressive drawing, centered design canvas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
