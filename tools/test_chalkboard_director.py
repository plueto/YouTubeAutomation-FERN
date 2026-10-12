"""Regression tests for the scene planner and its renderer contract."""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from PIL import Image, ImageChops
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import moduli.chalkboard_director as director_module
from moduli.chalkboard_director import build_scene_plan, available_alignment_backends, COMPOSITIONS, MOTIFS, _sanitize_ai_actions
from moduli.chalkboard import _board_background, _draw_scene, _draw_motif


def norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.casefold()))


def test_plan() -> dict:
    script = (
        "A company made a bold promise, then quietly changed the incentives. "
        "Customers saw a price on the front page, but the real costs appeared after signup. "
        "The hidden score changed how the entire business made money. "
        "This was the moment the promise became a trap."
    )
    plan = build_scene_plan(script, 12.5)
    scenes = plan["scenes"]
    assert len(scenes) >= 2
    assert " ".join(s["narration_excerpt"] for s in scenes).casefold() == " ".join(script.split()).casefold()
    assert len({norm(s["narration_excerpt"]) for s in scenes}) == len(scenes)
    assert abs(scenes[0]["start_seconds"]) <= .01
    assert abs(scenes[-1]["end_seconds"]-12.5) <= .02
    for i, scene in enumerate(scenes):
        assert scene["composition"] in COMPOSITIONS
        assert scene["motif"] in MOTIFS
        assert scene["title"]
        assert 1 <= len(scene["labels"]) <= 3
        assert len({norm(x) for x in scene["labels"]}) == len(scene["labels"])
        assert scene["timing_source"] == "syllable_weighted_duration_estimate"
        assert 0 <= scene["start_seconds"] < scene["end_seconds"] <= 12.5
        actions = scene.get("drawing_actions")
        assert isinstance(actions, list) and len(actions) >= 4
        assert all(a.get("type") in {"line", "arrow", "box", "circle", "motif", "text", "underline", "scribble"} for a in actions)
        action_texts = [norm(a.get("text", "")) for a in actions if a.get("type") == "text"]
        assert len(action_texts) == len(set(action_texts)), f"duplicate drawing text in scene {i+1}"
        assert {norm(x) for x in scene["labels"]}.issubset(set(action_texts)), f"scene labels not drawn in scene {i+1}"
        assert max(float(a["end"]) for a in actions) >= .78
        if i:
            assert abs(scene["start_seconds"]-scenes[i-1]["end_seconds"]) <= .01
    return plan


def test_renderer_all_compositions(plan: dict) -> None:
    from moduli.chalkboard import _draw_scene
    base = _board_background(1280, 720)
    text = "The company hid the real cost while the score changed incentives."
    labels = ["Hidden cost", "Changed score", "Business incentives"]
    for i, kind in enumerate(sorted(COMPOSITIONS)):
        row = {
            "composition": kind, "visual_kind": kind, "title": f"QA {kind.replace('_',' ')}",
            "labels": labels, "phrases": labels, "motif": "coin" if kind == "numbers" else "eye",
            "narration_excerpt": text, "start_seconds": 0.0, "end_seconds": 4.0,
        }
        frame_a = _draw_scene(base, text, i, 0.35, row)
        frame_b = _draw_scene(base, text, i, 0.95, row)
        assert frame_a.size == (1280, 720) and frame_b.size == (1280, 720)
        assert ImageChops.difference(frame_a, frame_b).getbbox() is not None, f"{kind} did not animate"


def test_freeform_action_renderer(plan: dict) -> None:
    base = _board_background(1280, 720)
    for i, row in enumerate(plan["scenes"]):
        early = _draw_scene(base, row["narration_excerpt"], i, 0.12, row)
        late = _draw_scene(base, row["narration_excerpt"], i, 0.94, row)
        assert early.size == late.size == (1280, 720)
        assert ImageChops.difference(early, late).getbbox() is not None, f"freeform scene {i+1} did not animate"


def test_coordinate_action_schema() -> None:
    labels = ["Hidden score", "Business incentives"]
    title = "The hidden lever"
    actions = [
        {"type": "motif", "motif": "eye", "x": 0.50, "y": 0.43, "size": 0.10,
         "start": 0.10, "end": 0.34, "color": "yellow"},
        {"type": "line", "x1": 0.38, "y1": 0.52, "x2": 0.62, "y2": 0.52,
         "start": 0.18, "end": 0.42, "color": "teal"},
        {"type": "text", "text": labels[0], "x": 0.31, "y": 0.65, "size": "medium",
         "align": "center", "max_width": 0.24, "start": 0.28, "end": 0.58, "color": "chalk"},
        {"type": "text", "text": labels[1], "x": 0.70, "y": 0.65, "size": "medium",
         "align": "center", "max_width": 0.30, "start": 0.48, "end": 0.80, "color": "yellow"},
        {"type": "underline", "x1": 0.32, "y1": 0.77, "x2": 0.68, "y2": 0.77,
         "start": 0.72, "end": 0.92, "color": "yellow"},
    ]
    clean = _sanitize_ai_actions(actions, title, labels)
    assert clean and len(clean) == len(actions)
    duplicate = [dict(a) for a in actions]
    duplicate[3]["text"] = labels[0]
    assert _sanitize_ai_actions(duplicate, title, labels) is None


def test_compact_ai_direction_uses_local_chalk_geometry() -> None:
    """AI chooses story beats, while local code keeps coordinate output compact and reliable."""
    script = (
        "A company made a bold promise, then quietly changed the incentives. "
        "Customers saw a price on the front page, but the real costs appeared after signup. "
        "The hidden score changed how the entire business made money."
    )
    rows = build_scene_plan(script, 12.5)["scenes"]
    ai_scenes = []
    for row in rows:
        labels = row["labels"][:3]
        if len(labels) < 2:
            labels = labels + ["Key consequence"]
        ai_scenes.append({
            "scene": row["scene"],
            "purpose": "Reveal the mechanism",
            "composition": row["composition"],
            "motif": row["motif"],
            "title": f"Scene focus {row['scene']}",
            "labels": labels,
            "visual_metaphor": "A hidden mechanism changes the visible outcome",
        })
    content = json.dumps({"scenes": ai_scenes}).encode("utf-8")

    class FakeResponse:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            return False
        def read(self):
            return json.dumps({
                "choices": [{
                    "message": {"content": content.decode("utf-8")},
                    "finish_reason": "stop",
                }]
            }).encode("utf-8")

    old_key = os.environ.get("OPENROUTER_API_KEY")
    old_urlopen = director_module.urllib.request.urlopen
    captured = {}
    def fake_urlopen(request, timeout=50):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    try:
        os.environ["OPENROUTER_API_KEY"] = "test-only-key"
        director_module.urllib.request.urlopen = fake_urlopen
        result, error = director_module._ai_direct(rows, script)
    finally:
        director_module.urllib.request.urlopen = old_urlopen
        if old_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old_key

    assert error is None, error
    assert result and len(result) == len(rows)
    prompt = captured["payload"]["messages"][0]["content"]
    assert "Do NOT output drawing_actions" in prompt
    assert "x1,y1,x2,y2" not in prompt
    assert captured["payload"]["max_tokens"] <= 2500
    for row in result:
        actions = row["drawing_actions"]
        assert len(actions) >= 4
        texts = [norm(a.get("text", "")) for a in actions if a["type"] == "text"]
        assert {norm(x) for x in row["labels"]}.issubset(set(texts))
        assert max(float(a["end"]) for a in actions) >= .78


def test_ai_failure_explicitly_falls_back_to_chalkboard() -> None:
    original = director_module._ai_direct
    director_module._ai_direct = lambda rows, script: (None, "planned test failure")
    try:
        plan = build_scene_plan("A small test script with enough words.", 3.0, use_ai=True)
    finally:
        director_module._ai_direct = original

    assert plan["director"] == "local_heuristic_director"
    assert plan["ai_director_note"] == "planned test failure"
    assert plan["scenes"] and all(scene.get("drawing_actions") for scene in plan["scenes"])


def test_all_motifs() -> None:
    from PIL import ImageDraw
    from moduli.chalkboard import _board_background
    from moduli.chalkboard_director import MOTIFS
    for i, motif in enumerate(sorted(MOTIFS)):
        board = _board_background(640, 360)
        draw = ImageDraw.Draw(board)
        _draw_motif(draw, motif, 320, 180, 110, 0.88, seed=i)


def main() -> int:
    plan = test_plan()
    test_renderer_all_compositions(plan)
    test_freeform_action_renderer(plan)
    test_coordinate_action_schema()
    test_ai_failure_explicitly_falls_back_to_chalkboard()
    test_compact_ai_direction_uses_local_chalk_geometry()
    test_all_motifs()
    inventory = available_alignment_backends()
    assert isinstance(inventory.get("packages"), dict)
    print(f"PASS: {plan['scene_count']} contiguous beats; complete script, unique text, validated coordinate actions, and all {len(COMPOSITIONS)} legacy compositions.")
    print("PASS: freeform scene actions draw progressively across the beat and render to 1280x720.")
    print("PASS: compact AI direction plus locally generated timed chalk geometry; transparent local chalkboard fallback.")
    print(f"PASS: all {len(MOTIFS)} chalk motifs render without errors.")
    print("PASS: timing provenance explicit; no alignment packages/models installed or downloaded by test.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
