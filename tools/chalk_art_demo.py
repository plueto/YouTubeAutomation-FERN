#!/usr/bin/env python3
"""Apply the three hand-composed chalk scenes to semantically matching plan windows.

Usage:
    python tools/chalk_art_demo.py plan.json out_plan.json

Unlike the original index-based demo, this helper finds narration windows containing
all required anchor phrases, merges adjacent beats only when needed, and fails closed
when a scene cannot be anchored to its spoken words. It edits a plan file only; it
does not render a video or change the director's defaults.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any


SPECS: tuple[dict[str, Any], ...] = (
    {
        "art": "rejected_judges",
        "target_seconds": 8.0,
        "anchors": {
            "judge0": ("teacher",),
            "judge1": ("boss",),
            "judge2": ("credit bureau", "bureau", "credit"),
        },
    },
    {
        "art": "two_dials",
        "target_seconds": 8.0,
        "anchors": {
            "gauge": ("dangerous", "danger"),
            "valve": ("effort",),
            "person": ("you",),
        },
    },
    {
        "art": "frosted_score",
        "target_seconds": 12.0,
        "anchors": {
            "card": ("score",),
            "eye": ("assigns", "assign"),
            "pane": ("never",),
            "person": ("see", "seen"),
            "stamp": ("appeal",),
            "grade": ("exists", "exist"),
        },
    },
)

_TOKEN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(str(value).casefold())


def _find_phrase(text: str, options: tuple[str, ...]) -> str | None:
    haystack = _tokens(text)
    for option in options:
        needle = _tokens(option)
        if not needle:
            continue
        n = len(needle)
        if any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1)):
            return option
    return None


def _window_match(rows: list[dict[str, Any]], start: int, end: int,
                  spec: dict[str, Any]) -> tuple[str, dict[str, str]]:
    text = " ".join(str(r.get("narration_excerpt", "")) for r in rows[start:end + 1])
    anchors = {
        key: phrase
        for key, options in spec["anchors"].items()
        if (phrase := _find_phrase(text, options)) is not None
    }
    return text, anchors


def _candidates(rows: list[dict[str, Any]], spec: dict[str, Any], max_window_scenes: int,
                occupied: set[int]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    n = len(rows)
    for start in range(n):
        for end in range(start, min(n, start + max_window_scenes)):
            indices = set(range(start, end + 1))
            if indices & occupied:
                continue
            text, anchors = _window_match(rows, start, end, spec)
            if len(anchors) != len(spec["anchors"]):
                continue
            try:
                t0 = float(rows[start]["start_seconds"])
                t1 = float(rows[end]["end_seconds"])
            except (KeyError, TypeError, ValueError):
                continue
            duration = t1 - t0
            if duration <= 0:
                continue
            cost = abs(duration - float(spec["target_seconds"])) + 0.6 * (end - start)
            found.append({
                "start": start, "end": end, "indices": indices, "text": text,
                "anchors": anchors, "start_seconds": t0, "end_seconds": t1,
                "duration": duration, "cost": cost,
            })
    return sorted(found, key=lambda c: (c["cost"], c["start"], c["end"]))


def _best_coverage(rows: list[dict[str, Any]], spec: dict[str, Any], max_window_scenes: int) -> str:
    best: tuple[int, int, str, list[str]] | None = None
    for start in range(len(rows)):
        for end in range(start, min(len(rows), start + max_window_scenes)):
            text, anchors = _window_match(rows, start, end, spec)
            missing = [k for k in spec["anchors"] if k not in anchors]
            candidate = (len(anchors), -(end - start), f"scenes {start + 1}-{end + 1}", missing)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return "the plan contains no scenes"
    return f"{best[2]} matched {len(spec['anchors']) - len(best[3])}/{len(spec['anchors'])} anchors; missing: {', '.join(best[3]) or 'none'}"


def _choose_non_overlapping(candidate_lists: list[list[dict[str, Any]]]) -> list[dict[str, Any]] | None:
    best: list[dict[str, Any]] | None = None
    best_cost = float("inf")

    def visit(i: int, chosen: list[dict[str, Any]], occupied: set[int], cost: float) -> None:
        nonlocal best, best_cost
        if cost >= best_cost:
            return
        if i == len(candidate_lists):
            best = list(chosen)
            best_cost = cost
            return
        for candidate in candidate_lists[i]:
            if candidate["indices"] & occupied:
                continue
            visit(i + 1, chosen + [candidate], occupied | candidate["indices"], cost + candidate["cost"])

    visit(0, [], set(), 0.0)
    return best


def _merge_window(rows: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    window = rows[start:end + 1]
    merged = copy.deepcopy(window[0])
    excerpts = [str(row.get("narration_excerpt", "")).strip() for row in window]
    merged["narration_excerpt"] = " ".join(x for x in excerpts if x)
    merged["start_seconds"] = float(window[0]["start_seconds"])
    merged["end_seconds"] = float(window[-1]["end_seconds"])
    merged["word_count"] = len(merged["narration_excerpt"].split())
    merged["demo_source_scenes"] = list(range(start + 1, end + 2))

    # Preserve measured word times only when every source row has a complete,
    # token-aligned list. Otherwise the composer uses its clearly labelled estimate.
    all_times: list[Any] = []
    times_complete = True
    for row, excerpt in zip(window, excerpts):
        word_times = row.get("word_times")
        if not isinstance(word_times, list) or len(word_times) != len(excerpt.split()):
            times_complete = False
            break
        all_times.extend(word_times)
    if times_complete and all_times:
        merged["word_times"] = all_times
    else:
        merged.pop("word_times", None)
        merged.pop("word_times_source", None)

    return merged


def apply_illustration_overrides(plan: dict[str, Any], max_window_scenes: int = 3) -> list[dict[str, Any]]:
    """Mutate a scene plan so every custom drawing is anchored to a matched narration window.

    Adjacent source beats may be joined to give a scene-specific composition enough time
    to show its complete spoken idea. The function fails rather than placing an unanchored
    drawing on a vaguely related scene.
    """
    if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list) or not plan["scenes"]:
        raise ValueError("Illustration demo needs a non-empty scene plan")
    if max_window_scenes < 1:
        raise ValueError("max_window_scenes must be at least 1")

    # Import for its registration side effect and make a broken registration fail early.
    from moduli import chalk_scene_art as _chalk_scene_art  # noqa: F401
    from moduli.chalk_composer import ART
    missing_art = [spec["art"] for spec in SPECS if spec["art"] not in ART]
    if missing_art:
        raise RuntimeError(f"Hand-composed art was not registered: {', '.join(missing_art)}")

    rows: list[dict[str, Any]] = plan["scenes"]
    candidate_lists: list[list[dict[str, Any]]] = []
    for spec in SPECS:
        candidates = _candidates(rows, spec, max_window_scenes, set())
        if not candidates:
            coverage = _best_coverage(rows, spec, max_window_scenes)
            raise ValueError(
                f"Cannot place {spec['art']} on a fully anchored narration window: {coverage}. "
                f"Increase preview duration or allow a longer window (max {max_window_scenes} scenes)."
            )
        candidate_lists.append(candidates)

    selected = _choose_non_overlapping(candidate_lists)
    if selected is None:
        raise ValueError(
            "The three illustration windows overlap; increase preview duration so each "
            "composition has its own narration window."
        )

    replacements: dict[int, dict[str, Any]] = {}
    placements: list[dict[str, Any]] = []
    for spec, candidate in zip(SPECS, selected):
        start, end = candidate["start"], candidate["end"]
        row = _merge_window(rows, start, end)
        ch = dict(row.get("chalk") or {})
        ch.update({
            "art": spec["art"],
            "layout": "hero",       # fallback layout; ART selects the hand-composed builder
            "title": "",
            "labels": [],
            "motifs": [],
            "anchors": dict(candidate["anchors"]),
            "camera": "hold",
            "transition": "erase",
            "detail": 3,
        })
        row["chalk"] = ch
        row["illustration_demo_art"] = spec["art"]
        replacements[start] = row
        placements.append({
            "art": spec["art"],
            "source_scenes": list(range(start + 1, end + 2)),
            "duration_seconds": round(candidate["duration"], 2),
            "anchors": dict(candidate["anchors"]),
            "narration_excerpt": candidate["text"],
        })

    occupied: set[int] = set()
    result_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in replacements:
            replacement = replacements[index]
            result_rows.append(replacement)
            occupied.update(replacement["demo_source_scenes"][i] - 1 for i in range(len(replacement["demo_source_scenes"])))
        elif index not in occupied:
            result_rows.append(copy.deepcopy(row))

    plan["scenes"] = result_rows
    plan["illustration_demo"] = {
        "enabled": True,
        "placements": sorted(placements, key=lambda p: p["source_scenes"][0]),
    }
    return plan["illustration_demo"]["placements"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the three hand-composed chalk scenes to semantically matching narration windows."
    )
    parser.add_argument("plan", help="input scene-plan JSON")
    parser.add_argument("out_plan", help="output scene-plan JSON")
    parser.add_argument(
        "--max-window-scenes", type=int, default=3,
        help="maximum adjacent source scenes to combine to satisfy one composition's anchors (default: 3)",
    )
    args = parser.parse_args()
    source, destination = Path(args.plan), Path(args.out_plan)
    try:
        plan = json.loads(source.read_text(encoding="utf-8-sig"))
        placements = apply_illustration_overrides(plan, max_window_scenes=args.max_window_scenes)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        print(f"Illustration demo failed: {exc}", file=sys.stderr)
        return 2
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    for placement in placements:
        print(
            f"applied {placement['art']} to source scenes {placement['source_scenes']} "
            f"({placement['duration_seconds']:.1f}s); anchors: {placement['anchors']}"
        )
    print(f"saved anchored illustration demo plan to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
