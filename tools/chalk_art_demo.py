#!/usr/bin/env python3
"""Apply the hand-composed scene layouts (chalk_scene_art) to a baseline plan - an opt-in override.

    python tools/chalk_art_demo.py plan.json out_plan.json

It only edits the ``chalk`` object of three scenes; the director and the generic layouts are untouched.
"""
import json
import sys

OVERRIDES = {
    2: {"art": "rejected_judges", "anchors": {"judge0": "teacher", "judge1": "boss", "judge2": "credit"}, "camera": "hold"},
    5: {"art": "two_dials", "anchors": {"gauge": "dangerous", "valve": "effort", "person": "on you"}, "camera": "hold"},
    8: {"art": "frosted_score", "anchors": {"card": "score", "eye": "assigns", "pane": "never", "person": "see", "stamp": "appeal", "grade": "exists"}, "camera": "hold"},
}


def main() -> int:
    plan = json.load(open(sys.argv[1], encoding="utf-8"))
    for row in plan["scenes"]:
        ov = OVERRIDES.get(row["scene"])
        if ov:
            row["chalk"] = {**row["chalk"], **ov}
    json.dump(plan, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
    print("applied to scenes", sorted(OVERRIDES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
