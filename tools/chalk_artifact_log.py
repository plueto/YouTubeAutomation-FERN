#!/usr/bin/env python3
"""Render docs/qa/artifact_log.json as docs/qa/artifact_log.md.

An entry may only be marked ``resolved`` if its ``verification`` names a re-render and the evidence
(frame reference or measurement); this tool refuses to render a resolved entry without one.
"""
import json
import sys
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "docs" / "qa" / "artifact_log.json"
STATUSES = {"open", "fixed-unverified", "resolved", "regressed", "wont-fix", "needs-human-review"}


def main() -> int:
    data = json.loads(LOG.read_text(encoding="utf-8"))
    problems = []
    for e in data["entries"]:
        if e["status"] not in STATUSES:
            problems.append(f"{e['id']}: unknown status {e['status']!r}")
        if e["status"] == "resolved":
            v = e.get("verification") or {}
            if not (v.get("rerender") and v.get("evidence") and v.get("result")):
                problems.append(f"{e['id']}: marked resolved without a re-render + evidence + result")
        if e["status"] == "fixed-unverified" and e.get("verification", {}).get("result"):
            problems.append(f"{e['id']}: has a verification result but is still 'fixed-unverified'")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    out = ["# Artifact log", "", data.get("note", ""), "",
           "| ID | Found | Frame / evidence | Severity | Suspected cause | Fix | Verification (after re-render) | Status |",
           "|---|---|---|---|---|---|---|---|"]
    for e in data["entries"]:
        v = e.get("verification") or {}
        ver = "-" if not v else f"{v.get('result', '')} <br>_re-render:_ {v.get('rerender', '-')} <br>_evidence:_ {v.get('evidence', '-')}"
        out.append("| {id} | {found} | {frame} | {severity} | {cause} | {fix} | {ver} | **{status}** |".format(
            ver=ver, **{k: (e.get(k) or "-") for k in ("id", "found", "frame", "severity", "cause", "fix", "status")}))
    LOG.with_suffix(".md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LOG.with_suffix('.md')} ({len(data['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
