"""Build a narration-bound FERN footage manifest from a local candidate export.

This command performs no search and downloads nothing. The existing source finder
can export candidate metadata + local file paths, then this command binds that
catalog to the current narration so stale clips cannot leak into another story.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moduli.credible_footage import MANIFEST_SCHEMA, script_fingerprint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="Exact narration script for this render.")
    parser.add_argument("--candidates", required=True, help="JSON list (or {'candidates': [...]}) exported by the source finder.")
    parser.add_argument("--output", default=str(ROOT / "output" / "footage_candidates.json"))
    parser.add_argument("--force", action="store_true", help="Replace an existing manifest.")
    args = parser.parse_args()

    script_path = Path(args.script).expanduser().resolve()
    candidate_path = Path(args.candidates).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()

    if not script_path.is_file():
        print(f"ERROR: narration script not found: {script_path}", file=sys.stderr)
        return 2
    if not candidate_path.is_file():
        print(f"ERROR: candidate export not found: {candidate_path}", file=sys.stderr)
        return 2
    if output_path.exists() and not args.force:
        print(f"ERROR: refusing to replace {output_path}; choose another --output or pass --force.", file=sys.stderr)
        return 2

    try:
        raw = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot read candidate export: {exc}", file=sys.stderr)
        return 2

    candidates = raw.get("candidates") if isinstance(raw, dict) else raw
    if not isinstance(candidates, list):
        print("ERROR: candidate export must be a JSON list or an object with a candidates list.", file=sys.stderr)
        return 2
    candidates = [row for row in candidates if isinstance(row, dict)]
    script = script_path.read_text(encoding="utf-8-sig")
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "script_sha256": script_fingerprint(script),
        "script_file": script_path.name,
        "candidate_export": candidate_path.name,
        "candidates": candidates,
        "policy": {
            "require_relevant_title_or_description": True,
            "require_approved_source_type": True,
            "require_reusable_license_or_documented_permission": True,
            "fallback": "chalk",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output_path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"ERROR: could not save manifest: {exc}", file=sys.stderr)
        return 2

    print("FERN FOOTAGE MANIFEST")
    print(f"Script: {script_path.name}")
    print(f"Candidate records: {len(candidates)}")
    print(f"Output: {output_path}")
    print("The renderer will still reject irrelevant, unlicensed, untraceable, missing, or unsupported candidates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
