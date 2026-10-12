"""Render a full local chalkboard MP4 only after the matching preview is checked.

This workflow never invokes main.py, thumbnail generation, YouTube OAuth, or upload.
It verifies the source script/audio hashes recorded by the preview QA report, then
requires an explicit typed approval before the longer render begins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a full local chalkboard video after preview QA and human approval.")
    parser.add_argument("--audio", default=str(ROOT / "output" / "narration.mp3"), help="Original narration audio")
    parser.add_argument("--script", default=str(ROOT / "director_input" / "script.txt"), help="Matching Director input script")
    parser.add_argument("--preview", default=str(ROOT / "output" / "chalkboard_preview.mp4"), help="Preview MP4 previously reviewed")
    parser.add_argument("--output", default=str(ROOT / "output" / "chalkboard_full_local.mp4"), help="Full local output MP4; never uploaded")
    parser.add_argument("--force", action="store_true", help="Allow replacing the full local output path")
    args = parser.parse_args()

    audio, script, preview, output = (Path(x).resolve() for x in (args.audio, args.script, args.preview, args.output))
    if not audio.is_file() or not script.is_file() or not preview.is_file():
        print("STOP: audio, script, and preview must all exist. Render and review the 45-second preview first.", file=sys.stderr)
        return 2
    if output.exists() and not args.force:
        print(f"STOP: refusing to overwrite {output}. Use --force or choose a new --output path.", file=sys.stderr)
        return 2

    report_path = ROOT / "output" / "chalkboard_review" / preview.stem / "quality_report.json"
    if not report_path.is_file():
        print(f"STOP: quality report not found: {report_path}\nRun tools\\chalkboard_quality.py on this preview first.", file=sys.stderr)
        return 2
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"STOP: couldn't parse quality report: {exc}", file=sys.stderr)
        return 2
    if report.get("technical_status") == "FAIL" or report.get("failures"):
        print("STOP: preview technical QA failed. Fix it and rerun the preview first.", file=sys.stderr)
        return 2
    if report.get("video_sha256") != sha256_file(preview):
        print("STOP: the preview MP4 changed after its quality report. Rerun QA on the current preview first.", file=sys.stderr)
        return 2
    if report.get("script_sha256") != sha256_file(script):
        print("STOP: the script changed after the preview. Rebuild and review the preview first.", file=sys.stderr)
        return 2
    if report.get("source_audio_sha256") != sha256_file(audio):
        print("STOP: narration audio changed after the preview. Rebuild and review the preview first.", file=sys.stderr)
        return 2
    ai = report.get("ai_review")
    if ai and (str(ai.get("verdict", "")).upper() == "FAIL" or int(ai.get("score", 100)) < 55):
        print("STOP: the optional AI reviewer flagged the preview as FAIL. Revise it before continuing.", file=sys.stderr)
        return 2

    print("\nPRE-FLIGHT APPROVAL CHECK")
    print(f"Preview:  {preview}")
    print(f"Script:   {script}")
    print(f"Narration:{audio}")
    print(f"Technical QA: {report.get('technical_status')}")
    if ai:
        print(f"AI vision review: {ai.get('verdict')} ({ai.get('score')}/100) — advisory only")
    print("\nThe next step renders the FULL narration locally. It does not upload anything.")
    print("Watch the preview MP4 and contact sheet before proceeding.")
    try:
        confirmation = input('Type APPROVE to start the full local render (anything else cancels): ').strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled; no render started.")
        return 1
    if confirmation != "APPROVE":
        print("Cancelled; no render started.")
        return 1

    try:
        from moduli.montaggio import _media_duration
        from moduli.chalkboard import render_chalkboard_video
        duration = float(_media_duration(str(audio)))
        output.parent.mkdir(parents=True, exist_ok=True)
        def progress(pct, message):
            print(f"[{pct:3d}%] {message}", flush=True)
        render_chalkboard_video(str(audio), script.read_text(encoding="utf-8-sig", errors="replace"), str(output), on_progress=progress)
    except Exception as exc:
        print(f"FULL RENDER FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"\nFull local render ready ({duration:.1f}s): {output}")
    from subprocess import run
    qa = [sys.executable, str(ROOT / "tools" / "chalkboard_quality.py"), str(output), "--script", str(script), "--source-audio", str(audio)]
    qa_result = run(qa, cwd=str(ROOT), check=False)
    print("No upload was attempted. Review the full MP4 and its contact sheet separately.")
    return qa_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
