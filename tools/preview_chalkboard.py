"""Render an existing FERN narration to a short local chalkboard preview.

Never uploads. By default it refuses to overwrite an existing output MP4.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="FERN Chalkboard Pass 4: AI-directed action plans + local preview; no upload.")
    parser.add_argument("--version", action="version", version="FERN Chalkboard Preview Pass 4 (coordinate drawing actions)")
    parser.add_argument("--duration", type=float, default=45.0, help="Preview length in seconds (default: 45).")
    parser.add_argument("--audio", default=str(ROOT / "output" / "narration.mp3"), help="Narration audio path.")
    parser.add_argument("--script", default=str(ROOT / "director_input" / "script.txt"), help="Matching narration script path.")
    parser.add_argument("--output", default=str(ROOT / "output" / "chalkboard_preview.mp4"), help="Output video path.")
    parser.add_argument("--force", action="store_true", help="Allow replacing the named preview MP4.")
    parser.add_argument("--ai-review", action="store_true", help="Send preview frames to OpenRouter for advisory vision review (separate from scene planning).")
    parser.add_argument("--ai-plan", action="store_true", help="Ask OpenRouter to direct the short scene plan. Sends this excerpt to the configured model; otherwise planning remains local.")
    parser.add_argument("--align-audio", action="store_true", help="Use a locally installed ASR model with cached weights for word timestamps. Does not install or download anything.")
    parser.add_argument("--aligner-model", default="base", choices=("tiny", "base", "small", "medium"), help="Name of an already-cached local alignment model (default: base).")
    parser.add_argument("--strict-quality", action="store_true", help="Return a failure code if technical quality checks fail.")
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")

    audio = Path(args.audio).resolve()
    script_path = Path(args.script).resolve()
    output = Path(args.output).resolve()
    if not audio.is_file():
        print(f"ERROR: narration not found: {audio}\nRun a FERN build first so output/narration.mp3 exists.", file=sys.stderr)
        return 2
    if not script_path.is_file():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2
    if output.exists() and not args.force:
        print(f"ERROR: refusing to overwrite existing preview: {output}\nUse --force to replace this preview, or choose --output with a new filename.", file=sys.stderr)
        return 2

    try:
        from moduli.montaggio import _ffmpeg, _media_duration
        from moduli.chalkboard import render_chalkboard_video
        full_duration = float(_media_duration(str(audio)))
    except Exception as exc:
        print(f"ERROR: FERN video dependencies could not be loaded: {exc}", file=sys.stderr)
        return 2

    duration = min(args.duration, full_duration)
    script = script_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    words = script.split()
    if full_duration > duration and words:
        take = max(1, min(len(words), math.ceil(len(words) * duration / full_duration)))
        script = " ".join(words[:take])

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fern_chalk_preview_", dir=str(output.parent)) as tmp:
        audio_preview = str(Path(tmp) / "narration_preview.m4a")
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(audio), "-t", f"{duration:.3f}", "-vn", "-c:a", "aac",
            "-b:a", "192k", audio_preview,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=180)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or str(exc))[-2000:]
            print(f"ERROR: couldn't trim narration for preview:\n{detail}", file=sys.stderr)
            return 2
        def report(pct, eta):
            print(f"[{pct:3d}%] {eta}", flush=True)
        try:
            render_chalkboard_video(
                audio_preview, script, str(output), on_progress=report,
                use_ai_director=args.ai_plan,
                align_audio=args.align_audio,
                aligner_model=args.aligner_model,
            )
        except Exception as exc:
            print(f"ERROR: chalkboard render failed: {exc}", file=sys.stderr)
            return 1

    print(f"Preview ready: {output}")
    scene_plan_path = output.with_suffix(".scene_plan.json")
    try:
        scene_plan_data = json.loads(scene_plan_path.read_text(encoding="utf-8"))
        planner_used = str(scene_plan_data.get("director", "unknown"))
        planner_note = scene_plan_data.get("ai_director_note")
    except (OSError, json.JSONDecodeError, AttributeError):
        planner_used, planner_note = "unknown", None
    print(f"Planner requested: {'OpenRouter AI' if args.ai_plan else 'local rules'}")
    print(f"Planner used: {planner_used}")
    if planner_note:
        print(f"Planner note: {planner_note}")
    qa_cmd = [sys.executable, str(ROOT / "tools" / "chalkboard_quality.py"), str(output), "--script", str(script_path), "--source-audio", str(audio)]
    if args.ai_review:
        qa_cmd.append("--ai-review")
    if args.strict_quality:
        qa_cmd.append("--strict")
    print("\n=== CHALKBOARD QUALITY CHECK ===", flush=True)
    qa_result = subprocess.run(qa_cmd, cwd=str(ROOT), check=False)
    if qa_result.returncode:
        print(f"Quality checker returned code {qa_result.returncode}. Inspect its report before continuing.", file=sys.stderr)
    print("\nNothing was uploaded. Watch the full preview before rendering a longer version.")
    return qa_result.returncode if args.strict_quality else 0


if __name__ == "__main__":
    raise SystemExit(main())
