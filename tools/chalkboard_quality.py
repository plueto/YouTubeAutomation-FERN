"""Technical QA, contact-sheet generation, and optional OpenRouter vision review.

This tool only reads a rendered video and writes QA artefacts beside output/.
It never uploads or modifies the video itself.
"""
from __future__ import annotations

import argparse
import base64
import json
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageStat

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "google/gemini-2.5-flash"


def load_project_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    # Prefer python-dotenv when present; fall back to a small non-expanding parser.
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(env_path, override=False)
        return
    except Exception:
        pass
    try:
        for raw in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"\'')
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def sha256_file(path: Path | None) -> str | None:
    if not path or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_ffmpeg() -> str:
    candidates = [ROOT / "ffmpeg.exe", ROOT / "ffmpeg", Path(shutil.which("ffmpeg") or "")]
    for candidate in candidates:
        if str(candidate) not in {"", "."} and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("FFmpeg not found. Expected project-root ffmpeg.exe or ffmpeg on PATH.")


def probe_video(video: Path) -> dict[str, Any]:
    """Get usable metadata without requiring ffprobe to be installed separately."""
    ffmpeg = resolve_ffmpeg()
    proc = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(video)], capture_output=True,
        text=True, errors="replace", timeout=45, check=False,
    )
    info = proc.stderr or proc.stdout or ""
    duration_match = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", info)
    duration = 0.0
    if duration_match:
        hh, mm, ss = duration_match.groups()
        duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    dims = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})(?:[,\s]|$)", info, re.S)
    width = int(dims.group(1)) if dims else 0
    height = int(dims.group(2)) if dims else 0
    video_present = bool(re.search(r"Stream #\S+.*Video:", info, re.I | re.S))
    audio_present = bool(re.search(r"Stream #\S+.*Audio:", info, re.I | re.S))
    fps_match = re.search(r"(\d+(?:\.\d+)?)\s*fps", info)
    fps = float(fps_match.group(1)) if fps_match else None
    # Some builds hide stream labels at the end of a long stderr capture; direct
    # stream-type detection is deliberately permissive rather than a false pass.
    if not video_present:
        video_present = bool(dims)
    if not audio_present:
        audio_present = bool(re.search(r"Audio:\s*(?:aac|mp3|opus|vorbis|pcm|flac|ac-3|eac3|alac)", info, re.I))
    return {
        "duration_seconds": round(duration, 3),
        "width": width,
        "height": height,
        "fps": fps,
        "has_video_stream": video_present,
        "has_audio_stream": audio_present,
        "probe_source": "ffmpeg -i metadata",
    }


def _frame_metrics(path: Path) -> dict[str, float]:
    with Image.open(path) as img:
        gray = img.convert("L")
        stats = ImageStat.Stat(gray)
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_stats = ImageStat.Stat(edges)
        # Fraction of pixels with appreciable edge intensity as a crude drawing
        # activity signal; this is not an OCR or aesthetic score.
        hist = edges.histogram()
        total = max(1, sum(hist))
        edge_pixels = sum(hist[45:]) / total
        return {
            "mean_luma": round(stats.mean[0], 2),
            "luma_stddev": round(stats.stddev[0], 2),
            "edge_activity": round(edge_pixels, 4),
        }


def extract_contact_sheet(video: Path, output_dir: Path, duration: float) -> tuple[Path, list[dict[str, Any]]]:
    ffmpeg = resolve_ffmpeg()
    frames_dir = output_dir / "sampled_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # Remove only generated thumbnails in our dedicated QA folder.
    for old in frames_dir.glob("frame_*.jpg"):
        old.unlink(missing_ok=True)
    count = max(8, min(24, int(round(duration / 2.5))))
    rate = count / max(1.0, duration)
    pattern = frames_dir / "frame_%03d.jpg"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video), "-vf", f"fps={rate:.8f},scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2:black",
        "-frames:v", str(count), "-q:v", "3", str(pattern),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=max(120, int(duration * 2)))
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError("FFmpeg did not extract any review frames.")

    thumb_w, thumb_h = 384, 216
    label_h, gap, header_h = 28, 12, 54
    cols = 4
    rows = (len(frame_paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, header_h + rows * (thumb_h + label_h + gap) + gap), "#101719")
    draw = ImageDraw.Draw(sheet)
    draw.text((gap, 13), f"FERN CHALKBOARD QA  |  {video.name}", fill="#f0eee2")
    draw.text((gap, 32), f"{duration:.1f}s | {len(frame_paths)} samples | generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", fill="#aab8b2")

    frames: list[dict[str, Any]] = []
    previous_small: Image.Image | None = None
    repeated_pairs = 0
    dark_frames = 0
    for i, frame_path in enumerate(frame_paths):
        timestamp = min(duration, i / rate)
        x = gap + (i % cols) * (thumb_w + gap)
        y = header_h + gap + (i // cols) * (thumb_h + label_h + gap)
        with Image.open(frame_path) as src:
            thumb = src.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            small = thumb.convert("L").resize((64, 36), Image.Resampling.BILINEAR)
            metric = _frame_metrics(frame_path)
            if metric["mean_luma"] < 17 and metric["luma_stddev"] < 7:
                dark_frames += 1
            difference = None
            if previous_small is not None:
                # Compute actual inter-sample difference on small grayscale thumbnails.
                from PIL import ImageChops
                diff_img = ImageChops.difference(previous_small, small)
                difference = round(ImageStat.Stat(diff_img).mean[0], 3)
                if difference < 0.25:
                    repeated_pairs += 1
            previous_small = small.copy()
            draw.text((x, y + thumb_h + 5), f"{timestamp:6.1f}s  L={metric['mean_luma']:.0f}  E={metric['edge_activity']:.2f}", fill="#d5ded7")
            frames.append({"timestamp_seconds": round(timestamp, 2), **metric, "difference_from_previous_sample": difference})
    sheet_path = output_dir / "contact_sheet.jpg"
    sheet.save(sheet_path, quality=88, optimize=True)
    # Attach summary info to first frame analysis via returned metadata at caller.
    return sheet_path, frames


def _normalise_response_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "\n".join(chunks)
    return str(content or "")


def run_ai_review(contact_sheet: Path, script_path: Path | None) -> dict[str, Any]:
    load_project_env()
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env or run without --ai-review.")
    model = os.environ.get("FERN_CHALKBOARD_REVIEW_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    script = ""
    if script_path and script_path.is_file():
        script = script_path.read_text(encoding="utf-8-sig", errors="replace")[:12000]
    encoded = base64.b64encode(contact_sheet.read_bytes()).decode("ascii")
    prompt = (
        "Review this storyboard contact sheet for a YouTube narration video rendered in a chalkboard style. "
        "Evaluate composition variety, readability at normal video size, visual clutter, repetition, contrast, "
        "whether diagrams communicate a coherent point, and alignment with the supplied narration. Do not "
        "reward it merely for looking consistent. Be strict and specific. The contact sheet is sampled across "
        "the full preview; this is not a single still. Return ONLY JSON with keys: verdict (PASS/REVISE/FAIL), "
        "score (0-100), strengths (array of strings), issues (array of strings), fixes (array of concrete strings), "
        "and notes (string). A score is an advisory judgment, not objective truth.\n\n"
        f"Narration excerpt:\n{script[:10000] if script else '(no script file supplied)'}"
    )
    payload = {
        "model": model,
        "temperature": 0.1,
        "max_tokens": 1200,
        "messages": [
            {"role": "system", "content": "You are a demanding visual QA reviewer. Follow the requested JSON schema exactly."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
            ]},
        ],
    }
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "FERN Chalkboard QA",
        }, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[-2000:]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"OpenRouter visual review failed: {exc}") from exc
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {json.dumps(body)[:1200]}")
    raw = _normalise_response_content((choices[0].get("message") or {}).get("content"))
    first, last = raw.find("{"), raw.rfind("}")
    if first < 0 or last <= first:
        raise RuntimeError("Vision reviewer did not return a JSON object. Raw response: " + raw[:1500])
    parsed = json.loads(raw[first:last + 1])
    verdict = str(parsed.get("verdict", "REVISE")).upper()
    if verdict not in {"PASS", "REVISE", "FAIL"}:
        verdict = "REVISE"
    score = parsed.get("score", 0)
    try:
        score = max(0, min(100, int(float(score))))
    except (TypeError, ValueError):
        score = 0
    return {"model": model, "verdict": verdict, "score": score, "strengths": parsed.get("strengths", []),
            "issues": parsed.get("issues", []), "fixes": parsed.get("fixes", []), "notes": str(parsed.get("notes", ""))}


def quality_check(video: Path, output_dir: Path, script_path: Path | None, source_audio_path: Path | None, do_ai: bool) -> tuple[dict[str, Any], int]:
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.stat().st_size < 30_000:
        raise RuntimeError(f"Video is unexpectedly small ({video.stat().st_size:,} bytes): {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = probe_video(video)
    warnings: list[str] = []
    failures: list[str] = []
    if metadata["duration_seconds"] <= 0:
        failures.append("Could not read a valid video duration.")
    if not metadata["has_video_stream"] or not metadata["width"] or not metadata["height"]:
        failures.append("A usable video stream or frame dimensions could not be confirmed.")
    if metadata["width"] < 960 or metadata["height"] < 540:
        warnings.append(f"Resolution is {metadata['width']}x{metadata['height']}; verify this was intentional.")
    if not metadata["has_audio_stream"]:
        failures.append("No audio stream was detected in the rendered video.")

    contact_sheet = None
    frames: list[dict[str, Any]] = []
    if metadata["duration_seconds"] > 0:
        contact_sheet, frames = extract_contact_sheet(video, output_dir, metadata["duration_seconds"])
        dark_count = sum(1 for f in frames if f["mean_luma"] < 17 and f["luma_stddev"] < 7)
        duplicate_pairs = sum(1 for f in frames if f["difference_from_previous_sample"] is not None and f["difference_from_previous_sample"] < 0.25)
        if dark_count:
            warnings.append(f"{dark_count}/{len(frames)} sampled frames look almost blank or very dark; inspect the contact sheet.")
        if len(frames) > 1 and duplicate_pairs >= max(3, len(frames) // 3):
            warnings.append(f"{duplicate_pairs} adjacent sample pairs look very similar; check for repetitive or static visuals.")
        edge_low = sum(1 for f in frames if f["edge_activity"] < 0.018)
        if len(frames) and edge_low / len(frames) >= 0.4:
            warnings.append("Many sampled frames contain little drawing/detail; possible empty layouts or overly sparse scenes.")

    ai_result = None
    ai_error = None
    if do_ai and contact_sheet:
        try:
            ai_result = run_ai_review(contact_sheet, script_path)
            if ai_result["verdict"] == "FAIL" or ai_result["score"] < 55:
                warnings.append("AI visual review recommends FAIL; do not use this as a final video without revising it.")
            elif ai_result["verdict"] == "REVISE" or ai_result["score"] < 75:
                warnings.append("AI visual review recommends revisions; manual viewing is still required.")
        except Exception as exc:
            ai_error = str(exc)
            warnings.append("AI visual review could not complete: " + ai_error)

    scene_plan_path = video.with_suffix(".scene_plan.json")
    scene_plan_exists = scene_plan_path.is_file()
    plan_checks: dict[str, Any] = {"loaded": False, "scene_count": 0, "duplicate_scene_excerpts": [],
                                   "duplicate_scene_labels": [], "duplicate_titles": [], "long_labels": [],
                                   "timing_gaps": [], "composition_runs": [], "unique_compositions": 0,
                                   "timing_source": None, "timing_confidence": None,
                                   "average_speaking_rate_wpm": None,
                                   "director": None, "drawing_action_total": 0,
                                   "scenes_without_executable_actions": [], "duplicate_action_text_scenes": [],
                                   "plan_source_counts": {}}
    if not scene_plan_exists:
        warnings.append("Renderer scene-plan JSON was not found beside the video; visual choices cannot be audited from a plan file.")
    else:
        try:
            plan = json.loads(scene_plan_path.read_text(encoding="utf-8"))
            plan_scenes = plan.get("scenes") or []
            plan_checks["loaded"] = True
            plan_checks["scene_count"] = len(plan_scenes)
            plan_checks["timing_source"] = plan.get("timing_source")
            plan_checks["timing_confidence"] = plan.get("timing_confidence")
            plan_checks["director"] = plan.get("director")
            excerpt_seen: set[str] = set()
            duplicate_excerpts: list[str] = []
            label_issues: list[str] = []
            title_issues: list[str] = []
            long_labels: list[str] = []
            comp_sequence: list[str] = []
            previous_end = 0.0
            word_total = 0
            for scene in plan_scenes:
                scene_id = str(scene.get("scene", "?"))
                excerpt = re.sub(r"\W+", " ", str(scene.get("narration_excerpt", "")).casefold()).strip()
                if excerpt and excerpt in excerpt_seen:
                    duplicate_excerpts.append(scene_id)
                excerpt_seen.add(excerpt)
                raw_labels = scene.get("labels") or scene.get("phrases") or []
                labels = [re.sub(r"\W+", " ", str(x).casefold()).strip() for x in raw_labels]
                labels = [x for x in labels if x]
                if len(labels) != len(set(labels)):
                    label_issues.append(scene_id)
                if any(len(str(x).split()) > 5 for x in raw_labels):
                    long_labels.append(scene_id)
                title = re.sub(r"\W+", " ", str(scene.get("title", "")).casefold()).strip()
                if title and title in labels:
                    title_issues.append(scene_id)
                comp_sequence.append(str(scene.get("composition") or scene.get("visual_kind") or "unknown"))
                source = str(scene.get("plan_source") or plan.get("director") or "unknown")
                plan_checks["plan_source_counts"][source] = plan_checks["plan_source_counts"].get(source, 0) + 1
                actions = scene.get("drawing_actions")
                if not isinstance(actions, list) or len(actions) < 4:
                    plan_checks["scenes_without_executable_actions"].append(scene_id)
                else:
                    plan_checks["drawing_action_total"] += len(actions)
                    action_texts = [re.sub(r"\W+", " ", str(a.get("text", "")).casefold()).strip()
                                    for a in actions if isinstance(a, dict) and a.get("type") == "text"]
                    if len(action_texts) != len(set(x for x in action_texts if x)):
                        plan_checks["duplicate_action_text_scenes"].append(scene_id)
                start = float(scene.get("start_seconds", 0.0) or 0.0)
                end = float(scene.get("end_seconds", 0.0) or 0.0)
                if start > previous_end + 0.12:
                    plan_checks["timing_gaps"].append({"after_scene": scene_id, "gap_seconds": round(start-previous_end, 3)})
                previous_end = max(previous_end, end)
                word_total += int(scene.get("word_count", len(str(scene.get("narration_excerpt", "")).split())) or 0)
            plan_checks["duplicate_scene_excerpts"] = duplicate_excerpts
            plan_checks["duplicate_scene_labels"] = label_issues
            plan_checks["duplicate_titles"] = title_issues
            plan_checks["long_labels"] = long_labels
            plan_checks["unique_compositions"] = len(set(comp_sequence))
            # Repeated layout runs are a creative warning, not a technical failure.
            run_len = 1
            for i in range(1, len(comp_sequence)):
                if comp_sequence[i] == comp_sequence[i-1]:
                    run_len += 1
                else:
                    if run_len >= 3:
                        plan_checks["composition_runs"].append({"composition": comp_sequence[i-1], "length": run_len})
                    run_len = 1
            if run_len >= 3 and comp_sequence:
                plan_checks["composition_runs"].append({"composition": comp_sequence[-1], "length": run_len})
            if metadata["duration_seconds"] > 0 and word_total:
                rate = word_total / metadata["duration_seconds"] * 60.0
                plan_checks["average_speaking_rate_wpm"] = round(rate, 1)
                if rate > 185:
                    warnings.append(f"Narration pacing is about {rate:.0f} words/min; consider shorter visual beats or a slower narration delivery.")
                elif rate < 95:
                    warnings.append(f"Narration pacing is about {rate:.0f} words/min; some visual beats may linger too long relative to speech.")
            if plan_checks["scenes_without_executable_actions"]:
                warnings.append("Some scenes lack executable coordinate-based drawing actions: " + ", ".join(plan_checks["scenes_without_executable_actions"]) + ".")
            if plan_checks["duplicate_action_text_scenes"]:
                warnings.append("Some drawing-action plans repeat text within a scene: " + ", ".join(plan_checks["duplicate_action_text_scenes"]) + ".")
            if plan_checks.get("director") != "openrouter_ai_director":
                warnings.append("Preview used local heuristic scene planning, not OpenRouter AI direction. Use --ai-plan to compare the AI-directed version.")
            if duplicate_excerpts:
                warnings.append("Scene plan repeats narration excerpts in scenes: " + ", ".join(duplicate_excerpts) + ".")
            if label_issues:
                warnings.append("Scene plan repeats labels within scenes: " + ", ".join(label_issues) + ".")
            if title_issues:
                warnings.append("Scene plan repeats a title verbatim as a label in scenes: " + ", ".join(title_issues) + ".")
            if long_labels:
                warnings.append("Some scene labels exceed five words and may be difficult to read: " + ", ".join(long_labels) + ".")
            if len(plan_scenes) >= 4 and plan_checks["unique_compositions"] < 3:
                warnings.append("Few unique scene compositions were used; the visual grammar may feel repetitive.")
            if plan_checks["composition_runs"]:
                warnings.append("One composition repeats for three or more consecutive scenes; inspect visual variety.")
            if plan_checks.get("timing_source") != "local_asr_word_timestamps":
                warnings.append("Scene timing is estimated from the script/audio duration rather than local word timestamps.")
            if plan_checks["timing_gaps"]:
                warnings.append("Scene plan has timeline gaps larger than 0.12 seconds; inspect scene-plan timing.")
        except Exception as exc:
            warnings.append(f"Renderer scene-plan JSON could not be validated: {exc}")
    technical_status = "FAIL" if failures else ("WARN" if warnings else "PASS")
    report: dict[str, Any] = {
        "video": str(video),
        "scene_plan_path": str(scene_plan_path) if scene_plan_exists else None,
        "file_size_bytes": video.stat().st_size,
        "script_path": str(script_path) if script_path else None,
        "script_sha256": sha256_file(script_path),
        "source_audio_path": str(source_audio_path) if source_audio_path else None,
        "source_audio_sha256": sha256_file(source_audio_path),
        "video_sha256": sha256_file(video),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "technical_status": technical_status,
        "failures": failures,
        "warnings": warnings,
        "metadata": metadata,
        "sampled_frame_count": len(frames),
        "sampled_frames": frames,
        "scene_plan_checks": plan_checks,
        "contact_sheet": str(contact_sheet) if contact_sheet else None,
        "ai_review": ai_result,
        "ai_review_error": ai_error,
        "manual_approval": "REQUIRED",
        "note": "Automated checks are not an aesthetic guarantee. Watch the entire preview before approving a full render or upload.",
    }
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "FERN CHALKBOARD QUALITY REPORT",
        "=" * 34,
        f"Video: {video}",
        f"Technical status: {technical_status}",
        f"Duration: {metadata['duration_seconds']:.2f} sec | Resolution: {metadata['width']}x{metadata['height']} | FPS: {metadata['fps']}",
        f"Audio stream: {'yes' if metadata['has_audio_stream'] else 'NO'} | File: {video.stat().st_size:,} bytes",
        f"Sampled frames: {len(frames)}",
        f"Contact sheet: {contact_sheet}",
        f"Scene plan: {scene_plan_path if scene_plan_exists else 'NOT FOUND'}",
        f"Scene-plan checks: {'passed' if plan_checks.get('loaded') else 'unavailable'} | Scenes: {plan_checks.get('scene_count', 0)} | Director: {plan_checks.get('director') or 'unknown'}",
        f"Executable drawing actions: {plan_checks.get('drawing_action_total', 0)} | Timing source: {plan_checks.get('timing_source') or 'unknown'} | Estimated narration rate: {plan_checks.get('average_speaking_rate_wpm') or 'n/a'} WPM",
        "",
        "FAILURES:",
        *(f"  - {x}" for x in (failures or ["None"])),
        "",
        "WARNINGS:",
        *(f"  - {x}" for x in (warnings or ["None"])),
        "",
    ]
    if ai_result:
        lines += [f"AI visual review: {ai_result['verdict']} ({ai_result['score']}/100) via {ai_result['model']}", "Strengths:"]
        lines += [f"  - {x}" for x in (ai_result.get("strengths") or ["None listed"])]
        lines += ["Issues:"]
        lines += [f"  - {x}" for x in (ai_result.get("issues") or ["None listed"])]
        lines += ["Suggested fixes:"]
        lines += [f"  - {x}" for x in (ai_result.get("fixes") or ["None listed"])]
        if ai_result.get("notes"):
            lines += ["Notes: " + ai_result["notes"]]
    elif ai_error:
        lines += ["AI visual review error: " + ai_error]
    lines += ["", "MANUAL APPROVAL: REQUIRED — open the MP4, contact sheet, and scene plan before a full render."]
    (output_dir / "quality_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report, (2 if failures else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a FERN chalkboard contact sheet and quality report.")
    parser.add_argument("video", help="Rendered MP4 to inspect")
    parser.add_argument("--script", default=str(ROOT / "director_input" / "script.txt"), help="Narration script to give the optional AI reviewer")
    parser.add_argument("--source-audio", default=None, help="Original narration audio; records its hash so the full render can verify it is unchanged")
    parser.add_argument("--output-dir", default=None, help="QA output folder (default: output/chalkboard_review/<video-stem>)")
    parser.add_argument("--ai-review", action="store_true", help="Send the contact sheet and script excerpt to OpenRouter for advisory vision review")
    parser.add_argument("--require-ai-pass", action="store_true", help="Exit nonzero unless the AI reviewer returns PASS with score >= 75")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if technical checks report a failure")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    script = Path(args.script).resolve() if args.script else None
    source_audio = Path(args.source_audio).resolve() if args.source_audio else None
    out_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "output" / "chalkboard_review" / video.stem
    try:
        report, exit_code = quality_check(video, out_dir, script, source_audio, args.ai_review)
    except Exception as exc:
        print(f"QUALITY CHECK ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Technical status: {report['technical_status']}")
    print(f"Contact sheet: {report.get('contact_sheet')}")
    print(f"Report: {out_dir / 'quality_report.txt'}")
    for item in report["failures"]:
        print("FAIL: " + item)
    for item in report["warnings"]:
        print("WARN: " + item)
    if report.get("ai_review"):
        ai = report["ai_review"]
        print(f"AI review: {ai['verdict']} ({ai['score']}/100) — {ai['model']}")
    if args.require_ai_pass:
        ai = report.get("ai_review")
        if not ai or ai.get("verdict") != "PASS" or int(ai.get("score", 0)) < 75:
            return 3
    return exit_code if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
