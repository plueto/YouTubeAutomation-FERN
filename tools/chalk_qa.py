#!/usr/bin/env python3
"""Quality assurance for a rendered chalkboard video.

Usage:
    python tools/chalk_qa.py video.mp4 [--out qa_dir] [--strict]

The JSON report distinguishes PASS, WARNING, FAIL, and BLOCKED. A blocked check is
never reported as passed. Default mode exits nonzero on a failed check; strict mode
also exits nonzero when any required check is blocked because its inputs were absent.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moduli.ffmpeg_utils import ffmpeg_path, ffprobe_path
import cv2
import numpy as np


Issue = tuple[str, str, str]
CheckSpec = tuple[str, str, bool, str]


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def classify_checks(catalog: list[CheckSpec], issues: list[Issue]) -> list[dict[str, str]]:
    """Classify every declared check without treating unavailable inputs as a pass."""
    checks: list[dict[str, str]] = []
    for key, name, can_run, blocked_reason in catalog:
        mine = [(level, message) for level, category, message in issues if category == key]
        failures = [message for level, message in mine if level == "FAIL"]
        warnings = [message for level, message in mine if level in ("WARN", "WARNING")]
        if failures:
            status, detail = "FAIL", "; ".join(failures)
        elif not can_run:
            status, detail = "BLOCKED", blocked_reason or "required input was unavailable"
        elif warnings:
            status, detail = "WARNING", "; ".join(warnings)
        else:
            status, detail = "PASS", "executed; criterion met"
        checks.append({"check": name, "status": status, "detail": detail})
    return checks


def overall_status(checks: list[dict[str, str]]) -> str:
    statuses = {check["status"] for check in checks}
    if "FAIL" in statuses:
        return "FAIL"
    if "BLOCKED" in statuses:
        return "BLOCKED"
    if "WARNING" in statuses:
        return "WARNING"
    return "PASS" if checks and statuses == {"PASS"} else "BLOCKED"


def qa_exit_code(checks: list[dict[str, str]], strict: bool = False) -> int:
    """Fail for any FAIL; strict mode also fails when required checks are BLOCKED."""
    if any(c["status"] == "FAIL" for c in checks):
        return 1
    if strict and any(c["status"] == "BLOCKED" for c in checks):
        return 1
    return 0


def _run(cmd: list[str]) -> bytes:
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def _frame_at(video: Path, t: float, w: int, h: int) -> np.ndarray | None:
    raw = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-nostdin", "-ss", f"{max(0.0, t):.3f}", "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True).stdout
    if len(raw) < w * h * 3:
        return None
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3).copy()


def _ink_fraction(frame: np.ndarray) -> float:
    """Fraction of pixels that are clearly brighter than the slate board."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float((g > 95).mean())


def audio_stats(video: Path) -> dict[str, Any]:
    raw = subprocess.run(
        [ffmpeg_path(), "-v", "error", "-nostdin", "-i", str(video), "-vn", "-ac", "1", "-ar", "22050",
         "-f", "f32le", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32) if raw else np.zeros(0, np.float32)
    if len(x) == 0:
        return {"present": False, "seconds": 0.0}
    finite = np.isfinite(x)
    clean = x[finite]
    peak = float(np.max(np.abs(clean))) if len(clean) else None
    rms = float(np.sqrt(np.mean(clean * clean))) if len(clean) else None
    return {
        "present": True,
        "nan": int(np.isnan(x).sum()),
        "infinite": int(np.isinf(x).sum()),
        "nonfinite_samples": int((~finite).sum()),
        "peak": round(peak, 4) if peak is not None else None,
        "clipped_samples": int((np.abs(clean) >= 0.999).sum()),
        "rms": round(rms, 4) if rms is not None else None,
        "seconds": round(len(x) / 22050, 2),
    }


def _probe_video(video: Path) -> dict[str, Any]:
    """Probe with ffprobe when available; fall back to ffmpeg metadata output."""
    cmd = [ffprobe_path(), "-v", "error",
           "-show_entries", "stream=codec_type,width,height,r_frame_rate,duration:format=duration",
           "-of", "json", str(video)]
    try:
        return json.loads(_run(cmd))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        proc = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", str(video)],
            capture_output=True, text=True, timeout=45, check=False,
        )
        info = (proc.stderr or "") + "\n" + (proc.stdout or "")
        dm = re.search(r"Duration:\s*(\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", info)
        vm = re.search(r"Video:.*?(\d{3,5})x(\d{3,5})(?:[,\s]|$)", info, re.S)
        fm = re.search(r"(\d+(?:\.\d+)?)\s*fps", info)
        if not dm or not vm:
            raise RuntimeError("Could not probe video with ffprobe or ffmpeg metadata.")
        hh, mm, ss = dm.groups()
        duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
        fps = float(fm.group(1)) if fm else 24.0
        streams = [{
            "codec_type": "video", "width": int(vm.group(1)), "height": int(vm.group(2)),
            "r_frame_rate": f"{max(1, round(fps))}/1",
        }]
        if re.search(r"Audio:", info):
            streams.append({"codec_type": "audio"})
        return {"streams": streams, "format": {"duration": str(duration)}}


def _load_json(path: Path, default: dict[str, Any], missing: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("top-level JSON value is not an object")
        return value
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        missing.append(f"{path.name}: {exc}")
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thumb", type=int, default=480, help="thumbnail width in the contact sheet")
    ap.add_argument("--strict", action="store_true",
                    help="also return exit code 1 if any required QA check is BLOCKED")
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_file():
        print(f"ERROR: video does not exist: {video}", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else video.parent / "qa"
    out.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    plan = _load_json(video.with_suffix(".scene_plan.json"), {"scenes": []}, missing)
    report = _load_json(video.with_suffix(".chalk_report.json"), {}, missing)
    plan_rows = plan.get("scenes")
    if not isinstance(plan_rows, list):
        plan_rows = []
        missing.append("scene plan has no scenes array")
    probe = _probe_video(video)
    vs = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if vs is None:
        print("ERROR: video stream could not be found.", file=sys.stderr)
        return 2
    w, h = int(vs["width"]), int(vs["height"])
    vdur = float(probe["format"]["duration"])
    issues: list[Issue] = []
    thumbs, rows = [], []
    scenes_meta = report.get("scenes", [])
    if not isinstance(scenes_meta, list):
        scenes_meta = []

    frames_ok = 0
    for i, row in enumerate(plan_rows):
        try:
            t0, t1 = float(row["start_seconds"]), float(row["end_seconds"])
        except (KeyError, TypeError, ValueError):
            missing.append(f"scene {i + 1} has invalid start/end times")
            continue
        meta = scenes_meta[i] if i < len(scenes_meta) and isinstance(scenes_meta[i], dict) else {}
        busy = meta.get("pen_busy_until")
        if _finite_number(busy):
            t_end = max(t0 + 0.2, min(t1 - 0.3, float(busy) + 0.2))
        else:
            t_end = max(t0 + 0.2, min(t1 - 0.3, t1 - 0.5))
        t_mid = t0 + (t_end - t0) * 0.45
        f_end, f_mid = _frame_at(video, t_end, w, h), _frame_at(video, t_mid, w, h)
        if f_end is not None:
            frames_ok += 1
            cov: float | None = _ink_fraction(f_end)
        else:
            cov = None
        ch = row.get("chalk", {}) if isinstance(row.get("chalk"), dict) else {}
        layout = ch.get("art") or ch.get("layout")
        info = {
            "scene": i + 1, "t": [round(t0, 2), round(t1, 2)], "layout": layout,
            "ink_fraction_end": round(cov, 4) if cov is not None else None,
            "frame_end_ok": f_end is not None,
            "pen_busy_until": meta.get("pen_busy_until"), "pacing": meta.get("pacing"),
            "detail": meta.get("detail"), "word_timing": meta.get("word_timing"),
            "illustration_demo_art": row.get("illustration_demo_art") or ch.get("art"),
        }
        if cov is not None:
            if cov < 0.004:
                issues.append(("FAIL", "content", f"scene {i + 1} ({layout}): nearly empty at its end (ink {cov:.3%})"))
            elif cov < 0.012:
                issues.append(("WARNING", "content", f"scene {i + 1} ({layout}): sparse at its end (ink {cov:.3%})"))
        if _finite_number(busy) and float(busy) > t1 - 0.12:
            issues.append(("WARNING", "pen_end", f"scene {i + 1}: pen still drawing at scene end ({float(busy):.2f}s vs {t1:.2f}s)"))
        if meta.get("fits") is False:
            issues.append(("WARNING", "fits", f"scene {i + 1}: could not fit its drawing without hurrying the pen ({meta.get('pacing')})"))
        rows.append(info)
        for tag, fr in (("mid", f_mid), ("end", f_end)):
            if fr is not None:
                t_label = t_end if tag == "end" else t_mid
                label = f"{i + 1}.{tag} {layout} {t_label:.1f}s"
                tt = cv2.resize(fr, (args.thumb, int(args.thumb * h / w)), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tt, (0, 0), (tt.shape[1], 20), (0, 0, 0), -1)
                cv2.putText(tt, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 230, 220), 1, cv2.LINE_AA)
                thumbs.append(tt)
                cv2.imwrite(str(out / f"scene{i + 1:02d}_{tag}.png"), fr)

    # Contact sheet: mid/end frames paired by scene, wrapped into sheets of 4 scenes.
    per = 4
    for s in range(0, len(thumbs), per * 2):
        chunk = thumbs[s:s + per * 2]
        if len(chunk) % 2:
            chunk.append(np.zeros_like(chunk[0]))
        rowsimg = [cv2.hconcat(chunk[j:j + 2]) for j in range(0, len(chunk), 2)]
        cv2.imwrite(str(out / f"contact_{s // (per * 2) + 1}.png"), cv2.vconcat(rowsimg))

    # Frame-level blank detection across the whole video.
    blank = 0
    sampled = 0
    for t in np.arange(0.5, vdur - 0.2, max(0.5, vdur / 60)):
        fr = _frame_at(video, float(t), w, h)
        if fr is None:
            continue
        sampled += 1
        if float(fr.mean()) < 6.0:
            blank += 1
    if blank:
        issues.append(("WARNING", "blank", f"{blank}/{sampled} sampled frames are nearly black"))

    a = audio_stats(video)
    if not a.get("present"):
        issues.append(("FAIL", "audio_present", "video has no decodable audio samples"))
    else:
        if a.get("nan", 0):
            issues.append(("FAIL", "audio_nan", f"audio contains {a['nan']} NaN samples"))
        if a.get("infinite", 0):
            issues.append(("FAIL", "audio_finite", f"audio contains {a['infinite']} infinite samples"))
        if a.get("clipped_samples", 0) > 50:
            issues.append(("WARNING", "audio_clip", f"audio clips ({a['clipped_samples']} samples)"))
        if _finite_number(a.get("seconds")) and abs(float(a["seconds"]) - vdur) > 0.3:
            issues.append(("WARNING", "audio_len", f"audio length {a['seconds']}s differs from video {vdur:.2f}s"))

    sound = report.get("sound", {})
    if not isinstance(sound, dict):
        sound = {}
    if sound and not sound.get("disabled"):
        ratio = sound.get("sfx_rms_vs_voice")
        if not _finite_number(ratio):
            issues.append(("FAIL", "sfx", f"sound-effects ratio must be finite; received {ratio!r}"))
        elif float(ratio) > 0.5:
            issues.append(("WARNING", "sfx", f"sound effects are loud relative to the narration ({ratio})"))

    # Report-side summary fields are evidence too: never silently accept a
    # stale declared scene count after a plan rewrite or beat merge.
    scene_count_present = "scene_count" in plan
    scene_count_matches = (
        isinstance(plan.get("scene_count"), int)
        and not isinstance(plan.get("scene_count"), bool)
        and plan.get("scene_count") == len(plan_rows)
    )
    if scene_count_present and not scene_count_matches:
        issues.append((
            "FAIL", "scene_count",
            f"scene_plan.json declares {plan.get('scene_count')!r} scenes but contains {len(plan_rows)} rows",
        ))

    plan_ok = bool(plan_rows)
    if not plan_ok:
        content_reason = "no valid scene-plan rows"
    elif frames_ok != len(plan_rows):
        content_reason = f"only {frames_ok}/{len(plan_rows)} scene-end frames could be extracted"
    else:
        content_reason = ""
    have_meta = len(scenes_meta) == len(plan_rows) and bool(plan_rows)

    # In opt-in demo mode, check not only the plan labels but the signature
    # drawing elements recorded by the composer for each rendered scene.
    demo_cfg = plan.get("illustration_demo", {})
    demo_requested = isinstance(demo_cfg, dict) and demo_cfg.get("enabled") is True
    demo_specs = {
        "rejected_judges": {"illus:gradebook", "illus:orgchart", "illus:score_dial"},
        "two_dials": {"illus:danger_gauge", "illus:ration_valve"},
        "frosted_score": {"illus:sealed_card", "illus:frosted_pane", "illus:stamp_final"},
    }
    demo_plan_rows = [
        (i, row, str(row.get("illustration_demo_art") or
                      (row.get("chalk") or {}).get("art") or ""))
        for i, row in enumerate(plan_rows)
    ]
    if demo_requested:
        for art_name in demo_specs:
            occurrences = [item for item in demo_plan_rows if item[2] == art_name]
            if len(occurrences) != 1:
                issues.append(("FAIL", "illustration_demo",
                               f"expected exactly one {art_name} scene in the demo plan; found {len(occurrences)}"))
        demo_meta_ok = have_meta and all(
            isinstance(scenes_meta[i], dict) and isinstance(scenes_meta[i].get("elements"), list)
            for i, _row, art_name in demo_plan_rows if art_name in demo_specs
        )
        if demo_meta_ok:
            for i, _row, art_name in demo_plan_rows:
                if art_name not in demo_specs:
                    continue
                meta = scenes_meta[i]
                if meta.get("art") != art_name:
                    issues.append(("FAIL", "illustration_demo",
                                   f"scene {i + 1}: plan requested {art_name}, renderer reported {meta.get('art')!r}"))
                emitted = {str(element.get("tag")) for element in meta.get("elements", [])
                           if isinstance(element, dict) and element.get("tag")}
                absent = sorted(demo_specs[art_name] - emitted)
                if absent:
                    issues.append(("FAIL", "illustration_demo",
                                   f"scene {i + 1} ({art_name}) did not emit its signature elements: {', '.join(absent)}"))
    else:
        demo_meta_ok = False
    pen_meta_ok = have_meta and all(isinstance(m, dict) and _finite_number(m.get("pen_busy_until")) for m in scenes_meta)
    fit_meta_ok = have_meta and all(isinstance(m, dict) and isinstance(m.get("fits"), bool) for m in scenes_meta)
    audio_ok = bool(a.get("present"))
    sound_ok = bool(sound) and not sound.get("disabled")
    sound_blocked_reason = "no sound report was available or sound was disabled"
    audio_blocked_reason = "no audio stream to analyse"
    catalog: list[CheckSpec] = [
        ("scene_count", "declared scene count matches the scene list", scene_count_present, "scene plan has no scene_count summary"),
        ("content", "every scene has visible content at its end", plan_ok and frames_ok == len(plan_rows), content_reason),
        ("pen_end", "the pen finishes before each scene ends", pen_meta_ok, "per-scene pen timing metadata is incomplete"),
        ("fits", "each scene fits without hurrying the pen", fit_meta_ok, "per-scene fits metadata is incomplete"),
        ("blank", "no near-black frames", sampled > 0, "no frame could be sampled"),
        ("audio_present", "the video has an audio stream", True, ""),
        ("audio_nan", "audio has no NaN samples", audio_ok, audio_blocked_reason),
        ("audio_finite", "audio has no infinite samples", audio_ok, audio_blocked_reason),
        ("audio_clip", "audio does not clip", audio_ok, audio_blocked_reason),
        ("audio_len", "audio length equals video length", audio_ok, audio_blocked_reason),
        ("sfx", "effects level is finite and under the narration", sound_ok, sound_blocked_reason),
    ]
    if demo_requested:
        catalog.append((
            "illustration_demo",
            "all three hand-composed drawings were emitted by the renderer",
            demo_meta_ok,
            "renderer report lacks per-scene element evidence",
        ))
    checks = classify_checks(catalog, issues)
    counts = {key: sum(1 for check in checks if check["status"] == key)
              for key in ("PASS", "WARNING", "FAIL", "BLOCKED")}
    worst = overall_status(checks)
    result = {
        "video": str(video), "duration": round(vdur, 2), "resolution": [w, h],
        "fps": vs.get("r_frame_rate"), "audio": a, "scenes": rows,
        "checks": checks,
        "issues": [{"level": level, "category": category, "message": message}
                   for level, category, message in issues],
        "missing_inputs": missing,
        "counts": counts,
        "overall_status": worst,
        "strict_mode": bool(args.strict),
    }
    (out / "qa_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"QA {worst}: {video.name}  {w}x{h}  {vdur:.1f}s | {counts}")
    if missing:
        for item in missing:
            print(f"  [BLOCKED] input: {item}")
    for check in checks:
        print(f"  [{check['status']:<7}] {check['check']}: {check['detail']}")
    for row in rows:
        ink = f"{row['ink_fraction_end']:.3%}" if row["ink_fraction_end"] is not None else "N/A"
        print(f"  scene {row['scene']:>2} {row['t'][0]:6.1f}-{row['t'][1]:<6.1f} {str(row['layout']):<16} ink {ink}  detail {row['detail']}  {row['pacing'] or ''}")
    print(f"  contact sheets in {out}")
    if args.strict and counts["BLOCKED"]:
        print("  strict QA failed: blocked checks are not passes", file=sys.stderr)
    return qa_exit_code(checks, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
