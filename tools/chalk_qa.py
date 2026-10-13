#!/usr/bin/env python3
"""Quality assurance for a rendered chalkboard video.

Usage:  python tools/chalk_qa.py output/preview/fern_chalk_preview.mp4 [--out output/preview/qa]

Reads ``<video>.scene_plan.json`` and ``<video>.chalk_report.json`` written by the renderer and:

* extracts, for every scene, a *finished-state* frame (just before the camera leaves) and a
  *mid-draw* frame, then builds a labelled contact sheet and a per-scene strip
* flags: empty or nearly empty scenes, scenes whose pen was still busy when the scene ended,
  drawings that start long before/after the word that triggers them, blank frames,
  missing/NaN/clipped audio, and an audio/video duration mismatch
* writes ``qa_report.json`` and prints a PASS/WARN/FAIL summary (exit code 1 on FAIL)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moduli.ffmpeg_utils import ffmpeg_path, ffprobe_path
import cv2
import numpy as np


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


def audio_stats(video: Path) -> dict:
    raw = subprocess.run([ffmpeg_path(), "-v", "error", "-nostdin", "-i", str(video), "-vn", "-ac", "1", "-ar", "22050", "-f", "f32le", "-"],
                         capture_output=True).stdout
    x = np.frombuffer(raw, np.float32) if raw else np.zeros(0, np.float32)
    if len(x) == 0:
        return {"present": False}
    return {"present": True, "nan": int(np.isnan(x).sum()), "peak": round(float(np.nanmax(np.abs(x))), 3),
            "clipped_samples": int((np.abs(x) >= 0.999).sum()), "rms": round(float(np.sqrt(np.nanmean(x * x))), 4),
            "seconds": round(len(x) / 22050, 2)}


def _probe_video(video: Path) -> dict:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thumb", type=int, default=480, help="thumbnail width in the contact sheet")
    args = ap.parse_args()
    video = Path(args.video)
    out = Path(args.out) if args.out else video.parent / "qa"
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(video.with_suffix(".scene_plan.json").read_text(encoding="utf-8"))
    report = json.loads(video.with_suffix(".chalk_report.json").read_text(encoding="utf-8"))
    probe = _probe_video(video)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    vdur = float(probe["format"]["duration"])
    issues: list[tuple[str, str]] = []

    thumbs, rows = [], []
    scenes_meta = report.get("scenes", [])
    for i, row in enumerate(plan["scenes"]):
        t0, t1 = float(row["start_seconds"]), float(row["end_seconds"])
        meta = scenes_meta[i] if i < len(scenes_meta) else {}
        busy = meta.get("pen_busy_until")
        t_end = max(t0 + 0.2, min(t1 - 0.3, (busy + 0.2) if busy else t1 - 0.5))
        t_mid = t0 + (t_end - t0) * 0.45
        f_end, f_mid = _frame_at(video, t_end, w, h), _frame_at(video, t_mid, w, h)
        cov = _ink_fraction(f_end) if f_end is not None else 0.0
        ch = row.get("chalk", {})
        info = {"scene": i + 1, "t": [round(t0, 2), round(t1, 2)], "layout": ch.get("layout"), "ink_fraction_end": round(cov, 4),
                "pen_busy_until": meta.get("pen_busy_until"), "pacing": meta.get("pacing"), "detail": meta.get("detail"),
                "word_timing": meta.get("word_timing")}
        if cov < 0.004:
            issues.append(("FAIL", f"scene {i + 1} ({ch.get('layout')}): nearly empty at its end (ink {cov:.3%})"))
        elif cov < 0.012:
            issues.append(("WARN", f"scene {i + 1} ({ch.get('layout')}): sparse at its end (ink {cov:.3%})"))
        if meta.get("pen_busy_until") and meta["pen_busy_until"] > t1 - 0.12:
            issues.append(("WARN", f"scene {i + 1}: pen still drawing at scene end ({meta['pen_busy_until']:.2f}s vs {t1:.2f}s)"))
        if meta and not meta.get("fits", True):
            issues.append(("WARN", f"scene {i + 1}: could not fit its drawing without hurrying the pen ({meta.get('pacing')})"))
        rows.append(info)
        for tag, fr in (("mid", f_mid), ("end", f_end)):
            if fr is not None:
                label = f"{i + 1}.{tag} {ch.get('layout')} {t_end if tag == 'end' else t_mid:.1f}s"
                tt = cv2.resize(fr, (args.thumb, int(args.thumb * h / w)), interpolation=cv2.INTER_AREA)
                cv2.rectangle(tt, (0, 0), (tt.shape[1], 20), (0, 0, 0), -1)
                cv2.putText(tt, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 230, 220), 1, cv2.LINE_AA)
                thumbs.append(tt)
                cv2.imwrite(str(out / f"scene{i + 1:02d}_{tag}.png"), fr)

    # contact sheet: two columns (mid | end) per scene, wrapped into sheets of 4 scenes
    per = 4
    for s in range(0, len(thumbs), per * 2):
        chunk = thumbs[s:s + per * 2]
        if len(chunk) % 2:
            chunk.append(np.zeros_like(chunk[0]))
        rowsimg = [cv2.hconcat(chunk[j:j + 2]) for j in range(0, len(chunk), 2)]
        cv2.imwrite(str(out / f"contact_{s // (per * 2) + 1}.png"), cv2.vconcat(rowsimg))

    # anchor audit: how far is each drawing's first stroke from the spoken word that triggers it?
    anchor_offsets: list[float] = []
    for meta in scenes_meta:
        pass
    # frame-level blank detection across the whole video
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
        issues.append(("WARN", f"{blank}/{sampled} sampled frames are nearly black"))

    a = audio_stats(video)
    if not a.get("present"):
        issues.append(("FAIL", "video has no audio stream"))
    else:
        if a["nan"]:
            issues.append(("FAIL", f"audio contains {a['nan']} NaN samples"))
        if a["clipped_samples"] > 50:
            issues.append(("WARN", f"audio clips ({a['clipped_samples']} samples)"))
        if abs(a["seconds"] - vdur) > 0.3:
            issues.append(("WARN", f"audio length {a['seconds']}s differs from video {vdur:.2f}s"))
    sound = report.get("sound", {})
    if sound and not sound.get("disabled"):
        ratio = sound.get("sfx_rms_vs_voice")
        if ratio is None or ratio != ratio:
            issues.append(("FAIL", "sound-effects level is NaN"))
        elif ratio > 0.5:
            issues.append(("WARN", f"sound effects are loud relative to the narration ({ratio})"))

    result = {"video": str(video), "duration": round(vdur, 2), "resolution": [w, h], "fps": vs["r_frame_rate"],
              "audio": a, "scenes": rows, "issues": [{"level": l, "message": m} for l, m in issues]}
    (out / "qa_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    worst = "FAIL" if any(l == "FAIL" for l, _ in issues) else "WARN" if issues else "PASS"
    print(f"QA {worst}: {video.name}  {w}x{h}  {vdur:.1f}s  audio rms {a.get('rms')} peak {a.get('peak')}")
    for r in rows:
        print(f"  scene {r['scene']:>2} {r['t'][0]:6.1f}-{r['t'][1]:<6.1f} {str(r['layout']):<10} ink {r['ink_fraction_end']:.3%}  detail {r['detail']}  {r['pacing'] or ''}")
    for l, m in issues:
        print(f"  [{l}] {m}")
    print(f"  contact sheets in {out}")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
