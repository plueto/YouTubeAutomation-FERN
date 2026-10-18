#!/usr/bin/env python3
"""Read-only artifact and pacing scanner for a rendered chalkboard video.

    python tools/chalk_artifact_scan.py output/preview/fern_chalk_preview.mp4 --out docs/qa/scan_baseline

It never modifies the video or the engine.  It re-composes the drawing program from the saved
``.scene_plan.json`` (deterministic, same seed as the renderer) and decodes the video itself, then reports
*measurements* (``measurements.json``) and *candidates* for human review (``candidates.json``).  Every check is
classified honestly:

    PASS     executed and met its criterion
    WARNING  executed; possible issue that needs a human decision
    FAIL     executed; verified defect
    BLOCKED  could not run (missing input/dependency) - never counted as a pass
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moduli.chalk_composer import compose_program, find_phrase_time, scene_words  # noqa: E402
from moduli.chalk_engine import PAGE_H, PAGE_W, ChalkEngine  # noqa: E402


def decode_frames(video: Path, w: int, h: int, fps: int):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = w * h * 3
    i = 0
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        i += 1
    p.wait()


def groups_by_scene(prog, plan):
    """{scene_index: {tag: [x0,y0,x1,y1,t_start,t_end,n]}}; strokes are attributed using the
    composer's own per-scene stroke counts (the composer appends scene by scene), not by time."""
    out: dict[int, dict[str, list]] = defaultdict(dict)
    cursor = 0
    for si, meta in enumerate(prog.meta["scenes"]):
        for st in prog.strokes[cursor:cursor + int(meta["strokes"])]:
            x0, y0 = st.pts.min(axis=0)
            x1, y1 = st.pts.max(axis=0)
            g = out[si].get(st.tag)
            if g is None:
                out[si][st.tag] = [float(x0), float(y0), float(x1), float(y1), st.t0, st.t1, 1]
            else:
                g[0], g[1] = min(g[0], float(x0)), min(g[1], float(y0))
                g[2], g[3] = max(g[2], float(x1)), max(g[3], float(y1))
                g[4], g[5], g[6] = min(g[4], st.t0), max(g[5], st.t1), g[6] + 1
        cursor += int(meta["strokes"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    video = Path(args.video)
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    plan_path = video.with_suffix(".scene_plan.json")
    checks: list[dict] = []
    cands: list[dict] = []

    def check(name, status, detail, evidence=None):
        checks.append({"check": name, "status": status, "detail": detail, **({"evidence": evidence} if evidence else {})})

    if not plan_path.is_file():
        check("scene plan present", "BLOCKED", f"{plan_path.name} not found; program-based checks cannot run")
        (out / "measurements.json").write_text(json.dumps({"checks": checks}, indent=2))
        return 2
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    prog = compose_program(plan, seed=args.seed)
    # The program is recomposed from the plan with the CURRENT code. It is only a valid description of this video
    # if it matches what the renderer recorded for it - otherwise every program-based number would be about a
    # different video (this exact mistake happened once: new code scanned against an old render).
    rep_path = video.with_suffix(".chalk_report.json")
    if rep_path.is_file():
        recorded = [int(m["strokes"]) for m in json.loads(rep_path.read_text(encoding="utf-8")).get("scenes", [])]
        recomposed = [int(m["strokes"]) for m in prog.meta["scenes"]]
        if recorded == recomposed:
            check("recomposed program matches the rendered video", "PASS", f"per-scene stroke counts equal ({sum(recomposed)} strokes)")
        else:
            check("recomposed program matches the rendered video", "FAIL",
                  f"recorded {recorded} vs recomposed {recomposed}: run this scanner from the checkout that rendered the video; program-based results are INVALID")
    else:
        check("recomposed program matches the rendered video", "BLOCKED", f"{rep_path.name} not found; cannot verify the code version")
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                                       "-of", "json", str(video)], capture_output=True, text=True).stdout)["streams"][0]
    w, h = int(probe["width"]), int(probe["height"])
    num, den = probe["r_frame_rate"].split("/")
    fps = round(int(num) / int(den))
    eng = ChalkEngine(prog, w, h, fps, seed=args.seed, dust=False)
    scale0 = w / PAGE_W

    # ---------------------------------------------------------------- video decode pass
    diffs, edge_ink, t_list, full_diffs, med_luma, dark_frac = [], [], [], [], [], []
    prev = None
    keep: dict[int, np.ndarray] = {}
    band = max(10, int(h * 0.02))
    for i, fr in decode_frames(video, w, h, fps):
        t = i / fps
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(g, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        full_diffs.append(float(np.abs(small - prev).mean()) if prev is not None else 0.0)
        med_luma.append(float(np.median(small)))
        dark_frac.append(float((small < 70).mean()))
        diffs.append(0.0 if prev is None else max(abs(med_luma[-1] - med_luma[-2]), 40.0 * abs(dark_frac[-1] - dark_frac[-2])))
        prev = small
        ink = g > 105
        edge = int(ink[:band].sum() + ink[-band:].sum() + ink[:, :band].sum() + ink[:, -band:].sum())
        edge_ink.append(edge)
        t_list.append(t)
        keep[i] = fr if i % 6 == 0 else None  # sparse cache for evidence frames
        if keep[i] is None:
            del keep[i]
    n_frames = len(t_list)
    check("video decodes completely", "PASS" if n_frames >= int(probe.get("nb_frames", n_frames)) - 1 else "FAIL",
          f"{n_frames} frames decoded ({w}x{h} @ {fps} fps)")

    # camera speed (world px/s) to exclude deliberate pans from the edge/pop checks
    cam_speed = []
    for i, t in enumerate(t_list):
        a, b = eng.camera_at(max(0, t - 1 / fps)), eng.camera_at(t)
        cam_speed.append(float(np.hypot(b[0] - a[0], b[1] - a[1]) * fps))
    moving = np.array(cam_speed) > 250

    # ------------------------------------------------------------- edge contact (content near/at the border)
    flagged = [i for i, e in enumerate(edge_ink) if e > 60 and not moving[i]]
    runs = []
    for i in flagged:
        if runs and i - runs[-1][-1] <= 3:
            runs[-1].append(i)
        else:
            runs.append([i])
    for r in runs:
        t0, t1 = r[0] / fps, r[-1] / fps
        peak = max(edge_ink[i] for i in r)
        cands.append({"type": "content_at_frame_edge", "t": round(t0, 2), "t_end": round(t1, 2), "severity": "medium" if peak > 400 else "low",
                      "metric": f"{peak} bright pixels in the outer {band}px border", "frame": r[len(r) // 2]})
    check("content stays clear of the frame border (static camera)", "WARNING" if runs else "PASS",
          f"{len(runs)} stretches have ink within {band}px of the border" if runs else "no ink in the border band outside pans")

    # ------------------------------------------------------------- sudden pops (single-frame jumps)
    d = np.array(diffs)
    med = float(np.median(d[d > 0])) if (d > 0).any() else 0.0
    # a pop moves the whole frame's luminance distribution (median luma, share of dark board pixels)
    pops = [i for i in range(2, n_frames - 1) if d[i] > 3.0 and not moving[i] and not moving[i - 1]]
    pop_runs = []
    for i in pops:
        if pop_runs and i - pop_runs[-1][-1] <= 2:
            pop_runs[-1].append(i)
        else:
            pop_runs.append([i])
    erase_windows = [(e.t0, e.t0 + e.dur) for e in prog.erases if e.visible]
    for r in pop_runs:
        t = r[0] / fps
        if any(a - 0.1 <= t <= b + 0.3 for a, b in erase_windows):
            continue  # an eraser sweep legitimately changes many pixels
        cands.append({"type": "frame_pop", "t": round(t, 2), "severity": "medium", "metric": f"global luminance step {d[r[0]]:.1f} levels (median step {med:.2f})", "frame": r[0]})
    pops_real = [c for c in cands if c["type"] == "frame_pop"]
    check("no unexplained single-frame jumps", "WARNING" if pops_real else "PASS",
          f"{len(pops_real)} unexplained jumps" if pops_real else f"largest global luminance step {d.max():.2f} levels (threshold 3.0); pan windows excluded, erase windows excluded")

    # ------------------------------------------------------------- program-geometry checks per scene
    groups = groups_by_scene(prog, plan)
    rows = plan["scenes"]
    metrics = {"scenes": []}
    for si, row in enumerate(rows):
        t0, t1 = row["start_seconds"], row["end_seconds"]
        g = groups.get(si, {})
        text = {k: v for k, v in g.items() if k.startswith("text:")}
        illus = {k: v for k, v in g.items() if k.startswith("illus:")}
        last_end = max([v[5] for v in g.values()] or [t0])
        cam_t = t1 - 0.2
        dwell = cam_t - last_end                       # time the finished scene is on screen before the camera/eraser moves
        # legibility: single-line text groups, size on screen at the output resolution
        zoom_end = eng.camera_at(max(t0, last_end))[2]
        for tag, v in text.items():
            bw, bh = v[2] - v[0], v[3] - v[1]
            if bw / max(bh, 1) > 2.8:                   # a single line
                cap_px = bh * scale0 * zoom_end
                if cap_px < 0.032 * h:
                    cands.append({"type": "small_text", "scene": si + 1, "t": round(v[4], 2), "severity": "low",
                                  "metric": f"{tag} is {cap_px:.0f}px tall on a {h}px frame ({cap_px / h:.1%})", "frame": int(v[5] * fps)})
        # reading time of each caption once it is complete
        for tag, v in text.items():
            words = max(1, len(tag[5:].split()))
            need = 0.5 + 0.25 * words
            visible = cam_t - v[5]
            if visible < need - 1e-6:
                cands.append({"type": "caption_too_brief", "scene": si + 1, "t": round(v[5], 2), "severity": "medium" if visible < need * 0.6 else "low",
                              "metric": f"{tag} is complete for {visible:.2f}s, needs ~{need:.2f}s ({words} word(s))", "frame": int(v[5] * fps)})
        # text over drawings
        for tt, tv in text.items():
            ta = (tv[2] - tv[0]) * (tv[3] - tv[1])
            for ii, iv in illus.items():
                ix = max(0.0, min(tv[2], iv[2]) - max(tv[0], iv[0]))
                iy = max(0.0, min(tv[3], iv[3]) - max(tv[1], iv[1]))
                if ta > 0 and ix * iy / ta > 0.12:
                    cands.append({"type": "text_over_drawing", "scene": si + 1, "t": round(max(tv[4], iv[4]), 2), "severity": "medium",
                                  "metric": f"{tt} overlaps {ii} ({ix * iy / ta:.0%} of the caption)", "frame": int(tv[5] * fps)})
        # content too close to the frame edge *as the viewer sees it* (camera zoom and pan included)
        for tag, v in g.items():
            if tag.split(":")[0] not in ("text", "illus"):
                continue
            t_view = min(t1 - 0.25, v[5] + 0.3)
            vx0, vy0, vs = eng._view(max(t0, t_view))
            sx0, sy0, sx1, sy1 = (v[0] - vx0) * vs, (v[1] - vy0) * vs, (v[2] - vx0) * vs, (v[3] - vy0) * vs
            margin = min(sx0, sy0, w - sx1, h - sy1) / min(w, h)
            if margin < 0.035:
                cands.append({"type": "too_close_to_frame_edge", "scene": si + 1, "t": round(v[5], 2),
                              "severity": "medium" if (margin < 0.0 or tag.startswith("text")) else "low",
                              "metric": f"{tag} ends {margin * min(w, h):.0f}px from the frame edge on screen ({margin:.1%}; guideline 3.5%)",
                              "frame": int(min(t1 - 0.25, v[5] + 0.3) * fps)})
        # writing speed and sync of every caption (measured from the composed program)
        words, wsrc = scene_words(row)
        for tag, v in text.items():
            label = tag[5:]
            letters = sum(1 for ch in label if ch.isalnum())
            dur_w = max(1e-3, v[5] - v[4])
            first = label.split()[0] if label.split() else ""
            tw = find_phrase_time(words, first) if first else None
            rec = {"scene": si + 1, "caption": label, "letters_per_s": round(letters / dur_w, 1), "writing_s": round(dur_w, 2),
                   "starts_at": round(v[4], 2), "ends_at": round(v[5], 2)}
            if tw is not None:
                rec["word_at"] = round(tw, 2)
                rec["start_minus_word_s"] = round(v[4] - tw, 2)       # < 0: drawn ahead of the word (anticipation)
                rec["end_minus_word_s"] = round(v[5] - tw, 2)
                rec["word_timing"] = wsrc
                if abs(v[5] - tw) > 1.8:
                    cands.append({"type": "caption_out_of_sync", "scene": si + 1, "t": round(v[5], 2), "severity": "medium",
                                  "metric": f"{label!r} finishes {v[5] - tw:+.2f}s from the word {first!r} ({wsrc} timing)", "frame": int(v[5] * fps)})
            metrics.setdefault("captions", []).append(rec)
        # pen activity: fraction of the scene during which something is being drawn
        busy = sum(min(t1, st.t1) - max(t0, st.t0) for st in prog.strokes if st.t0 < t1 and st.t1 > t0)
        meta = next((m for m in prog.meta["scenes"] if m["scene"] == si + 1), {})
        scene_strokes = [st for st in prog.strokes if st.tag in g and t0 - 1e-6 <= st.t0 < t1 + 0.6 and any(abs(st.t0 - x[4]) < 30 for x in [g[st.tag]])]
        ivs = sorted((st.t0, st.t1) for st in scene_strokes)
        gaps, cursor_t = [], None
        for a_, b_ in ivs:
            if cursor_t is not None and a_ - cursor_t > 0.6:
                gaps.append(round(a_ - cursor_t, 2))
            cursor_t = b_ if cursor_t is None else max(cursor_t, b_)
        moving_starts = sum(1 for st in scene_strokes if math.hypot(*[a - b for a, b in zip(eng.camera_at(st.t0)[:2], eng.camera_at(max(0, st.t0 - 1 / fps))[:2])]) * fps > 250)
        metrics["scenes"].append({
            "pen_idle_gaps_over_0.6s": gaps, "strokes_started_while_camera_moving": moving_starts, "strokes_in_scene": len(scene_strokes),
            "scene": si + 1, "start": t0, "end": t1, "duration": round(t1 - t0, 2), "layout": row["chalk"]["layout"],
            "first_stroke": meta.get("first_stroke"), "last_stroke_end": round(last_end, 2), "finished_dwell_s": round(dwell, 2),
            "pen_busy_fraction": round(busy / max(t1 - t0, 1e-6), 2), "pen_speedup": meta.get("pen_speedup"), "anchor_pull": meta.get("anchor_pull"),
            "pacing": meta.get("pacing"), "strokes": meta.get("strokes"),
        })
        if dwell < 0.5:
            cands.append({"type": "short_dwell", "scene": si + 1, "t": round(last_end, 2), "severity": "medium" if dwell < 0.25 else "low",
                          "metric": f"finished drawing visible for {dwell:.2f}s before the camera/eraser moves", "frame": int(last_end * fps)})

    for kind, name in (("small_text", "text is large enough to read"), ("caption_too_brief", "captions stay visible long enough to read"),
                       ("text_over_drawing", "captions do not overlap drawings"), ("too_close_to_frame_edge", "content keeps 3.5% clear of the frame edge on screen"),
                       ("short_dwell", "finished drawings get at least 0.5s of dwell"), ("caption_out_of_sync", "captions finish within 1.8s of their spoken word")):
        n = sum(1 for c in cands if c["type"] == kind)
        check(name, "WARNING" if n else "PASS", f"{n} candidate(s)" if n else "all scenes met the criterion")

    # ------------------------------------------------------------- long visually static holds
    static_runs, run = [], 0
    for i in range(1, n_frames):
        if full_diffs[i] < 0.02 and not moving[i]:
            run += 1
        else:
            if run / fps >= 2.0:
                static_runs.append(((i - run) / fps, i / fps))
            run = 0
    for a, b in static_runs:
        cands.append({"type": "static_hold", "t": round(a, 2), "t_end": round(b, 2), "severity": "low", "metric": f"{b - a:.1f}s with no visible change", "frame": int(a * fps)})
    check("no visually frozen stretch over 2s", "WARNING" if static_runs else "PASS", f"{len(static_runs)} stretch(es): {[(round(a, 1), round(b, 1)) for a, b in static_runs]}" if static_runs else "none")

    # ------------------------------------------------------------- shape sensitivity (code-level, deterministic)
    src = (ROOT / "moduli" / "chalk_illustrations.py").read_text(encoding="utf-8")
    import re
    six = len(re.findall(r"_star\([^)]*?,\s*6\s*,", src))
    if six:
        cands.append({"type": "six_pointed_star", "severity": "high", "metric": f"{six} _star(...) accent calls use 6 points (hexagram); visible as the yellow accent on several illustrations",
                      "frame": None})
    check("accent marks avoid hexagram shapes", "WARNING" if six else "PASS", f"{six} six-point star accents found" if six else "none")

    # evidence frames for every candidate
    cap = cv2.VideoCapture(str(video))
    for k, c in enumerate(cands):
        fi = c.get("frame")
        if fi is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok:
            name = f"{k:02d}_{c['type']}_t{c['t']:.1f}.png"
            cv2.imwrite(str(out / "frames" / name), fr)
            c["frame_file"] = f"frames/{name}"
    cap.release()

    metrics["fps"] = fps
    metrics["frames"] = n_frames
    metrics["frame_diff_median"] = round(med, 3)
    (out / "measurements.json").write_text(json.dumps({"video": str(video), "checks": checks, **metrics}, indent=2), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps(cands, indent=2), encoding="utf-8")
    counts = defaultdict(int)
    for c in checks:
        counts[c["status"]] += 1
    print("checks:", dict(counts), "| candidates:", len(cands))
    for c in checks:
        print(f"  [{c['status']:<7}] {c['check']}: {c['detail']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
