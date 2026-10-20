"""End-to-end renderer for the FERN chalk engine.

narration (+ word-timing sidecar) -> director plan -> drawing program ->
frames (OpenCV/numpy) -> ffmpeg -> soundtrack (chalk sound design + music) -> mp4

``render_chalk_video`` is what ``chalkboard.render_chalkboard_video`` calls; it
can also be used directly (see ``tools/make_chalk_preview.py``).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from moduli.chalk_composer import compose_program
from moduli.chalk_engine import ChalkEngine, Program
from moduli.chalk_sound import build_soundtrack


def _cover_source_frame(frame, width: int, height: int, cv2):
    """Center-crop one decoded BGR frame to the output canvas and return RGB."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    source_h, source_w = frame.shape[:2]
    if source_h <= 0 or source_w <= 0:
        return None
    scale = max(width / source_w, height / source_h)
    resized_w = max(width, int(round(source_w * scale)))
    resized_h = max(height, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=interpolation)
    left = max(0, (resized_w - width) // 2)
    top = max(0, (resized_h - height) // 2)
    cropped = resized[top:top + height, left:left + width]
    if cropped.shape[1] != width or cropped.shape[0] != height:
        cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)


def _program_report(program: Program, plan: dict[str, Any]) -> dict[str, Any]:
    strokes = program.strokes
    return {
        "scenes": program.meta.get("scenes", []),
        "pages": program.meta.get("pages"),
        "stroke_count": len(strokes),
        "camera_keys": len(program.camera),
        "erases": sum(1 for e in program.erases if e.visible),
        "cues": {k: sum(1 for c in program.cues if c.get("kind") == k) for k in {c.get("kind") for c in program.cues}},
        "timing_source": plan.get("timing_source"),
        "alignment_backend": plan.get("alignment_backend"),
        "director": plan.get("director"),
    }


def render_chalk_video(audio_path: str, script_text: str, output_path: str, *, mood: str | None = None,
                       on_progress: Callable[[int, str], None] | None = None, custom_music: str | None = None,
                       use_ai_director: bool = False, align_audio: bool = False, aligner_model: str = "base",
                       scene_plan: dict[str, Any] | None = None, width: int | None = None, height: int | None = None,
                       fps: int | None = None, seed: int = 7, sound: bool = True) -> dict[str, Any]:
    from moduli.chalkboard import _find_audio_duration
    from moduli.chalkboard_director import build_scene_plan
    from moduli.montaggio import (
        _bg_volume, _ffmpeg, _filtro_loudnorm, _pick_music, AUDIO_BITRATE, AUDIO_CHANNELS, AUDIO_SAMPLE_RATE,
        FFMPEG_THREADS, OUTPUT_FPS, OUTPUT_H, OUTPUT_W, X264_CRF, X264_PRESET,
    )

    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Chalkboard mode: narration audio not found: {audio_path}")
    duration = _find_audio_duration(audio_path)
    width, height = int(width or OUTPUT_W), int(height or OUTPUT_H)
    fps = int(fps or max(12, min(30, int(OUTPUT_FPS))))
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if scene_plan is None:
        beat = float(os.environ.get("FERN_CHALK_BEAT_SECONDS", "7.5"))
        plan = build_scene_plan(script_text, duration, audio_path=audio_path, use_ai=use_ai_director,
                                align_audio=align_audio, aligner_model=aligner_model, target_seconds=beat)
    else:
        plan = scene_plan
        if not isinstance(plan, dict) or not isinstance(plan.get("scenes"), list) or not plan["scenes"]:
            raise ValueError("Provided chalkboard scene plan is invalid")
        if any("chalk" not in r for r in plan["scenes"]):
            from moduli.chalk_story import plan_chalk_scenes
            plan_chalk_scenes([r for r in plan["scenes"] if "chalk" not in r])
    # Model-generated paths are never trusted. Real footage must pass the local
    # manifest's relevance, source-type, and usage-rights gates; otherwise this
    # chunk remains in the chalk engine.
    for row in plan.get("scenes", []):
        if isinstance(row, dict):
            row.pop("footage", None)
    root = Path(__file__).resolve().parents[1]
    configured_manifest = os.environ.get("FERN_FOOTAGE_MANIFEST", "").strip()
    manifest_path = Path(configured_manifest).expanduser() if configured_manifest else root / "output" / "footage_candidates.json"
    try:
        from moduli.credible_footage import apply_footage_manifest
        plan = apply_footage_manifest(plan, script_text, manifest_path)
    except Exception as exc:
        plan["footage_report"] = {"status": "selection_error", "message": f"{type(exc).__name__}: {exc}"}
        print(f"[chalkboard] footage selector unavailable; all chunks stay chalk-only ({type(exc).__name__})", flush=True)
    footage_report = plan.get("footage_report") if isinstance(plan.get("footage_report"), dict) else {}
    selected_footage_scenes = sum(1 for item in plan.get("scenes", []) if isinstance(item, dict) and isinstance(item.get("footage"), dict))
    print(f"[chalkboard] hybrid visual plan: {selected_footage_scenes} footage chunks, {len(plan.get('scenes', [])) - selected_footage_scenes} chalk chunks", flush=True)
    rows = plan["scenes"]
    plan["duration_seconds"] = round(duration, 3)
    plan["render_dimensions"] = [width, height]
    plan["fps"] = fps
    plan["planner_version"] = "chalk-engine-v2"

    program = compose_program(plan, seed=seed)
    report = _program_report(program, plan)
    report["footage_selection"] = footage_report
    (target.with_suffix(".scene_plan.json")).write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (target.with_suffix(".chalk_report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    engine = ChalkEngine(program, width, height, fps, seed=seed)
    total_frames = max(1, int(round(duration * fps)))
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="fern_chalk_", dir=str(target.parent)) as tmp:
        silent = str(Path(tmp) / "silent.mp4")
        cmd = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", str(X264_PRESET), "-crf", str(X264_CRF),
            "-threads", str(FFMPEG_THREADS), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-t", f"{duration:.3f}", silent,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        last = -1
        source_frames_rendered = 0
        footage_scene_failures: list[str] = []
        scene_index = 0
        active_scene_index = -1
        active_footage_path = None
        active_capture = None
        try:
            try:
                import cv2
            except ImportError:
                cv2 = None
            assert proc.stdin is not None
            for i in range(total_frames):
                timestamp = i / fps
                while scene_index < len(rows) - 1 and timestamp >= float(rows[scene_index].get("end_seconds", duration)):
                    scene_index += 1
                if scene_index != active_scene_index:
                    if active_capture is not None:
                        active_capture.release()
                        active_capture = None
                    active_scene_index = scene_index
                    footage = rows[scene_index].get("footage") or {}
                    candidate_path = str(footage.get("path") or "").strip() if isinstance(footage, dict) else ""
                    active_footage_path = candidate_path or None
                    if candidate_path and cv2 is not None and os.path.isfile(candidate_path):
                        try:
                            active_capture = cv2.VideoCapture(candidate_path)
                            if active_capture.isOpened():
                                source_start = max(0.0, float(footage.get("source_start_seconds") or 0.0))
                                if source_start:
                                    active_capture.set(cv2.CAP_PROP_POS_MSEC, source_start * 1000.0)
                            else:
                                active_capture.release()
                                active_capture = None
                                footage_scene_failures.append(f"{rows[scene_index].get('id', scene_index)}: couldn't open clip")
                        except Exception as exc:
                            if active_capture is not None:
                                active_capture.release()
                            active_capture = None
                            footage_scene_failures.append(f"{rows[scene_index].get('id', scene_index)}: {type(exc).__name__}")
                    elif candidate_path:
                        footage_scene_failures.append(f"{rows[scene_index].get('id', scene_index)}: video decoder unavailable or file missing")
                frame = None
                if active_capture is not None:
                    try:
                        ok, decoded = active_capture.read()
                        if ok:
                            frame = _cover_source_frame(decoded, width, height, cv2)
                            if frame is not None:
                                source_frames_rendered += 1
                        if frame is None:
                            active_capture.release()
                            active_capture = None
                            footage_scene_failures.append(f"{rows[scene_index].get('id', scene_index)}: clip ended or decode failed; chalk fallback")
                    except Exception as exc:
                        active_capture.release()
                        active_capture = None
                        footage_scene_failures.append(f"{rows[scene_index].get('id', scene_index)}: decoder error ({type(exc).__name__}); chalk fallback")
                if frame is None:
                    frame = engine.render(timestamp)
                proc.stdin.write(frame.tobytes())
                pct = int((i + 1) / total_frames * 82)
                if on_progress and pct >= last + 5:
                    last = pct
                    el = time.time() - started
                    eta = el / (i + 1) * (total_frames - i - 1)
                    mode = "licensed footage + chalk chunks" if selected_footage_scenes else "chalk drawing"
                    on_progress(pct, f"{mode}; about {int(eta // 60)}m {int(eta % 60)}s left")
            if active_capture is not None:
                active_capture.release()
                active_capture = None
            proc.stdin.close()
            err = proc.stderr.read() if proc.stderr else b""
            if proc.wait():
                raise RuntimeError("Chalkboard FFmpeg render failed: " + err.decode("utf-8", "replace")[-3000:])
        except (BrokenPipeError, OSError) as exc:
            if active_capture is not None:
                active_capture.release()
            if proc.poll() is None:
                proc.kill()
            err = proc.stderr.read() if proc.stderr else b""
            raise RuntimeError("Chalkboard frame stream failed: " + err.decode("utf-8", "replace")[-2000:]) from exc
        report["footage_render"] = {
            "source_frames_rendered": source_frames_rendered,
            "scene_failures": footage_scene_failures,
            "mode": "hybrid" if source_frames_rendered else "chalk_only",
        }

        if on_progress:
            on_progress(86, "designing chalk sound and mixing")
        music = custom_music if custom_music and os.path.isfile(custom_music) else _pick_music(mood)
        mix_wav = str(Path(tmp) / "mix.wav")
        if sound:
            sound_report = build_soundtrack(program, audio_path, music, duration, mix_wav, music_volume=_bg_volume(), seed=seed)
        else:
            sound_report = {"disabled": True}
        mux = [
            _ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", silent,
            "-i", mix_wav if sound else audio_path,
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
            "-af", _filtro_loudnorm(), "-c:a", "aac", "-b:a", str(AUDIO_BITRATE), "-ar", str(AUDIO_SAMPLE_RATE),
            "-ac", str(AUDIO_CHANNELS), "-t", f"{duration:.3f}", "-movflags", "+faststart", str(target),
        ]
        res = subprocess.run(mux, capture_output=True)
        if res.returncode:
            raise RuntimeError("Chalkboard audio mux failed: " + res.stderr.decode("utf-8", "replace")[-2000:])
        if on_progress:
            on_progress(100, "chalkboard video ready")
    report["sound"] = sound_report
    report["render_seconds"] = round(time.time() - started, 1)
    report["frames"] = total_frames
    report["fps"] = fps
    report["resolution"] = [width, height]
    report["duration"] = round(duration, 3)
    (target.with_suffix(".chalk_report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[chalkboard] Chalk engine rendered {len(rows)} scenes, {len(program.strokes)} strokes: {target}", flush=True)
    return report
