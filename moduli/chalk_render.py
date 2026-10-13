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
    rows = plan["scenes"]
    plan["duration_seconds"] = round(duration, 3)
    plan["render_dimensions"] = [width, height]
    plan["fps"] = fps
    plan["planner_version"] = "chalk-engine-v2"

    program = compose_program(plan, seed=seed)
    report = _program_report(program, plan)
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
        try:
            assert proc.stdin is not None
            for i in range(total_frames):
                frame = engine.render(i / fps)
                proc.stdin.write(frame.tobytes())
                pct = int((i + 1) / total_frames * 82)
                if on_progress and pct >= last + 5:
                    last = pct
                    el = time.time() - started
                    eta = el / (i + 1) * (total_frames - i - 1)
                    on_progress(pct, f"chalkboard drawing; about {int(eta // 60)}m {int(eta % 60)}s left")
            proc.stdin.close()
            err = proc.stderr.read() if proc.stderr else b""
            if proc.wait():
                raise RuntimeError("Chalkboard FFmpeg render failed: " + err.decode("utf-8", "replace")[-3000:])
        except (BrokenPipeError, OSError) as exc:
            if proc.poll() is None:
                proc.kill()
            err = proc.stderr.read() if proc.stderr else b""
            raise RuntimeError("Chalkboard frame stream failed: " + err.decode("utf-8", "replace")[-2000:]) from exc

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
