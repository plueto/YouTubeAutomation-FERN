"""Offline dependency and input preflight for the FERN chalkboard workflow."""
from __future__ import annotations
import os
import platform
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    print("FERN CHALKBOARD PREFLIGHT")
    print("=" * 26)
    print(f"Project root: {ROOT}")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    errors = []
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: F401
        print("OK: Pillow imports")
    except Exception as exc:
        errors.append(f"Pillow import failed: {exc}")
    try:
        from moduli.montaggio import _ffmpeg, _media_duration, _mux_audio_ffmpeg, _pick_music
        ffmpeg = _ffmpeg()
        candidates = [
            Path(str(ffmpeg)),
            ROOT / "ffmpeg.exe",
            ROOT / "ffmpeg",
        ]
        found = shutil.which(str(ffmpeg))
        if found:
            candidates.append(Path(found))

        resolved = next((candidate for candidate in candidates
                         if candidate.is_file()), None)

        if resolved is None:
            errors.append(
                "FFmpeg not found. Checked FERN's local ffmpeg.exe and PATH."
            )
        else:
            ffmpeg = str(resolved.resolve())
            print(f"FFmpeg path: {ffmpeg}")
            import subprocess
            version = subprocess.run(
                [ffmpeg, "-version"],
                capture_output=True, text=True, timeout=15, check=False
            )
            lines = (version.stdout or version.stderr).splitlines()
            print("FFmpeg version: " + (lines[0] if lines else "unknown"))
            if version.returncode:
                errors.append("FFmpeg -version returned a nonzero exit code")

        print("OK: FERN montage helper imports")
    except Exception as exc:
        errors.append(f"FERN montage helpers could not load: {exc}")
        _media_duration = None
    audio = ROOT / "output" / "narration.mp3"
    script = ROOT / "director_input" / "script.txt"
    for label, path in (("Narration", audio), ("Director script", script)):
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
            continue
        print(f"{label}: {path} ({path.stat().st_size:,} bytes)")
        if path.stat().st_size == 0:
            errors.append(f"{label} exists but is empty: {path}")
    if audio.is_file() and audio.stat().st_size > 0 and _media_duration:
        try:
            duration = float(_media_duration(str(audio)))
            print(f"Narration duration: {duration:.2f} seconds")
            if duration <= 0:
                errors.append("Narration duration is not positive")
        except Exception as exc:
            errors.append(f"Narration duration probe failed: {exc}")
    if not (ROOT / "output").is_dir():
        print("Output folder does not exist yet; the preview tool can create it.")
    print()
    if errors:
        print("PREFLIGHT FAILED")
        for error in errors:
            print(" - " + error)
        print("No video was rendered. Fix these items before proceeding.")
        return 2
    print("PREFLIGHT PASSED. This check made no API calls and rendered nothing.")
    print("Next safe step: python .\\tools\\preview_chalkboard.py --duration 45")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
