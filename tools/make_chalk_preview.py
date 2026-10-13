#!/usr/bin/env python3
"""Render a chalkboard preview end to end and run QA on it.

    python tools/make_chalk_preview.py                       # first paragraph of director_input/script.txt
    python tools/make_chalk_preview.py --script my.txt --words 120
    python tools/make_chalk_preview.py --audio narration.mp3 --script my.txt   # your own narration

Narration: with ``--audio`` it is used as is (a ``<audio>.words.json`` sidecar next to it gives word-exact
timing).  Without it, narration is synthesized: Edge TTS when the network allows (word timestamps captured),
otherwise local ``espeak-ng`` (robotic *placeholder* voice, phrase-accurate timing).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--script", default=str(ROOT / "director_input" / "script.txt"))
    ap.add_argument("--audio", help="existing narration (mp3/wav); otherwise it is synthesized")
    ap.add_argument("--words", type=int, default=0, help="use only the first N words (0 = first paragraph)")
    ap.add_argument("--out", default=str(ROOT / "output" / "preview" / "fern_chalk_preview.mp4"))
    ap.add_argument("--mood", default="mysterious")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--no-qa", action="store_true")
    ap.add_argument("--force", action="store_true", help="Allow overwriting the named preview and generated narration.")
    args = ap.parse_args()

    text = Path(args.script).read_text(encoding="utf-8")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    script = " ".join(text.split()[:args.words]) if args.words else (paras[0] if paras else text.strip())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    audio = Path(args.audio).resolve() if args.audio else out.with_name("narration.mp3").resolve()
    if out.exists() and not args.force:
        print(f"Refusing to overwrite existing preview: {out}. Choose --out or pass --force.", file=sys.stderr)
        return 2
    if not args.audio and audio.exists() and not args.force:
        print(f"Refusing to overwrite existing generated narration: {audio}. Choose another --out or pass --force.", file=sys.stderr)
        return 2
    if not args.audio:
        from moduli.chalk_voice import espeak_available, synthesize_espeak
        made = False
        try:
            from moduli.audio import genera_audio
            genera_audio(script, str(audio))
            made = audio.is_file() and audio.stat().st_size > 1000
        except Exception as exc:                      # offline / no TTS available
            print(f"[preview] neural TTS unavailable ({type(exc).__name__}); trying local espeak-ng", flush=True)
        if not made:
            if not espeak_available():
                print("No narration available: pass --audio, or install espeak-ng + ffmpeg", file=sys.stderr)
                return 2
            info = synthesize_espeak(script, audio)
            print(f"[preview] placeholder narration: {info['duration']:.1f}s, {info['words']} words", flush=True)

    from moduli.chalk_render import render_chalk_video
    w, h = (int(x) for x in args.size.lower().split("x"))
    report = render_chalk_video(str(audio), script, str(out), mood=args.mood, width=w, height=h, fps=args.fps,
                                on_progress=lambda p, m: print(f"[preview] {p:3d}% {m}", flush=True) if p % 20 == 0 or p == 100 else None)
    print(f"[preview] {out}  ({report['render_seconds']}s to render {report['frames']} frames)")
    if args.no_qa:
        return 0
    import subprocess
    return subprocess.call([sys.executable, str(ROOT / "tools" / "chalk_qa.py"), str(out)])


if __name__ == "__main__":
    sys.exit(main())
