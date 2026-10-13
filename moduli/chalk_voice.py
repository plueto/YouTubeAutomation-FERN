"""Word-level narration timing for the chalkboard pipeline.

Edge TTS can report a timestamp for every word it speaks.  ``audio.py`` saves
those next to the narration as ``<audio>.words.json``; this module reads them,
aligns the spoken words to the script's words (TTS normalises numbers,
contractions and symbols), and hands the director a ``word_timings`` list.

For environments without a neural voice (CI, offline sandboxes) it can also
synthesize narration locally with ``espeak-ng``.  That voice is robotic - it is
a *timing-accurate placeholder*, not a production voice - and its sidecar says
so (``precision: "phrase"``: phrase boundaries are measured, words inside a
phrase are interpolated).
"""
from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

SIDE_SUFFIX = ".words.json"


def sidecar_path(audio_path: str | Path) -> Path:
    return Path(str(audio_path) + SIDE_SUFFIX)


def write_sidecar(audio_path: str | Path, words: list[dict[str, Any]], *, source: str, precision: str = "word") -> Path:
    path = sidecar_path(audio_path)
    path.write_text(json.dumps({"source": source, "precision": precision, "words": words}, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def read_sidecar(audio_path: str | Path) -> dict[str, Any] | None:
    path = sidecar_path(audio_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    words = data.get("words") if isinstance(data, dict) else None
    if not isinstance(words, list) or not words:
        return None
    clean = []
    for w in words:
        try:
            clean.append({"w": str(w["w"]), "s": float(w["s"]), "e": float(w["e"])})
        except (KeyError, TypeError, ValueError):
            return None
    return {"source": str(data.get("source", "unknown")), "precision": str(data.get("precision", "word")), "words": clean}


def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", w.casefold())


def align_to_script(sidecar: dict[str, Any], script: str, duration: float) -> dict[str, Any]:
    """Map spoken-word timestamps onto the script's whitespace-separated words."""
    sw = script.split()
    tw = sidecar["words"]
    if not sw or not tw:
        return {"ok": False, "reason": "empty script or sidecar"}
    a = [_norm(w) for w in sw]
    b = [_norm(w["w"]) for w in tw]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out: list[tuple[float, float] | None] = [None] * len(sw)
    matched = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                out[i1 + k] = (tw[j1 + k]["s"], tw[j1 + k]["e"])
            matched += i2 - i1
        elif tag == "replace" and j2 > j1:
            span0, span1 = tw[j1]["s"], tw[j2 - 1]["e"]
            weights = [max(1, len(a[i])) for i in range(i1, i2)]
            total = float(sum(weights))
            t = span0
            for i, wgt in zip(range(i1, i2), weights):
                d = (span1 - span0) * wgt / total
                out[i] = (t, t + d)
                t += d
    # interpolate anything still unknown between its timed neighbours
    for i in range(len(out)):
        if out[i] is None:
            lo = next((j for j in range(i - 1, -1, -1) if out[j] is not None), None)
            hi = next((j for j in range(i + 1, len(out)) if out[j] is not None), None)
            t0 = out[lo][1] if lo is not None else 0.0
            t1 = out[hi][0] if hi is not None else duration
            gap = [j for j in range(i, len(out)) if out[j] is None and (hi is None or j < hi)]
            step = max(0.0, t1 - t0) / max(1, len(gap))
            for n, j in enumerate(gap):
                out[j] = (t0 + n * step, t0 + (n + 1) * step)
    coverage = matched / len(sw)
    return {
        "ok": True,
        "word_timings": [(float(s), float(e)) for s, e in out],     # type: ignore[misc]
        "coverage": round(coverage, 3),
        "timing_source": "tts_word_boundaries" if sidecar.get("precision") == "word" else "tts_phrase_boundaries_interpolated",
        "backend": sidecar.get("source", "tts sidecar"),
        "precision": sidecar.get("precision", "word"),
    }


def sidecar_alignment(audio_path: str | Path | None, script: str, duration: float) -> dict[str, Any] | None:
    """Alignment from a TTS sidecar if one exists next to ``audio_path`` and fits the script."""
    if not audio_path:
        return None
    side = read_sidecar(audio_path)
    if not side:
        return None
    res = align_to_script(side, script, duration)
    if not res.get("ok") or res["coverage"] < 0.6:
        return None
    return res


# --------------------------------------------------------------- local synthesis
def espeak_available() -> bool:
    return shutil.which("espeak-ng") is not None and shutil.which("ffmpeg") is not None


def _read_wav(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as w:
        return w.readframes(w.getnframes()), w.getframerate()


def synthesize_espeak(script: str, out_audio: str | Path, *, speed: int = 158, pitch: int = 38, voice: str = "en-us+m3") -> dict[str, Any]:
    """Synthesize ``script`` phrase by phrase with espeak-ng; write audio + word-timing sidecar.

    Returns ``{"duration": seconds, "words": n, "path": ...}``.
    """
    import tempfile

    if not espeak_available():
        raise RuntimeError("espeak-ng and ffmpeg are required for local narration synthesis")
    out_audio = Path(out_audio)
    text = re.sub(r"\s+", " ", script).strip()
    # phrases end at punctuation; the punctuation decides the pause that follows
    parts = re.findall(r"[^.!?;:,—]+[.!?;:,—]*", text)
    pcm = bytearray()
    rate = 22050
    words_out: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        for n, phrase in enumerate(parts):
            phrase = phrase.strip()
            if not phrase:
                continue
            wav = Path(tmp) / f"p{n}.wav"
            subprocess.run(["espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch), "-g", "1", "-w", str(wav), phrase],
                           check=True, capture_output=True)
            data, rate = _read_wav(wav)
            # trim espeak's lead/tail silence so pauses are controlled by us
            samples = memoryview(bytes(data)).cast("h")
            thr = 400
            lo = next((i for i, v in enumerate(samples) if abs(v) > thr), 0)
            hi = next((i for i in range(len(samples) - 1, -1, -1) if abs(samples[i]) > thr), len(samples) - 1)
            lo = max(0, lo - int(rate * 0.02))
            hi = min(len(samples), hi + int(rate * 0.04))
            seg = bytes(data[lo * 2:hi * 2])
            t0 = len(pcm) / 2 / rate
            dur = len(seg) / 2 / rate
            toks = phrase.split()
            weights = [max(1, len(re.sub(r"\W", "", t))) ** 0.85 for t in toks]
            total = float(sum(weights))
            t = t0
            for tok, wgt in zip(toks, weights):
                d = dur * wgt / total
                words_out.append({"w": tok, "s": round(t, 3), "e": round(t + d, 3)})
                t += d
            pcm += seg
            end = phrase[-1]
            pause = 0.46 if end in ".!?" else 0.26 if end in ";:—" else 0.16 if end == "," else 0.05
            pcm += b"\x00\x00" * int(rate * pause)
    wav_path = out_audio.with_suffix(".tmp.wav")
    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(pcm))
    out_audio.parent.mkdir(parents=True, exist_ok=True)
    if out_audio.suffix.lower() == ".wav":
        shutil.move(str(wav_path), out_audio)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path), "-ar", "44100", "-ac", "2", "-b:a", "160k", str(out_audio)], check=True)
        wav_path.unlink(missing_ok=True)
    duration = len(pcm) / 2 / rate
    write_sidecar(out_audio, words_out, source=f"espeak-ng {voice} (local placeholder voice)", precision="phrase")
    return {"duration": duration, "words": len(words_out), "path": str(out_audio)}
