#!/usr/bin/env python3
"""Compare chalkboard sound-effect level variants with objective measurements and short A/B clips.

    python tools/chalk_sound_lab.py --plan out/v.scene_plan.json --audio out/narration.mp3 \\
        --music "assets/music/mysterious/The Princess - Evening Telecast.mp3" --out docs/qa/sound_lab_1 \\
        --video out/v.mp4 --clip 8:26

What it measures (all *proxies*; none replaces listening - see the status lines in the report):
  * STOI / ESTOI of the narration inside the mix  (objective intelligibility proxy; needs ``pystoi``)
  * speech-to-effects ratio in 300-4000 Hz over frames where the narrator speaks
  * per-effect level vs the narration, and per-effect peak vs the narration RMS (transient risk)
  * transient prominence of the effects bed (p99.9 / median of 5 ms RMS)
  * chalk level vs the music bed in the chalk band while strokes are drawn (presence)
  * integrated loudness / loudness range / true peak of the mix (EBU R128 via ffmpeg)

Variants are named gain tables in dB (see VARIANTS or ``--variants-json``). Gains above 0 dB are refused:
the experiment never boosts an effect to make it more noticeable.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moduli import chalk_sound as cs  # noqa: E402
from moduli.chalk_composer import compose_program  # noqa: E402

VARIANTS: dict[str, dict[str, float]] = {
    "A_original": {},
    "B_minus6dB": {"chalk": -6, "body": -6, "tick": -6, "erase": -6, "whoosh": -6},
}


def _db(x: float) -> float:
    return float(20 * np.log10(max(x, 1e-9)))


def _stft_power(x: np.ndarray, lo: float, hi: float, hop_s: float = 0.1) -> np.ndarray:
    hop = int(cs.SR * hop_s)
    win = hop * 2
    w = np.hanning(win)
    f = np.fft.rfftfreq(win, 1 / cs.SR)
    sel = (f >= lo) & (f <= hi)
    out = []
    for i in range(0, len(x) - win, hop):
        spec = np.fft.rfft(x[i:i + win] * w)
        out.append(float((np.abs(spec[sel]) ** 2).sum()))
    return np.array(out)


def ebur128(wav: Path) -> dict:
    r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(wav), "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True)
    txt = r.stderr
    def grab(pat):
        m = re.search(pat, txt)
        return float(m.group(1)) if m else None
    tail = txt[txt.rfind("Summary:"):] if "Summary:" in txt else txt
    return {"integrated_lufs": grab_in(tail, r"I:\s+(-?[\d.]+) LUFS"), "range_lu": grab_in(tail, r"LRA:\s+([\d.]+) LU"),
            "true_peak_dbtp": grab_in(tail, r"Peak:\s+(-?[\d.]+) dBFS")}


def grab_in(txt: str, pat: str):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else None


def stoi_scores(clean: np.ndarray, deg: np.ndarray):
    try:
        from pystoi import stoi
    except ImportError:
        return None
    n = min(len(clean), len(deg))
    return {"stoi": round(float(stoi(clean[:n], deg[:n], cs.SR, extended=False)), 4),
            "estoi": round(float(stoi(clean[:n], deg[:n], cs.SR, extended=True)), 4)}


def measure(name: str, comp: dict, program, duration: float, wav: Path) -> dict:
    narr, sfx, music, stems, va = comp["narration"], comp["sfx"], comp["music"], comp["stems"], comp["voice_activity"]
    n = len(narr)
    voice_rms = cs._rms(narr)
    m: dict = {"variant": name}
    # intelligibility proxy
    m["intelligibility_proxy"] = {
        "narration_plus_effects": stoi_scores(narr, narr + sfx),
        "narration_plus_music": stoi_scores(narr, narr + music),
        "full_mix": stoi_scores(narr, narr + sfx + music),
    }
    # speech-to-effects ratio over speaking frames
    hop_s = 0.1
    pv = _stft_power(narr, 300, 4000, hop_s)
    pf = _stft_power(sfx, 300, 4000, hop_s)
    pm = _stft_power(music, 300, 4000, hop_s)
    k = min(len(pv), len(pf), len(pm))
    # "speaking" = frames where the narrator's own band power is within 25 dB of its p95. (The voice-activity
    # curve has a 350 ms release tail; counting that tail made the p10 ratios meaningless.)
    thr = np.percentile(pv[:k], 95) * 10 ** (-25 / 10)
    speaking = pv[:k] > thr
    ser = 10 * np.log10((pv[:k][speaking] + 1e-12) / (pf[:k][speaking] + 1e-12))
    ser_m = 10 * np.log10((pv[:k][speaking] + 1e-12) / (pm[:k][speaking] + 1e-12))
    m["speech_to_effects_db"] = {"median": round(float(np.median(ser)), 2), "p10": round(float(np.percentile(ser, 10)), 2), "frames": int(speaking.sum())}
    m["speech_to_music_db"] = {"median": round(float(np.median(ser_m)), 2), "p10": round(float(np.percentile(ser_m, 10)), 2)}
    # levels and transients
    m["effects_rms_vs_voice_db"] = round(_db(cs._rms(sfx) / voice_rms), 2) if np.any(sfx) else None
    m["stems"] = {}
    for k_, v in stems.items():
        if not np.any(v):
            continue
        pk = float(np.abs(v).max())
        m["stems"][k_] = {"rms_vs_voice_db": round(_db(cs._rms(v) / voice_rms), 2), "peak_vs_voice_rms_db": round(_db(pk / voice_rms), 2)}
    hop = int(0.005 * cs.SR)
    fr = np.sqrt(np.mean(sfx[: len(sfx) // hop * hop].reshape(-1, hop).astype(np.float64) ** 2, axis=1))
    act = fr[fr > 0.05 * fr.max()]
    m["effects_transient_prominence_db"] = round(_db(float(np.percentile(act, 99.9)) / float(np.median(act))), 2)
    m["effects_peak_vs_voice_rms_db"] = round(_db(float(np.abs(sfx).max()) / voice_rms), 2)
    # presence of chalk against the music bed, in the chalk band, while strokes are drawn
    drawing = np.zeros(k, bool)
    for st in program.strokes:
        a, b = int(st.t0 / hop_s), int((st.t0 + st.dur) / hop_s) + 1
        drawing[a:min(b, k)] = True
    pc = _stft_power(stems["chalk"] + stems["body"] + stems["tick"], 1500, 7000, hop_s)[:k]
    pmb = _stft_power(music, 1500, 7000, hop_s)[:k]
    sel = drawing[: len(pc)] & (pc > 0)
    m["chalk_vs_music_db_in_chalk_band"] = round(float(np.median(10 * np.log10((pc[sel] + 1e-12) / (pmb[sel] + 1e-12)))), 2) if sel.any() else None
    m["mix_loudness"] = ebur128(wav)
    import wave
    with wave.open(str(wav)) as wf:
        written = np.frombuffer(wf.readframes(wf.getnframes()), np.int16).astype(np.float64) / 32768.0
    pre = float(np.abs(narr + sfx + music).max())
    m["mix_sample_peak_written_file"] = round(float(np.abs(written).max()), 3)
    m["mix_sum_peak_before_limiter"] = round(pre, 3)          # > 0.97 means the soft limiter had to act
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--music", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video")
    ap.add_argument("--clip", default="8:26", help="start:end seconds for the A/B clips")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--variants-json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    variants = json.loads(Path(args.variants_json).read_text()) if args.variants_json else VARIANTS
    for vn, g in variants.items():
        bad = {k: v for k, v in g.items() if v > 0}
        if bad:
            raise SystemExit(f"{vn}: refusing positive gains {bad}; this experiment never boosts an effect")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    duration = float(plan["duration_seconds"])
    program = compose_program(plan, seed=args.seed)
    results = {}
    for vn, g in variants.items():
        comp: dict = {}
        wav = out / f"{vn}.wav"
        cs.build_soundtrack(program, args.audio, args.music, duration, wav, music_volume=0.10, seed=5, gains=cs.SfxGains(**g), components=comp)
        results[vn] = {"gains_db": g, **measure(vn, comp, program, duration, wav)}
    (out / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if args.video:
        from moduli.montaggio import _filtro_loudnorm
        a, b = (float(x) for x in args.clip.split(":"))
        for vn in variants:
            clip = out / f"{vn}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", str(a), "-t", str(b - a), "-i", args.video, "-ss", str(a), "-t", str(b - a),
                            "-i", str(out / f"{vn}.wav"), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-af", _filtro_loudnorm(), "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2", str(clip)], check=True)
    print(json.dumps({k: {"gains": v["gains_db"], "stoi_full": (v["intelligibility_proxy"]["full_mix"] or {}).get("stoi"),
                          "ser_median_db": v["speech_to_effects_db"]["median"], "fx_vs_voice_db": v["effects_rms_vs_voice_db"],
                          "fx_peak_vs_voice_db": v["effects_peak_vs_voice_rms_db"], "lufs": v["mix_loudness"]["integrated_lufs"], "limiter_needed": v["mix_sum_peak_before_limiter"] > 0.97} for k, v in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
