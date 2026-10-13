"""Sound design for the chalkboard renderer.

Everything is *derived from the drawing program*, so sound and picture cannot
drift apart:

* **chalk** - a scratchy, granular noise whose loudness follows how fast each
  stroke is actually being drawn; a tiny tick marks the chalk touching down
* **eraser** - low, soft swishes that pulse with each pass of the eraser
* **impact** - a chalk-tap + low thump on the reveals the story cares about
* **whoosh** - a quiet swell under camera pans
* **music** - ducked under the narrator, lifted in pauses and gently swelled on
  the biggest reveals

Levels are set relative to the measured loudness of the narration, and the final
loudness is normalised by the caller's ffmpeg encode.
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any

import numpy as np

SR = 44100


# ----------------------------------------------------------------- dsp helpers
def _moving_average(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    c = np.cumsum(np.concatenate(([0.0], x.astype(np.float64))))
    out = (c[n:] - c[:-n]) / n
    pad = len(x) - len(out)
    return np.concatenate((np.full(pad // 2, out[0] if len(out) else 0.0), out, np.full(pad - pad // 2, out[-1] if len(out) else 0.0))).astype(np.float32)


def _smooth(x: np.ndarray, ms: float) -> np.ndarray:
    n = max(1, int(SR * ms / 1000))
    return _moving_average(_moving_average(x, n), n)


def _bandpass(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    spec = np.fft.rfft(x.astype(np.float64))
    f = np.fft.rfftfreq(len(x), 1.0 / SR)
    # smooth skirts so the filter does not ring
    g = np.clip((f - lo * 0.7) / (lo * 0.3 + 1e-9), 0, 1) * np.clip((hi * 1.3 - f) / (hi * 0.3 + 1e-9), 0, 1)
    return np.fft.irfft(spec * g, n=len(x)).astype(np.float32)


def _rms(x: np.ndarray, active_only: bool = True) -> float:
    if len(x) == 0:
        return 1e-6
    if active_only:
        thr = 0.05 * float(np.max(np.abs(x)) + 1e-9)
        sel = x[np.abs(x) > thr]
        if len(sel) > 100:
            x = sel
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))) + 1e-9)


def decode_audio(path: str | Path, duration: float | None = None) -> np.ndarray:
    """Decode any audio file to mono float32 at ``SR`` using ffmpeg."""
    from moduli.ffmpeg_utils import ffmpeg_path
    cmd = [ffmpeg_path(), "-v", "error", "-nostdin", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).copy()
    if duration:
        n = int(duration * SR)
        x = x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))
    return x


def write_wav(path: str | Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# --------------------------------------------------------------------- effects
def render_sfx(program: Any, duration: float, seed: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize the effects track (mono, unit-ish level) for a drawing program."""
    rng = np.random.default_rng(seed)
    n = int(duration * SR) + SR
    env = np.zeros(n, np.float32)
    ticks = np.zeros(n, np.float32)
    drawn_seconds = 0.0
    for st in program.strokes:
        a0 = max(0, int(st.t0 * SR))
        a1 = min(n, int((st.t0 + st.dur) * SR))
        if a1 <= a0:
            continue
        speed = st.length / max(0.02, st.dur)
        amp = float(np.clip(0.18 + 0.82 * min(1.0, speed / 2400.0), 0.0, 1.0)) * float(np.clip(st.width / 6.5, 0.7, 1.25))
        env[a0:a1] = np.maximum(env[a0:a1], amp)
        drawn_seconds += st.dur
        if st.length > 140:                      # chalk touching the board
            k = min(n - 1, a0)
            ticks[k:k + int(0.006 * SR)] += 0.55
    env = _smooth(env, 9.0)

    noise = rng.standard_normal(n).astype(np.float32)
    # granular "scritch": amplitude modulated by band-limited random texture
    ctrl = rng.standard_normal(n // 60 + 3).astype(np.float32)
    mod = np.abs(np.interp(np.arange(n), np.arange(len(ctrl)) * 60.0, ctrl)).astype(np.float32)
    mod = 0.45 + 0.55 * np.clip(mod / 1.6, 0, 1)
    chalk = _bandpass(noise * env * mod, 1700, 7600) * 1.0
    body = _bandpass(noise * env, 420, 1400) * 0.16
    tick_sig = _bandpass(rng.standard_normal(n).astype(np.float32) * _smooth(ticks, 2.0), 900, 5200) * 0.9
    track = chalk + body + tick_sig

    # ---- eraser, impacts, whooshes
    erase_events = [c for c in program.cues if c.get("kind") == "erase"]
    for c in erase_events:
        a0 = int(c["t"] * SR)
        dur = float(c.get("dur", 1.0))
        a1 = min(n, a0 + int(dur * SR))
        if a1 <= a0:
            continue
        t = np.linspace(0, 1, a1 - a0, dtype=np.float32)
        passes = 7.0
        swish = 0.55 + 0.45 * np.abs(np.sin(np.pi * passes * t))
        shape = np.clip(np.sin(np.pi * np.clip(t, 0, 1)), 0, None) ** 0.6
        seg = _bandpass(rng.standard_normal(a1 - a0).astype(np.float32), 220, 2600) * swish * shape
        track[a0:a1] += 0.55 * seg
    impacts = [c for c in program.cues if c.get("kind") == "impact"]
    for c in impacts:
        a0 = int(c["t"] * SR)
        g = float(c.get("gain", 1.0))
        length = int(0.34 * SR)
        if a0 + length >= n:
            continue
        tt = np.arange(length, dtype=np.float32) / SR
        freq = 52.0 + 55.0 * np.exp(-tt * 18.0)
        phase = 2 * np.pi * np.cumsum(freq) / SR
        thump = np.sin(phase) * np.exp(-tt * 11.0) * 0.9 * g
        tap = _bandpass(rng.standard_normal(length).astype(np.float32), 1200, 5200) * np.exp(-tt * 75.0) * 0.8 * g
        track[a0:a0 + length] += thump + tap
    for c in [c for c in program.cues if c.get("kind") == "whoosh"]:
        a0 = int(c["t"] * SR)
        dur = float(c.get("dur", 1.0))
        a1 = min(n, a0 + int(dur * SR))
        if a1 <= a0:
            continue
        t = np.linspace(0, 1, a1 - a0, dtype=np.float32)
        shape = np.clip(np.sin(np.pi * t), 0, None) ** 1.8
        seg = _bandpass(rng.standard_normal(a1 - a0).astype(np.float32), 260, 3200) * shape
        track[a0:a1] += 0.5 * float(c.get("gain", 1.0)) * seg

    track = track[: int(duration * SR)]
    report = {
        "chalk_strokes": len(program.strokes),
        "chalk_seconds": round(drawn_seconds, 2),
        "erase_events": len(erase_events),
        "impacts": len(impacts),
        "whooshes": sum(1 for c in program.cues if c.get("kind") == "whoosh"),
    }
    return track, report


# ----------------------------------------------------------------------- mixing
def _control_envelope(x: np.ndarray, rate: int = 100) -> np.ndarray:
    """RMS envelope at ``rate`` Hz."""
    hop = SR // rate
    m = len(x) // hop
    if m == 0:
        return np.zeros(1, np.float32)
    frames = x[: m * hop].reshape(m, hop).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1)).astype(np.float32)


def voice_activity(narration: np.ndarray, rate: int = 100) -> np.ndarray:
    """0..1 'narrator is speaking' curve at ``rate`` Hz with fast attack / slow release."""
    env = _control_envelope(narration, rate)
    thr = 0.12 * float(np.percentile(env, 95) + 1e-9)
    raw = (env > thr).astype(np.float32)
    out = np.zeros_like(raw)
    atk, rel = 1.0 - np.exp(-1.0 / (0.04 * rate)), 1.0 - np.exp(-1.0 / (0.35 * rate))
    v = 0.0
    for i, r in enumerate(raw):
        v += (r - v) * (atk if r > v else rel)
        out[i] = v
    return out


def build_soundtrack(program: Any, narration_path: str | Path, music_path: str | Path | None, duration: float,
                     out_wav: str | Path, *, music_volume: float = 0.10, sfx_gain: float = 1.0, seed: int = 5) -> dict[str, Any]:
    """Mix narration + chalk sound design + ducked music into ``out_wav`` (mono, 44.1 kHz)."""
    narr = decode_audio(narration_path, duration)
    n = len(narr)
    sfx, report = render_sfx(program, duration, seed)
    sfx = np.pad(sfx, (0, max(0, n - len(sfx))))[:n]
    narr_rms = _rms(narr)

    # effects sit well under the voice; they are texture, not content
    sfx_rms = _rms(sfx) if np.any(sfx) else 1.0
    sfx = sfx * (0.30 * narr_rms / sfx_rms) * sfx_gain
    # stop effects from stepping on the narrator's consonants: gentle ducking by voice activity
    va = voice_activity(narr)
    t_ctl = np.arange(len(va)) / 100.0
    va_full = np.interp(np.arange(n) / SR, t_ctl, va).astype(np.float32)
    sfx = sfx * (1.0 - 0.38 * va_full)

    music_report: dict[str, Any] = {"used": False}
    music = np.zeros(n, np.float32)
    if music_path and Path(music_path).is_file():
        try:
            m = decode_audio(music_path)
            if len(m) > SR:
                reps = int(np.ceil(n / len(m)))
                m = np.tile(m, reps)[:n]
                fade_in, fade_out = int(1.2 * SR), int(min(2.8, duration / 3) * SR)
                m[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
                m[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
                base = _rms(m, active_only=False)
                # level: ~ music_volume of the narration's loudness, rising in pauses
                target = narr_rms * float(np.clip(music_volume * 3.2, 0.12, 0.6))
                gain = np.full(n, target / base, dtype=np.float32)
                gain *= (0.40 + 0.60 * (1.0 - va_full) ** 1.0)          # duck under the voice, open in pauses
                # swell on the big reveals
                for c in program.cues:
                    if c.get("kind") == "impact" and float(c.get("gain", 1.0)) >= 1.0:
                        a0 = int(c["t"] * SR)
                        a1 = min(n, a0 + int(1.6 * SR))
                        if a1 > a0:
                            bump = np.sin(np.linspace(0, np.pi, a1 - a0, dtype=np.float32)) ** 2
                            gain[a0:a1] *= 1.0 + 0.45 * bump
                music = m * _smooth(gain, 40.0)
                music_report = {"used": True, "file": Path(music_path).name, "target_vs_voice": round(target / narr_rms, 3)}
        except (subprocess.CalledProcessError, OSError, ValueError):
            music_report = {"used": False, "error": "music decode failed"}

    mix = narr + sfx + music
    peak = float(np.max(np.abs(mix)) + 1e-9)
    if peak > 0.97:
        mix = np.tanh(mix / peak * 1.1) * 0.95          # soft limit, no hard clipping
    write_wav(out_wav, mix)
    report.update({
        "narration_rms": round(narr_rms, 4),
        "sfx_rms_vs_voice": round(_rms(sfx) / narr_rms, 3) if np.any(sfx) else 0.0,
        "music": music_report,
        "peak": round(float(np.max(np.abs(mix))), 3),
    })
    return report
