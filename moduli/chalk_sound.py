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
from dataclasses import dataclass
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
STEM_ORDER = ("chalk", "body", "tick", "erase", "thump", "tap", "whoosh")


@dataclass
class SfxGains:
    """Per-effect level offsets in dB, applied *after* the fixed reference normalisation.

    ``chalk`` = granular scratch, ``body`` = low chalk body, ``tick`` = chalk touching down,
    ``erase`` = eraser swishes, ``thump`` / ``tap`` = low and click parts of the reveal impact,
    ``whoosh`` = camera moves, ``music`` = the music bed. All 0 dB reproduces the v1 mix exactly.
    """

    chalk: float = 0.0
    body: float = 0.0
    tick: float = 0.0
    erase: float = 0.0
    thump: float = 0.0
    tap: float = 0.0
    whoosh: float = 0.0
    music: float = 0.0

    def linear(self, stem: str) -> float:
        return float(10.0 ** (getattr(self, stem) / 20.0))

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in (*STEM_ORDER, "music")}


DEFAULT_SFX_DB = -15.0
# User playback feedback found the initial -6 dB trial and the subsequent -12 dB setting still left effects too prominent. Use a further 3 dB reduction
# by default, then attenuate the sharp chalk touch-down tick and impact click by another 3 dB.
# That makes the default tick/tap level -18 dB relative to the old reference mix.
# These defaults are deliberately conservative; actual production narration still needs listening review.
DEFAULT_EXTRA_DB = {"tick": -3.0, "tap": -3.0}


def default_gains() -> SfxGains:
    """The shipped effect levels: every effect stem is -15 dB by default, with tick/tap another -3 dB.

    User playback feedback found the earlier -6 dB and -12 dB experiments still too loud. Values above 0 are refused:
    this setting exists to *reduce* effects, never to boost them. ``FERN_CHALK_SFX_DB=0`` explicitly
    restores the v1 gain level for troubleshooting or comparison.
    """
    import os
    raw = os.environ.get("FERN_CHALK_SFX_DB", "").strip()
    try:
        explicit = float(raw) if raw else None
    except ValueError:
        explicit = None
    if explicit is not None:                       # an explicit setting is uniform and never positive
        return SfxGains(**{k: min(0.0, explicit) for k in STEM_ORDER})
    return SfxGains(**{k: DEFAULT_SFX_DB + DEFAULT_EXTRA_DB.get(k, 0.0) for k in STEM_ORDER})


def render_sfx_stems(program: Any, duration: float, seed: int = 5) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Synthesize every effect as its own stem (mono float32, length ``duration``) so each can be tuned alone.

    Operations and random-number order are identical to the v1 single-track version.
    """
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
    stems = {k: np.zeros(n, np.float32) for k in STEM_ORDER}
    stems["chalk"] = _bandpass(noise * env * mod, 1700, 7600) * 1.0
    stems["body"] = _bandpass(noise * env, 420, 1400) * 0.16
    stems["tick"] = _bandpass(rng.standard_normal(n).astype(np.float32) * _smooth(ticks, 2.0), 900, 5200) * 0.9

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
        stems["erase"][a0:a1] += 0.55 * seg
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
        stems["thump"][a0:a0 + length] += np.sin(phase) * np.exp(-tt * 11.0) * 0.9 * g
        stems["tap"][a0:a0 + length] += _bandpass(rng.standard_normal(length).astype(np.float32), 1200, 5200) * np.exp(-tt * 75.0) * 0.8 * g
    for c in [c for c in program.cues if c.get("kind") == "whoosh"]:
        a0 = int(c["t"] * SR)
        dur = float(c.get("dur", 1.0))
        a1 = min(n, a0 + int(dur * SR))
        if a1 <= a0:
            continue
        t = np.linspace(0, 1, a1 - a0, dtype=np.float32)
        shape = np.clip(np.sin(np.pi * t), 0, None) ** 1.8
        seg = _bandpass(rng.standard_normal(a1 - a0).astype(np.float32), 260, 3200) * shape
        stems["whoosh"][a0:a1] += 0.5 * float(c.get("gain", 1.0)) * seg

    keep = int(duration * SR)
    stems = {k: v[:keep] for k, v in stems.items()}
    report = {
        "chalk_strokes": len(program.strokes),
        "chalk_seconds": round(drawn_seconds, 2),
        "erase_events": len(erase_events),
        "impacts": len(impacts),
        "whooshes": sum(1 for c in program.cues if c.get("kind") == "whoosh"),
    }
    return stems, report


def sum_stems(stems: dict[str, np.ndarray], gains: SfxGains | None = None) -> np.ndarray:
    """Sum stems in the v1 order, each scaled by its gain (``None`` = all 0 dB)."""
    gains = gains or SfxGains()
    total = None
    for k in STEM_ORDER:
        x = stems[k] * np.float32(gains.linear(k)) if gains.linear(k) != 1.0 else stems[k]
        total = x.copy() if total is None else total + x
    return total


def render_sfx(program: Any, duration: float, seed: int = 5) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize the effects track (mono, unit-ish level) for a drawing program (all stems, 0 dB)."""
    stems, report = render_sfx_stems(program, duration, seed)
    return sum_stems(stems), report


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
                     out_wav: str | Path, *, music_volume: float = 0.10, sfx_gain: float = 1.0, seed: int = 5,
                     gains: SfxGains | None = None, components: dict[str, Any] | None = None) -> dict[str, Any]:
    """Mix narration + chalk sound design + ducked music into ``out_wav`` (mono, 44.1 kHz)."""
    narr = decode_audio(narration_path, duration)
    n = len(narr)
    stems, report = render_sfx_stems(program, duration, seed)
    stems = {k: np.pad(v, (0, max(0, n - len(v))))[:n] for k, v in stems.items()}
    narr_rms = _rms(narr)

    # Reference level: the v1 normalisation (all stems at 0 dB -> 0.30 x narration RMS). It is computed
    # from the *unmodified* stems so per-effect gains below are true offsets from v1, never undone by it.
    ref = sum_stems(stems)
    ref_rms = _rms(ref) if np.any(ref) else 1.0
    norm = 0.30 * narr_rms / ref_rms
    gains = gains or default_gains()
    sfx = sum_stems(stems, gains) * norm * sfx_gain
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
                target = narr_rms * float(np.clip(music_volume * 3.2, 0.12, 0.6)) * float(10.0 ** (gains.music / 20.0))
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
    if components is not None:       # for measurement tools (tools/chalk_sound_lab.py)
        components.update({"narration": narr, "sfx": sfx, "music": music, "stems": {k: v * np.float32(norm * sfx_gain * gains.linear(k)) for k, v in stems.items()}, "voice_activity": va_full})
    peak = float(np.max(np.abs(mix)) + 1e-9)
    if peak > 0.97:
        mix = np.tanh(mix / peak * 1.1) * 0.95          # soft limit, no hard clipping
    write_wav(out_wav, mix)
    report.update({
        "narration_rms": round(narr_rms, 4),
        "sfx_rms_vs_voice": round(_rms(sfx) / narr_rms, 3) if np.any(sfx) else 0.0,
        "gains_db": gains.as_dict(),
        "stem_rms_vs_voice": {k: round(_rms(stems[k] * norm * sfx_gain * gains.linear(k)) / narr_rms, 4) if np.any(stems[k]) else 0.0 for k in STEM_ORDER},
        "music": music_report,
        "peak": round(float(np.max(np.abs(mix))), 3),
    })
    return report
