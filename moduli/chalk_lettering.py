"""Hand-lettering for the chalk engine.

A normal TTF can only be *typed in*; chalk lettering has to be *written*.  This
module defines a single-line (stroke) capital font: every glyph is a short list
of pen strokes, in the order a person would write them.  The layout code adds
the small human errors that make lettering look chalked rather than printed:
baseline wander, per-letter tilt and width, and a slight word-level slope.

Glyph space: cap height = 1.0 (y grows downward, 0 = top, 1 = baseline),
x starts at 0.  Each glyph is ``(advance, [stroke, ...])``.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

import numpy as np

Pt = tuple[float, float]


def arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int = 14) -> list[Pt]:
    """Elliptical arc, degrees, screen coordinates (y down)."""
    n = max(3, int(n * max(1.0, abs(a1 - a0) / 180.0)))
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def _o(cx: float, cy: float, rx: float, ry: float) -> list[Pt]:
    return arc(cx, cy, rx, ry, -90, 270, 18)


def _bowl(x: float, y0: float, y1: float, rx: float) -> list[Pt]:
    """Right-facing half-ellipse from (x,y0) to (x,y1)."""
    ry = (y1 - y0) / 2
    return arc(x, y0 + ry, rx, ry, -90, 90, 10)


GLYPHS: dict[str, tuple[float, list[list[Pt]]]] = {
    "A": (0.86, [[(0, 1), (0.4, 0), (0.8, 1)], [(0.14, 0.68), (0.66, 0.68)]]),
    "B": (0.7, [[(0, 0), (0, 1)],
                [(0, 0), (0.32, 0)] + _bowl(0.32, 0, 0.5, 0.26) + [(0, 0.5)],
                [(0.1, 0.5), (0.36, 0.5)] + _bowl(0.36, 0.5, 1.0, 0.29) + [(0, 1)]]),
    "C": (0.78, [arc(0.4, 0.5, 0.4, 0.5, -42, -318, 16)]),
    "D": (0.8, [[(0, 0), (0, 1)], [(0, 0), (0.26, 0)] + _bowl(0.26, 0, 1, 0.42) + [(0, 1)]]),
    "E": (0.66, [[(0.6, 0), (0, 0), (0, 1), (0.6, 1)], [(0, 0.5), (0.44, 0.5)]]),
    "F": (0.62, [[(0.58, 0), (0, 0), (0, 1)], [(0, 0.5), (0.42, 0.5)]]),
    "G": (0.82, [arc(0.4, 0.5, 0.4, 0.5, -42, -325, 16) + [(0.8, 0.8), (0.8, 0.54), (0.45, 0.54)]]),
    "H": (0.74, [[(0, 0), (0, 1)], [(0.62, 0), (0.62, 1)], [(0, 0.5), (0.62, 0.5)]]),
    "I": (0.34, [[(0.08, 0), (0.26, 0)], [(0.17, 0), (0.17, 1)], [(0.08, 1), (0.26, 1)]]),
    "J": (0.55, [[(0.5, 0), (0.5, 0.72)] + arc(0.25, 0.72, 0.25, 0.28, 0, 160, 8)]),
    "K": (0.72, [[(0, 0), (0, 1)], [(0.62, 0), (0, 0.6)], [(0.2, 0.42), (0.66, 1)]]),
    "L": (0.6, [[(0, 0), (0, 1), (0.54, 1)]]),
    "M": (0.98, [[(0, 1), (0, 0), (0.4, 0.66), (0.8, 0), (0.8, 1)]]),
    "N": (0.8, [[(0, 1), (0, 0), (0.62, 1), (0.62, 0)]]),
    "O": (0.86, [_o(0.4, 0.5, 0.4, 0.5)]),
    "P": (0.7, [[(0, 1), (0, 0)], [(0, 0), (0.34, 0)] + _bowl(0.34, 0, 0.56, 0.28) + [(0, 0.56)]]),
    "Q": (0.88, [_o(0.4, 0.5, 0.4, 0.5), [(0.5, 0.72), (0.82, 1.08)]]),
    "R": (0.74, [[(0, 1), (0, 0)], [(0, 0), (0.34, 0)] + _bowl(0.34, 0, 0.55, 0.28) + [(0, 0.55)], [(0.28, 0.55), (0.66, 1)]]),
    "S": (0.68, [arc(0.34, 0.25, 0.3, 0.25, -35, -270, 14) + arc(0.34, 0.75, 0.32, 0.25, -90, 145, 14)[1:]]),
    "T": (0.74, [[(0, 0), (0.7, 0)], [(0.35, 0), (0.35, 1)]]),
    "U": (0.76, [[(0, 0), (0, 0.7)] + arc(0.31, 0.7, 0.31, 0.3, 180, 0, 12) + [(0.62, 0)]]),
    "V": (0.82, [[(0, 0), (0.4, 1), (0.8, 0)]]),
    "W": (1.08, [[(0, 0), (0.22, 1), (0.52, 0.32), (0.82, 1), (1.04, 0)]]),
    "X": (0.74, [[(0, 0), (0.66, 1)], [(0.66, 0), (0, 1)]]),
    "Y": (0.76, [[(0, 0), (0.36, 0.5), (0.72, 0)], [(0.36, 0.5), (0.36, 1)]]),
    "Z": (0.68, [[(0, 0), (0.6, 0), (0, 1), (0.62, 1)]]),
    "0": (0.7, [_o(0.3, 0.5, 0.3, 0.5)]),
    "1": (0.46, [[(0.02, 0.22), (0.3, 0), (0.3, 1)]]),
    "2": (0.68, [arc(0.31, 0.27, 0.3, 0.27, -180, 28, 12) + [(0.0, 1), (0.62, 1)]]),
    "3": (0.66, [arc(0.28, 0.25, 0.28, 0.25, -150, 90, 12), arc(0.28, 0.745, 0.32, 0.255, -90, 150, 12)]),
    "4": (0.7, [[(0.5, 1), (0.5, 0), (0, 0.68), (0.66, 0.68)]]),
    "5": (0.66, [[(0.58, 0), (0.06, 0), (0.02, 0.45)] + arc(0.3, 0.72, 0.32, 0.28, -130, 125, 14)]),
    "6": (0.68, [[(0.5, 0.02), (0.2, 0.2), (0.02, 0.55), (0.02, 0.72)], _o(0.3, 0.7, 0.3, 0.3)]),
    "7": (0.64, [[(0, 0), (0.62, 0), (0.24, 1)]]),
    "8": (0.68, [_o(0.3, 0.26, 0.26, 0.26), _o(0.3, 0.755, 0.31, 0.245)]),
    "9": (0.68, [_o(0.3, 0.3, 0.3, 0.3), [(0.6, 0.3), (0.58, 0.66), (0.4, 0.95), (0.1, 1)]]),
    ".": (0.22, [[(0.06, 0.94), (0.06, 0.98)]]),
    ",": (0.24, [[(0.08, 0.92), (0.03, 1.12)]]),
    "!": (0.24, [[(0.07, 0), (0.07, 0.68)], [(0.07, 0.94), (0.07, 0.98)]]),
    "?": (0.62, [arc(0.3, 0.28, 0.29, 0.28, -175, 40, 12) + [(0.3, 0.64), (0.3, 0.72)], [(0.3, 0.95), (0.3, 0.99)]]),
    ":": (0.22, [[(0.07, 0.3), (0.07, 0.34)], [(0.07, 0.86), (0.07, 0.9)]]),
    ";": (0.24, [[(0.07, 0.3), (0.07, 0.34)], [(0.08, 0.84), (0.03, 1.04)]]),
    "-": (0.46, [[(0.02, 0.52), (0.4, 0.52)]]),
    "'": (0.18, [[(0.06, 0), (0.04, 0.24)]]),
    '"': (0.32, [[(0.06, 0), (0.05, 0.24)], [(0.22, 0), (0.21, 0.24)]]),
    "(": (0.3, [arc(0.34, 0.5, 0.26, 0.62, 128, 232, 10)]),
    ")": (0.3, [arc(-0.04, 0.5, 0.26, 0.62, -52, 52, 10)]),
    "/": (0.5, [[(0.46, -0.04), (0.0, 1.04)]]),
    "+": (0.6, [[(0.05, 0.5), (0.55, 0.5)], [(0.3, 0.25), (0.3, 0.75)]]),
    "=": (0.6, [[(0.05, 0.36), (0.55, 0.36)], [(0.05, 0.66), (0.55, 0.66)]]),
    "%": (0.9, [_o(0.14, 0.2, 0.13, 0.2), [(0.7, 0.0), (0.12, 1.0)], _o(0.68, 0.8, 0.13, 0.2)]),
    "$": (0.64, [arc(0.32, 0.27, 0.28, 0.23, -35, -270, 12) + arc(0.32, 0.73, 0.3, 0.23, -90, 145, 12)[1:], [(0.32, -0.08), (0.32, 1.08)]]),
    "&": (0.84, [[(0.74, 1.0), (0.12, 0.34), (0.2, 0.06), (0.4, 0.04), (0.5, 0.2), (0.34, 0.5), (0.05, 0.72),
                  (0.12, 0.96), (0.4, 1.0), (0.66, 0.72)]]),
    "#": (0.7, [[(0.22, 0.05), (0.14, 0.95)], [(0.5, 0.05), (0.42, 0.95)], [(0.04, 0.34), (0.62, 0.34)], [(0.02, 0.68), (0.6, 0.68)]]),
    "*": (0.5, [[(0.25, 0.05), (0.25, 0.5)], [(0.04, 0.16), (0.46, 0.4)], [(0.46, 0.16), (0.04, 0.4)]]),
    "→": (1.0, [[(0.0, 0.5), (0.92, 0.5)], [(0.64, 0.2), (0.94, 0.5), (0.64, 0.8)]]),
    "←": (1.0, [[(0.94, 0.5), (0.02, 0.5)], [(0.3, 0.2), (0.0, 0.5), (0.3, 0.8)]]),
    "@": (0.9, [arc(0.44, 0.5, 0.42, 0.5, -45, -330, 16), _o(0.42, 0.52, 0.16, 0.2)]),
}
SPACE_ADVANCE = 0.42
_ALIASES = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": "-", "–": "-", "…": ".", "[": "(", "]": ")"}


def _norm(ch: str) -> str:
    ch = _ALIASES.get(ch, ch)
    return ch.upper()


def text_width(text: str, cap_h: float, tracking: float = 0.16) -> float:
    w = 0.0
    for ch in text:
        c = _norm(ch)
        w += SPACE_ADVANCE * 1.0 if c == " " else (GLYPHS.get(c, (0.5, []))[0] + tracking)
    return max(0.0, w - tracking) * cap_h


def wrap_text(text: str, cap_h: float, max_width: float, tracking: float = 0.16) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if cur and text_width(trial, cap_h, tracking) > max_width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines or [""]


def letter_strokes(text: str, x: float, y: float, cap_h: float, *, align: str = "left",
                   max_width: float | None = None, line_gap: float = 0.42, tracking: float = 0.16,
                   seed: int = 0, slope: float | None = None) -> tuple[list[dict], tuple[float, float, float, float]]:
    """Lay out ``text`` and return writing strokes and the bounding box.

    ``(x, y)`` is the baseline anchor of the first line (x depends on ``align``).
    Each returned dict: ``{"pts": ndarray (N,2), "char": str, "line": int, "idx": int}``
    in writing order.  ``idx`` is the running character index (spaces skipped),
    which the composer uses to spread writing time across the word.
    """
    rng = random.Random(seed * 7919 + len(text))
    lines = wrap_text(text, cap_h, max_width, tracking) if max_width else [text]
    slope = rng.uniform(-0.018, 0.018) if slope is None else slope
    out: list[dict] = []
    char_idx = 0
    minx = miny = 1e9
    maxx = maxy = -1e9
    for li, line in enumerate(lines):
        width = text_width(line, cap_h, tracking)
        if align == "center":
            cx = x - width / 2
        elif align == "right":
            cx = x - width
        else:
            cx = x
        base_y = y + li * cap_h * (1.0 + line_gap)
        pen = 0.0
        for ch in line:
            c = _norm(ch)
            if c == " ":
                pen += SPACE_ADVANCE * cap_h
                continue
            adv, strokes = GLYPHS.get(c, (0.5, []))
            sx = rng.uniform(0.94, 1.06)
            tilt = math.radians(rng.uniform(-2.8, 2.8))
            dy = rng.uniform(-0.035, 0.035) * cap_h + slope * pen
            hscale = rng.uniform(0.96, 1.04)
            ox = cx + pen
            for s in strokes:
                arr = np.asarray(s, np.float32) * np.array([sx, hscale], np.float32)
                gx = ox + arr[:, 0] * cap_h
                gy = base_y + dy + (arr[:, 1] - 1.0) * cap_h
                # tilt around the glyph's baseline centre
                px, py = gx - (ox + adv * cap_h * 0.5), gy - (base_y + dy)
                ca, sa = math.cos(tilt), math.sin(tilt)
                gx2 = ox + adv * cap_h * 0.5 + px * ca - py * sa
                gy2 = base_y + dy + px * sa + py * ca
                # tiny hand wobble
                wob = rng.uniform(-0.012, 0.012) * cap_h
                pts = np.stack([gx2, gy2 + wob * np.sin(np.linspace(0, math.pi, len(gx2)))], axis=1)
                out.append({"pts": pts, "char": c, "line": li, "idx": char_idx})
                minx, maxx = min(minx, float(pts[:, 0].min())), max(maxx, float(pts[:, 0].max()))
                miny, maxy = min(miny, float(pts[:, 1].min())), max(maxy, float(pts[:, 1].max()))
            pen += (adv + tracking) * cap_h
            char_idx += 1
    if not out:
        return [], (x, y, x, y)
    return out, (minx, miny, maxx, maxy)
