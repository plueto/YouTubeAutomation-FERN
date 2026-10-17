"""Procedural chalk illustrations.

Every illustration is a function that returns a list of :class:`Part` in a unit
square (0..1 x 0..1).  Parts carry a *stage*:

    0  silhouette / outline        2  detail, shading, hatching
    1  key features                3  accent colour (the "point" of the drawing)

The composer draws stage by stage with small pauses, so a drawing visibly gets
richer as the narrator explains it.  :func:`build_illustration` places an
illustration on the board with hand-drawn imperfection (wobble, slight rotation,
overshoot).  New artwork can be added from SVG path data with
:func:`register_svg`.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from moduli.chalk_engine import densify
from moduli.chalk_lettering import arc, letter_strokes

Pt = tuple[float, float]


@dataclass
class Part:
    pts: np.ndarray
    stage: int = 0
    color: str = "chalk"
    weight: float = 1.0
    dash: tuple[float, float] | None = None    # (on, off) in unit-square lengths; scaled when placed


# ---------------------------------------------------------------- primitives
def _P(points: Iterable[Pt], stage: int = 0, color: str = "chalk", weight: float = 1.0) -> Part:
    return Part(np.asarray(list(points), np.float32), stage, color, weight)


def _rect(x: float, y: float, w: float, h: float, stage=0, color="chalk", weight=1.0, close=True) -> Part:
    pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if close:
        pts += [(x, y - 0.006)]
    return _P(pts, stage, color, weight)


def _ell(cx, cy, rx, ry, a0=-100, a1=265, stage=0, color="chalk", weight=1.0, n=18) -> Part:
    return _P(arc(cx, cy, rx, ry, a0, a1, n), stage, color, weight)


def _bez(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: int = 16) -> list[Pt]:
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def _hatch(poly: Sequence[Pt], gap: float = 0.035, angle: float = 45.0, inset: float = 0.008,
           stage: int = 2, color: str = "dim", weight: float = 0.55) -> list[Part]:
    """Fill a polygon with parallel pen strokes (classic hatching)."""
    a = math.radians(angle)
    ca, sa = math.cos(-a), math.sin(-a)
    pts = [(x * ca - y * sa, x * sa + y * ca) for x, y in poly]
    ys = [p[1] for p in pts]
    lo, hi = min(ys), max(ys)
    parts: list[Part] = []
    y = lo + gap * 0.5
    flip = False
    while y < hi:
        xs = []
        for i in range(len(pts)):
            (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
            if (y1 <= y < y2) or (y2 <= y < y1):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            xa, xb = xs[k] + inset, xs[k + 1] - inset
            if xb - xa > 0.012:
                seg = [(xa, y), (xb, y)] if not flip else [(xb, y), (xa, y)]
                back = [(px * math.cos(a) - py * math.sin(a), px * math.sin(a) + py * math.cos(a)) for px, py in seg]
                parts.append(_P(back, stage, color, weight))
        y += gap
        flip = not flip
    return parts


def _squig(x0: float, y0: float, x1: float, amp: float = 0.012, n: int = 7, stage=1, color="dim", weight=0.55) -> Part:
    """A line of scribbled 'text'."""
    pts = [(x0 + (x1 - x0) * i / n, y0 + (amp if i % 2 else -amp) * (0.6 if i % 3 == 0 else 1.0)) for i in range(n + 1)]
    return _P(pts, stage, color, weight)


def _bumpy(cx, cy, rx, ry, bumps=9, amp=0.12, a0=0, a1=360, seed=3) -> list[Pt]:
    rng = random.Random(seed)
    out = []
    n = bumps * 8
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        k = 1 + amp * abs(math.sin(a * bumps / 2 * (360 / max(1, a1 - a0)))) * (0.8 + 0.4 * rng.random()) - amp * 0.5
        out.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
    return out


def _star(cx, cy, r, points=8, inner=0.45, stage=3, color="yellow", weight=0.9) -> Part:
    """A chalk glint. Six points are refused: a six-pointed star reads as a hexagram, which is a
    loaded symbol on a video about being scored by a company (artifact log A-001)."""
    if points == 6:
        raise ValueError("six-pointed stars are not allowed as accents; use 4 (glint), 5 or 8 points")
    pts = []
    for i in range(points * 2 + 1):
        a = -math.pi / 2 + math.pi * i / points
        rr = r if i % 2 == 0 else r * inner
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return _P(pts, stage, color, weight)


def _glyph(text: str, cx: float, cy: float, h: float, stage=3, color="yellow", weight=1.0, seed=1) -> list[Part]:
    strokes, bb = letter_strokes(text, 0, h, h, seed=seed, slope=0.0)
    if not strokes:
        return []
    w = bb[2] - bb[0]
    parts = []
    for s in strokes:
        p = s["pts"].copy()
        p[:, 0] += cx - bb[0] - w / 2
        p[:, 1] += cy - h / 2 - 0.0
        p[:, 1] -= 0.0
        parts.append(Part(p.astype(np.float32), stage, color, weight))
    return parts




# ----------------------------------------------------------- scene-specific drawing tools
def _dashed(points: Iterable[Pt], dash: float = 0.04, gap: float = 0.028, stage: int = 0, color: str = "chalk",
            weight: float = 1.0) -> list[Part]:
    """Break a polyline into separate dashes (a drawing that is there, but not quite)."""
    pts = np.asarray(list(points), np.float32)
    if len(pts) < 2:
        return []
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    out: list[Part] = []
    s0 = 0.0
    while s0 < total - 1e-4:
        s1 = min(total, s0 + dash)
        ss = np.linspace(s0, s1, max(2, int((s1 - s0) / 0.012) + 2))
        out.append(Part(np.stack([np.interp(ss, cum, pts[:, 0]), np.interp(ss, cum, pts[:, 1])], axis=1).astype(np.float32), stage, color, weight))
        s0 += dash + gap
    return out


def _dashed1(points: Iterable[Pt], dash: float = 0.04, gap: float = 0.028, stage: int = 0, color: str = "chalk",
             weight: float = 1.0) -> Part:
    """The same dashed line as ``_dashed`` but as ONE pen stroke with lifts (cheap to draw, few strokes)."""
    return Part(np.asarray(list(points), np.float32), stage, color, weight, (dash, gap))


def _inside(poly: Sequence[Pt], xy: np.ndarray) -> np.ndarray:
    """Vectorised point-in-polygon (even-odd rule)."""
    px, py = np.array([p[0] for p in poly], np.float64), np.array([p[1] for p in poly], np.float64)
    x, y = xy[:, 0].astype(np.float64), xy[:, 1].astype(np.float64)
    res = np.zeros(len(xy), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        cond = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        res ^= cond
        j = i
    return res


def _occlude(parts: Iterable[Part], polygon: Sequence[Pt], spacing: float = 0.006) -> list[Part]:
    """Remove the pieces of ``parts`` that lie inside ``polygon``: whatever is drawn in front hides them."""
    out: list[Part] = []
    for part in parts:
        dense = densify(part.pts, spacing)
        hidden = _inside(polygon, dense)
        run: list[int] = []
        for i, h in enumerate(hidden):
            if not h:
                run.append(i)
            if (h or i == len(hidden) - 1) and len(run) >= 3:
                out.append(Part(dense[run[0]:run[-1] + 1].copy(), part.stage, part.color, part.weight))
            if h:
                run = []
    return out


def _dial(needle_deg: float, zone: tuple[float, float] | None, ends: tuple[str, str] | None, label_stage: int = 1) -> list[Part]:
    """A graduated semicircular dial. Angles in degrees on the upper half: 180 = far left, 360 = far right."""
    cx, cy, r = 0.5, 0.74, 0.43
    p: list[Part] = [_P(arc(cx, cy, r, r, 180, 360, 30)), _P([(cx - r, cy), (cx + r, cy)], 0), _P(arc(cx, cy, r * 0.66, r * 0.66, 180, 360, 22), 1, weight=0.6)]
    for i in range(0, 13, 2):
        a = math.radians(180 + 15 * i)
        long = i % 6 == 0
        r0 = r * (0.8 if long else 0.88)
        p.append(_P([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r * 0.97 * math.cos(a), cy + r * 0.97 * math.sin(a))], 0 if long else 1, weight=0.9 if long else 0.6))
    if zone:
        a0, a1 = zone
        ring = [(cx + r * 0.7 * math.cos(math.radians(a)), cy + r * 0.7 * math.sin(math.radians(a))) for a in np.linspace(a0, a1, 14)]
        ring += [(cx + r * 0.97 * math.cos(math.radians(a)), cy + r * 0.97 * math.sin(math.radians(a))) for a in np.linspace(a1, a0, 14)]
        p += _hatch(ring, 0.026, 40, inset=0.0, stage=2, color="chalk", weight=0.55)
        p += _hatch(ring, 0.05, 130, inset=0.0, stage=2, color="dim", weight=0.45)
        p.append(_P(arc(cx, cy, r * 0.985, r * 0.985, a0, a1, 12), 3, "yellow", 1.3))
    a = math.radians(needle_deg)
    p.append(_P([(cx, cy), (cx + r * 0.86 * math.cos(a), cy + r * 0.86 * math.sin(a))], 0, weight=1.4))
    p.append(_ell(cx, cy, 0.04, 0.04, stage=0, weight=1.0))
    p.append(_P([(cx + r * 0.6 * math.cos(a), cy + r * 0.6 * math.sin(a)), (cx + r * 0.86 * math.cos(a), cy + r * 0.86 * math.sin(a))], 3, "yellow", 1.6))
    p += _hatch([(cx - 0.12, cy), (cx + 0.12, cy), (cx + 0.1, cy + 0.1), (cx - 0.1, cy + 0.1)], 0.035, 45, stage=2)
    if ends:
        p += _glyph(ends[0], cx - r * 0.86, cy + 0.12, 0.07, stage=label_stage, color="dim", weight=0.8)
        p += _glyph(ends[1], cx + r * 0.86, cy + 0.12, 0.07, stage=label_stage, color="dim", weight=0.8)
    return p

# --------------------------------------------------------------- illustrations
def person() -> list[Part]:
    body = [(0.18, 0.98), (0.18, 0.74)] + arc(0.5, 0.74, 0.32, 0.2, 180, 360, 10) + [(0.82, 0.98)]
    p = [_ell(0.5, 0.27, 0.15, 0.17, weight=1.0), _P(body)]
    p += [_P([(0.45, 0.43), (0.45, 0.54)], 1, weight=0.7), _P([(0.55, 0.43), (0.55, 0.54)], 1, weight=0.7)]
    p += [_P([(0.5, 0.56), (0.42, 0.74)], 1, weight=0.7), _P([(0.5, 0.56), (0.58, 0.74)], 1, weight=0.7)]
    p += [_P([(0.43, 0.25), (0.43, 0.27)], 1, weight=0.8), _P([(0.57, 0.25), (0.57, 0.27)], 1, weight=0.8),
          _P(arc(0.5, 0.31, 0.06, 0.04, 20, 160, 6), 1, weight=0.7)]
    p += _hatch([(0.18, 0.74), (0.5, 0.56), (0.18, 0.98)], 0.04, 60, stage=2)
    p += [_P([(0.5, 0.58), (0.54, 0.72), (0.5, 0.9), (0.46, 0.72), (0.5, 0.58)], 3, "yellow", 0.9)]
    return p


def crowd() -> list[Part]:
    p: list[Part] = []
    for i, (cx, cy, s) in enumerate([(0.2, 0.46, 0.7), (0.8, 0.46, 0.7), (0.5, 0.4, 0.9), (0.33, 0.58, 0.8), (0.67, 0.58, 0.8)]):
        r = 0.09 * s
        p.append(_ell(cx, cy - 0.12 * s, r, r * 1.1, stage=min(i, 1)))
        body = [(cx - 0.14 * s, cy + 0.42 * s), (cx - 0.14 * s, cy + 0.12 * s)] + arc(cx, cy + 0.12 * s, 0.14 * s, 0.1 * s, 180, 360, 8) + [(cx + 0.14 * s, cy + 0.42 * s)]
        p.append(_P(body, min(i, 1)))
    p.append(_star(0.5, 0.1, 0.07, 4, 0.26, 3))
    return p


def robot() -> list[Part]:
    p = [_rect(0.2, 0.22, 0.6, 0.48), _P([(0.5, 0.22), (0.5, 0.1)]), _ell(0.5, 0.075, 0.035, 0.035, stage=1)]
    p += [_rect(0.1, 0.38, 0.1, 0.16, 1), _rect(0.8, 0.38, 0.1, 0.16, 1)]
    p += [_rect(0.3, 0.34, 0.14, 0.12, 1), _rect(0.56, 0.34, 0.14, 0.12, 1)]
    p += [_P([(0.34, 0.6), (0.66, 0.6)], 1), *[_P([(0.38 + i * 0.08, 0.57), (0.38 + i * 0.08, 0.64)], 1, weight=0.6) for i in range(4)]]
    p += [_P([(0.38, 0.7), (0.38, 0.8), (0.62, 0.8), (0.62, 0.7)], 0), _rect(0.22, 0.8, 0.56, 0.18, 0)]
    p += _hatch([(0.2, 0.22), (0.3, 0.22), (0.2, 0.7)], 0.035, 45, stage=2)
    p += _hatch([(0.24, 0.84), (0.76, 0.84), (0.76, 0.96), (0.24, 0.96)], 0.05, 135, stage=2)
    p += [_P([(0.33, 0.4), (0.41, 0.4)], 3, "yellow", 1.0), _P([(0.59, 0.4), (0.67, 0.4)], 3, "yellow", 1.0)]
    return p


def brain() -> list[Part]:
    outline = _bumpy(0.5, 0.48, 0.42, 0.34, bumps=8, amp=0.18, seed=5)
    p = [_P(outline)]
    p += [_P(_bez((0.5, 0.15), (0.45, 0.32), (0.56, 0.5), (0.5, 0.8)), 1)]
    for i, (x, y) in enumerate([(0.3, 0.34), (0.28, 0.52), (0.7, 0.34), (0.72, 0.52), (0.4, 0.66), (0.6, 0.66)]):
        p.append(_P(arc(x, y, 0.09, 0.06, 20 + i * 30, 250 + i * 30, 8), 1, weight=0.7))
    p += [_P([(0.44, 0.8), (0.42, 0.95), (0.58, 0.95), (0.56, 0.8)], 0, weight=0.9)]
    p += [_star(0.86, 0.1, 0.06, 4, 0.26, 3)]
    p += _hatch(outline[:12] + [(0.5, 0.48)], 0.045, 30, stage=2)
    return p


def network() -> list[Part]:
    nodes = [(0.5, 0.5), (0.18, 0.25), (0.82, 0.22), (0.12, 0.7), (0.5, 0.88), (0.88, 0.72), (0.5, 0.12)]
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 3), (2, 5), (1, 6), (4, 5)]
    p = [_ell(*nodes[0], 0.09, 0.09, stage=0)]
    for i, (x, y) in enumerate(nodes[1:], 1):
        p.append(_ell(x, y, 0.06, 0.06, stage=0 if i < 4 else 1))
    for a, b in edges:
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        d = math.hypot(x2 - x1, y2 - y1)
        ra = 0.09 if a == 0 else 0.06
        rb = 0.06
        ux, uy = (x2 - x1) / d, (y2 - y1) / d
        p.append(_P([(x1 + ux * ra, y1 + uy * ra), (x2 - ux * rb, y2 - uy * rb)], 1, weight=0.75))
    p += [_ell(*nodes[0], 0.045, 0.045, stage=3, color="yellow", weight=0.9)]
    return p


def server() -> list[Part]:
    p: list[Part] = []
    for i in range(3):
        y = 0.1 + i * 0.28
        p.append(_rect(0.15, y, 0.7, 0.22, 0))
        p.append(_P([(0.22, y + 0.11), (0.52, y + 0.11)], 1, weight=0.7))
        p.append(_ell(0.72, y + 0.11, 0.025, 0.025, stage=1))
        p.append(_ell(0.79, y + 0.11, 0.025, 0.025, stage=1 if i else 3, color="yellow" if i == 0 else "chalk"))
        p += _hatch([(0.15, y), (0.85, y), (0.85, y + 0.22), (0.15, y + 0.22)], 0.06, 45, stage=2)[::2]
    p += [_P([(0.5, 0.94), (0.5, 0.98), (0.2, 0.98)], 2, weight=0.7), _P([(0.5, 0.98), (0.8, 0.98)], 2, weight=0.7)]
    return p


def chip() -> list[Part]:
    p = [_rect(0.25, 0.25, 0.5, 0.5), _rect(0.36, 0.36, 0.28, 0.28, 1)]
    for i in range(4):
        t = 0.33 + i * 0.113
        p += [_P([(t, 0.25), (t, 0.1)], 1, weight=0.8), _P([(t, 0.75), (t, 0.9)], 1, weight=0.8),
              _P([(0.25, t), (0.1, t)], 1, weight=0.8), _P([(0.75, t), (0.9, t)], 1, weight=0.8)]
    p += [_P([(0.42, 0.5), (0.5, 0.5), (0.5, 0.42), (0.58, 0.42)], 2, weight=0.6),
          _P([(0.5, 0.5), (0.5, 0.58), (0.58, 0.58)], 2, weight=0.6)]
    p += [_ell(0.5, 0.5, 0.035, 0.035, stage=3, color="yellow", weight=1.0)]
    return p


def cloud() -> list[Part]:
    outline = [(0.15, 0.7)] + arc(0.25, 0.55, 0.12, 0.15, 90, 270, 8) + arc(0.42, 0.38, 0.17, 0.2, 200, 340, 9) \
        + arc(0.64, 0.36, 0.14, 0.17, 220, 360, 8) + arc(0.78, 0.56, 0.12, 0.14, 270, 450, 8) + [(0.85, 0.7), (0.15, 0.7)]
    p = [_P(outline)]
    p += [_P([(0.3, 0.8), (0.3, 0.92)], 1), _P([(0.5, 0.8), (0.5, 0.95)], 1), _P([(0.7, 0.8), (0.7, 0.9)], 1)]
    p += [_P([(0.26, 0.87), (0.3, 0.93), (0.34, 0.87)], 1, weight=0.8), _P([(0.66, 0.9), (0.7, 0.84), (0.74, 0.9)], 1, weight=0.8)]
    p += _hatch(outline[:-1], 0.05, 20, stage=2)[::2]
    p += [_P([(0.5, 0.82), (0.5, 0.96)], 3, "yellow", 1.0)]
    return p


def document() -> list[Part]:
    p = [_P([(0.2, 0.05), (0.62, 0.05), (0.82, 0.25), (0.82, 0.95), (0.2, 0.95), (0.2, 0.05)]),
         _P([(0.62, 0.05), (0.62, 0.25), (0.82, 0.25)], 1)]
    for i in range(6):
        p.append(_squig(0.28, 0.36 + i * 0.09, 0.74 - (0.12 if i % 3 == 2 else 0), n=8, stage=1))
    p += [_squig(0.28, 0.14, 0.52, n=5, stage=1, color="chalk", weight=0.8)]
    p += _hatch([(0.2, 0.05), (0.62, 0.05), (0.2, 0.3)], 0.045, 135, stage=2)
    p += [_ell(0.64, 0.78, 0.1, 0.1, stage=3, color="teal", weight=0.9), _P([(0.59, 0.78), (0.63, 0.83), (0.7, 0.72)], 3, "teal", 0.9)]
    return p


def papers() -> list[Part]:
    p: list[Part] = []
    for i, (dx, rot) in enumerate([(-0.1, -6), (0.0, 0), (0.1, 6)]):
        a = math.radians(rot)
        corners = [(-0.28, -0.4), (0.28, -0.4), (0.28, 0.4), (-0.28, 0.4), (-0.28, -0.4)]
        pts = [(0.5 + dx + x * math.cos(a) - y * math.sin(a), 0.5 + x * math.sin(a) + y * math.cos(a)) for x, y in corners]
        p.append(_P(pts, 0 if i < 2 else 1))
    for j in range(5):
        p.append(_squig(0.4, 0.3 + j * 0.1, 0.74, n=8, stage=2))
    p += [_P([(0.45, 0.2), (0.6, 0.2)], 3, "yellow", 1.0)]
    return p


def magnifier() -> list[Part]:
    p = [_ell(0.4, 0.4, 0.3, 0.3), _P([(0.62, 0.62), (0.92, 0.92)], 0, weight=1.6)]
    p += [_ell(0.4, 0.4, 0.23, 0.23, 205, 285, stage=1, weight=0.6)]
    p += _hatch(arc(0.4, 0.4, 0.28, 0.28, 0, 360, 20), 0.05, 50, stage=2)[::2]
    p += [_P([(0.62, 0.62), (0.92, 0.92)], 1, weight=0.5), _star(0.4, 0.4, 0.07, 4, 0.26, 3)]
    return p


def eye() -> list[Part]:
    top = _bez((0.04, 0.5), (0.28, 0.12), (0.72, 0.12), (0.96, 0.5))
    bot = _bez((0.96, 0.5), (0.72, 0.88), (0.28, 0.88), (0.04, 0.5))
    p = [_P(top + bot[1:])]
    p += [_ell(0.5, 0.5, 0.17, 0.17, stage=1), _ell(0.5, 0.5, 0.075, 0.075, stage=1, weight=0.8)]
    for i in range(7):
        a = math.radians(-150 + i * 20)
        p.append(_P([(0.5 + 0.43 * math.cos(a), 0.42 + 0.34 * math.sin(a)), (0.5 + 0.5 * math.cos(a), 0.38 + 0.4 * math.sin(a))], 2, weight=0.6))
    p += [_ell(0.5, 0.5, 0.045, 0.045, stage=3, color="yellow", weight=1.0)]
    p += _hatch(arc(0.5, 0.5, 0.15, 0.15, 0, 360, 16), 0.05, 30, stage=2)[::2]
    return p


def lock() -> list[Part]:
    p = [_rect(0.2, 0.42, 0.6, 0.5), _P([(0.3, 0.42), (0.3, 0.3)] + arc(0.5, 0.3, 0.2, 0.22, 180, 360, 12) + [(0.7, 0.42)])]
    p += [_ell(0.5, 0.62, 0.06, 0.06, stage=1), _P([(0.5, 0.68), (0.5, 0.8)], 1)]
    p += _hatch([(0.2, 0.42), (0.8, 0.42), (0.8, 0.92), (0.2, 0.92)], 0.055, 45, stage=2)
    p += [_ell(0.5, 0.62, 0.035, 0.035, stage=3, color="yellow", weight=1.0)]
    return p


def key() -> list[Part]:
    p = [_ell(0.25, 0.3, 0.17, 0.17), _P([(0.37, 0.42), (0.9, 0.95)], 0, weight=1.2),
         _P([(0.7, 0.75), (0.82, 0.63)], 1), _P([(0.82, 0.87), (0.92, 0.77)], 1),
         _ell(0.25, 0.3, 0.07, 0.07, stage=1, weight=0.8)]
    p += [_ell(0.25, 0.3, 0.115, 0.115, 200, 300, stage=2, weight=0.5), _ell(0.25, 0.3, 0.04, 0.04, stage=3, color="yellow")]
    return p


def shield() -> list[Part]:
    outline = [(0.5, 0.04), (0.88, 0.16), (0.88, 0.5)] + _bez((0.88, 0.5), (0.86, 0.75), (0.66, 0.9), (0.5, 0.97)) \
        + _bez((0.5, 0.97), (0.34, 0.9), (0.14, 0.75), (0.12, 0.5)) + [(0.12, 0.16), (0.5, 0.04)]
    p = [_P(outline), _P([(0.5, 0.04), (0.5, 0.97)], 1, weight=0.6)]
    p += _hatch(outline[:], 0.06, 45, stage=2)[::2]
    p += [_P([(0.32, 0.5), (0.45, 0.65), (0.7, 0.34)], 3, "yellow", 1.3)]
    return p


def warning() -> list[Part]:
    tri = [(0.5, 0.08), (0.95, 0.9), (0.05, 0.9), (0.5, 0.08)]
    p = [_P(tri), _P([(0.5, 0.18), (0.86, 0.84), (0.14, 0.84), (0.5, 0.18)], 1, weight=0.6)]
    p += _hatch([(0.5, 0.18), (0.86, 0.84), (0.14, 0.84)], 0.06, 45, stage=2)[::2]
    p += [_P([(0.5, 0.38), (0.5, 0.64)], 3, "yellow", 1.5), _P([(0.5, 0.74), (0.5, 0.75)], 3, "yellow", 1.5)]
    return p


def clock() -> list[Part]:
    p = [_ell(0.5, 0.5, 0.42, 0.42), _ell(0.5, 0.5, 0.38, 0.38, 0, 300, stage=1, weight=0.5)]
    for i in range(12):
        a = math.radians(i * 30 - 90)
        r0, r1 = (0.31, 0.37) if i % 3 == 0 else (0.34, 0.37)
        p.append(_P([(0.5 + r0 * math.cos(a), 0.5 + r0 * math.sin(a)), (0.5 + r1 * math.cos(a), 0.5 + r1 * math.sin(a))], 1, weight=0.8))
    p += [_P([(0.5, 0.5), (0.5, 0.2)], 1, weight=1.1), _P([(0.5, 0.5), (0.7, 0.58)], 1, weight=1.1)]
    p += [_ell(0.5, 0.5, 0.025, 0.025, stage=1)]
    p += [_P(arc(0.5, 0.5, 0.2, 0.2, -90, -10, 8), 3, "yellow", 0.9), _P([(0.69, 0.43), (0.7, 0.5), (0.64, 0.48)], 3, "yellow", 0.9)]
    return p


def hourglass() -> list[Part]:
    p = [_P([(0.2, 0.08), (0.8, 0.08)], 0, weight=1.2), _P([(0.2, 0.92), (0.8, 0.92)], 0, weight=1.2),
         _P([(0.26, 0.08), (0.26, 0.2), (0.5, 0.5), (0.26, 0.8), (0.26, 0.92)], 0),
         _P([(0.74, 0.08), (0.74, 0.2), (0.5, 0.5), (0.74, 0.8), (0.74, 0.92)], 0)]
    p += _hatch([(0.3, 0.14), (0.7, 0.14), (0.5, 0.42)], 0.05, 0, stage=2)[::1]
    p += [_P([(0.5, 0.5), (0.5, 0.78)], 1, "yellow", 0.8), _P([(0.34, 0.86), (0.5, 0.74), (0.66, 0.86)], 3, "yellow", 1.0)]
    return p


def gear() -> list[Part]:
    pts = []
    n = 10
    for i in range(n):
        a0 = 2 * math.pi * i / n
        da = 2 * math.pi / n
        for f, r in [(0.0, 0.34), (0.2, 0.34), (0.3, 0.46), (0.7, 0.46), (0.8, 0.34)]:
            a = a0 + da * f
            pts.append((0.5 + r * math.cos(a), 0.5 + r * math.sin(a)))
    pts.append(pts[0])
    p = [_P(pts), _ell(0.5, 0.5, 0.15, 0.15, stage=1), _ell(0.5, 0.5, 0.25, 0.25, 30, 250, stage=1, weight=0.5)]
    p += _hatch(arc(0.5, 0.5, 0.33, 0.33, 0, 360, 16), 0.06, 45, stage=2)[::2]
    p += [_ell(0.5, 0.5, 0.06, 0.06, stage=3, color="yellow", weight=1.0)]
    return p


def funnel() -> list[Part]:
    p = [_P([(0.1, 0.3), (0.9, 0.3), (0.58, 0.62), (0.58, 0.92), (0.42, 0.92), (0.42, 0.62), (0.1, 0.3)]),
         _ell(0.5, 0.3, 0.4, 0.06, 0, 360, stage=1, weight=0.7)]
    p += _hatch([(0.14, 0.34), (0.86, 0.34), (0.56, 0.62), (0.44, 0.62)], 0.05, 135, stage=2)
    for i, (x, y) in enumerate([(0.25, 0.1), (0.5, 0.05), (0.72, 0.12)]):
        p.append(_ell(x, y, 0.045, 0.045, stage=1))
        p.append(_P([(x, y + 0.07), (x, y + 0.14)], 1, weight=0.7))
    p += [_P([(0.5, 0.94), (0.5, 1.0)], 3, "yellow", 1.2), _star(0.5, 0.97, 0.05, 4, 0.26, 3)]
    return p


def scales() -> list[Part]:
    p = [_P([(0.5, 0.12), (0.5, 0.9)]), _P([(0.3, 0.92), (0.7, 0.92)], 0, weight=1.2), _P([(0.12, 0.26), (0.88, 0.2)], 0, weight=1.1),
         _ell(0.5, 0.12, 0.04, 0.04, stage=1)]
    for x, y in [(0.12, 0.26), (0.88, 0.2)]:
        p += [_P([(x, y), (x - 0.1, y + 0.3)], 1, weight=0.6), _P([(x, y), (x + 0.1, y + 0.3)], 1, weight=0.6),
              _P(arc(x, y + 0.3, 0.1, 0.1, 0, 180, 10), 0)]
    p += _hatch(arc(0.88, 0.5, 0.1, 0.1, 0, 180, 10), 0.04, 45, stage=2)
    p += [_P([(0.04, 0.44), (0.2, 0.44)], 1), _ell(0.88, 0.45, 0.04, 0.04, stage=3, color="yellow")]
    return p


def coins() -> list[Part]:
    p: list[Part] = []
    for i in range(4):
        y = 0.78 - i * 0.14
        p.append(_ell(0.4, y, 0.3, 0.09, 0, 360, stage=0 if i < 3 else 1))
        p.append(_P([(0.1, y), (0.1, y + 0.1)], 0 if i < 3 else 1, weight=0.9))
        p.append(_P([(0.7, y), (0.7, y + 0.1)], 0 if i < 3 else 1, weight=0.9))
        p.append(_P(arc(0.4, y + 0.1, 0.3, 0.09, 0, 180, 10), 0 if i < 3 else 1, weight=0.9))
    p += _glyph("$", 0.4, 0.28, 0.16, stage=3)
    p += [_ell(0.82, 0.38, 0.13, 0.13, stage=1), _P(arc(0.82, 0.38, 0.09, 0.09, 200, 300, 6), 2, weight=0.6)]
    p += _glyph("$", 0.82, 0.38, 0.11, stage=3, seed=2)
    return p


def moneybag() -> list[Part]:
    outline = [(0.38, 0.2), (0.62, 0.2)] + _bez((0.62, 0.2), (0.7, 0.4), (0.92, 0.55), (0.86, 0.8)) \
        + _bez((0.86, 0.8), (0.82, 0.97), (0.18, 0.97), (0.14, 0.8)) + _bez((0.14, 0.8), (0.08, 0.55), (0.3, 0.4), (0.38, 0.2))
    p = [_P(outline), _P([(0.34, 0.2), (0.4, 0.1), (0.5, 0.15), (0.6, 0.1), (0.66, 0.2)], 1), _P([(0.36, 0.24), (0.64, 0.24)], 1, weight=0.8)]
    p += _hatch(outline, 0.055, 45, stage=2)[::2]
    p += _glyph("$", 0.5, 0.62, 0.3, stage=3, weight=1.1)
    return p


def bank() -> list[Part]:
    p = [_P([(0.5, 0.06), (0.94, 0.3), (0.06, 0.3), (0.5, 0.06)]), _P([(0.1, 0.34), (0.9, 0.34)], 0)]
    for i in range(5):
        x = 0.18 + i * 0.16
        p += [_P([(x, 0.4), (x, 0.8)], 1, weight=1.0), _P([(x + 0.05, 0.4), (x + 0.05, 0.8)], 1, weight=0.7)]
    p += [_rect(0.08, 0.84, 0.84, 0.05, 0), _rect(0.04, 0.9, 0.92, 0.06, 0)]
    p += _hatch([(0.5, 0.06), (0.94, 0.3), (0.06, 0.3)], 0.05, 135, stage=2)[::2]
    p += [_ell(0.5, 0.2, 0.045, 0.045, stage=3, color="yellow")]
    return p


def factory() -> list[Part]:
    p = [_rect(0.08, 0.5, 0.84, 0.44), _P([(0.08, 0.5), (0.08, 0.34), (0.3, 0.5), (0.3, 0.34), (0.52, 0.5), (0.52, 0.34), (0.74, 0.5)], 0)]
    p += [_rect(0.74, 0.2, 0.1, 0.3, 0), _rect(0.58, 0.28, 0.08, 0.22, 1)]
    p += [_rect(0.16 + i * 0.2, 0.62, 0.12, 0.1, 1) for i in range(4)] + [_rect(0.42, 0.78, 0.16, 0.16, 1)]
    p += [_P(_bez((0.79, 0.18), (0.7, 0.1), (0.84, 0.05), (0.76, -0.02)), 2, "dim", 0.8), _P(_bez((0.62, 0.26), (0.54, 0.2), (0.66, 0.14), (0.6, 0.08)), 2, "dim", 0.7)]
    p += _hatch([(0.74, 0.2), (0.84, 0.2), (0.84, 0.5), (0.74, 0.5)], 0.045, 45, stage=2)
    p += [_P([(0.8, 0.3), (0.8, 0.45)], 3, "yellow", 0.9)]
    return p


def skyscraper() -> list[Part]:
    p = [_rect(0.3, 0.14, 0.4, 0.82), _P([(0.5, 0.14), (0.5, 0.02)], 0), _rect(0.12, 0.5, 0.18, 0.46, 0), _rect(0.7, 0.4, 0.18, 0.56, 0)]
    for r in range(8):
        for c in range(3):
            p.append(_P([(0.35 + c * 0.11, 0.2 + r * 0.09), (0.4 + c * 0.11, 0.2 + r * 0.09)], 1, weight=0.8))
    for r in range(4):
        p.append(_P([(0.15, 0.56 + r * 0.1), (0.26, 0.56 + r * 0.1)], 1, weight=0.6))
        p.append(_P([(0.74, 0.47 + r * 0.11), (0.84, 0.47 + r * 0.11)], 1, weight=0.6))
    p += _hatch([(0.3, 0.14), (0.4, 0.14), (0.4, 0.96), (0.3, 0.96)], 0.05, 45, stage=2)
    p += [_ell(0.5, 0.02, 0.03, 0.03, stage=3, color="yellow")]
    return p


def truck() -> list[Part]:
    p = [_rect(0.04, 0.22, 0.58, 0.5), _P([(0.62, 0.36), (0.82, 0.36), (0.96, 0.54), (0.96, 0.72), (0.62, 0.72)], 0),
         _P([(0.7, 0.42), (0.8, 0.42), (0.88, 0.54), (0.7, 0.54), (0.7, 0.42)], 1)]
    for x in (0.2, 0.52, 0.82):
        p += [_ell(x, 0.76, 0.1, 0.1, stage=0), _ell(x, 0.76, 0.04, 0.04, stage=1)]
    p += _hatch([(0.04, 0.22), (0.62, 0.22), (0.62, 0.72), (0.04, 0.72)], 0.07, 45, stage=2)[::2]
    p += [_P([(0.12, 0.36), (0.5, 0.36)], 3, "yellow", 1.0), _P([(0.12, 0.5), (0.4, 0.5)], 3, "yellow", 1.0)]
    return p


def chart_down() -> list[Part]:
    p = [_P([(0.1, 0.05), (0.1, 0.9), (0.95, 0.9)])]
    pts = [(0.14, 0.2), (0.3, 0.28), (0.44, 0.24), (0.6, 0.5), (0.74, 0.62), (0.9, 0.84)]
    p += [_P(pts, 1, weight=1.1)]
    for x, y in pts:
        p.append(_ell(x, y, 0.02, 0.02, stage=1, weight=0.7))
    p += _hatch(pts + [(0.9, 0.9), (0.14, 0.9)], 0.05, 90, stage=2)[::2]
    p += [_P([(0.82, 0.7), (0.9, 0.86), (0.72, 0.84)], 3, "yellow", 1.1)]
    return p


def chart_up() -> list[Part]:
    p = [_P([(0.1, 0.05), (0.1, 0.9), (0.95, 0.9)])]
    pts = [(0.14, 0.82), (0.3, 0.7), (0.44, 0.74), (0.6, 0.46), (0.74, 0.36), (0.9, 0.12)]
    p += [_P(pts, 1, weight=1.1)]
    for x, y in pts:
        p.append(_ell(x, y, 0.02, 0.02, stage=1, weight=0.7))
    p += _hatch(pts + [(0.9, 0.9), (0.14, 0.9)], 0.05, 90, stage=2)[::2]
    p += [_P([(0.8, 0.12), (0.92, 0.1), (0.9, 0.24)], 3, "yellow", 1.1)]
    return p


def bars() -> list[Part]:
    p = [_P([(0.08, 0.92), (0.95, 0.92)])]
    for i, h in enumerate([0.3, 0.5, 0.42, 0.78]):
        x = 0.14 + i * 0.2
        p.append(_rect(x, 0.92 - h, 0.14, h, 0 if i < 3 else 1))
        p += _hatch([(x, 0.92 - h), (x + 0.14, 0.92 - h), (x + 0.14, 0.92), (x, 0.92)], 0.05, 45, stage=2)[::2]
    p += [_P([(0.79, 0.1), (0.85, 0.04), (0.91, 0.1)], 3, "yellow", 1.2)]
    return p


def pie() -> list[Part]:
    p = [_ell(0.46, 0.54, 0.38, 0.38), _P([(0.46, 0.54), (0.46, 0.16)], 1), _P([(0.46, 0.54), (0.78, 0.74)], 1),
         _P([(0.46, 0.54), (0.1, 0.7)], 1)]
    wedge = [(0.58, 0.44), (0.58 + 0.34 * math.cos(math.radians(-40)), 0.44 + 0.34 * math.sin(math.radians(-40)))] \
        + arc(0.58, 0.44, 0.34, 0.34, -40, 40, 6) + [(0.58, 0.44)]
    p += [_P(wedge, 3, "yellow", 1.0)]
    p += _hatch([(0.46, 0.54), (0.1, 0.7), (0.15, 0.8), (0.46, 0.9)], 0.05, 45, stage=2)[::2]
    return p


def lightbulb() -> list[Part]:
    p = [_P(arc(0.5, 0.38, 0.28, 0.3, 120, 420, 20) + [(0.4, 0.78)]), _P([(0.4, 0.78), (0.6, 0.78)]),
         _rect(0.39, 0.82, 0.22, 0.05, 1), _rect(0.42, 0.89, 0.16, 0.05, 1)]
    p += [_P([(0.42, 0.7), (0.44, 0.5), (0.5, 0.42), (0.56, 0.5), (0.58, 0.7)], 1, weight=0.7)]
    for i in range(6):
        a = math.radians(-160 + i * 28)
        p.append(_P([(0.5 + 0.38 * math.cos(a), 0.38 + 0.4 * math.sin(a)), (0.5 + 0.47 * math.cos(a), 0.38 + 0.49 * math.sin(a))], 3, "yellow", 1.0))
    p += _hatch(arc(0.5, 0.38, 0.24, 0.26, 0, 360, 16), 0.06, 45, stage=2)[::3]
    return p


def megaphone() -> list[Part]:
    p = [_P([(0.1, 0.4), (0.3, 0.4), (0.7, 0.16), (0.7, 0.8), (0.3, 0.6), (0.1, 0.6), (0.1, 0.4)]),
         _P([(0.7, 0.16), (0.7, 0.8)], 1, weight=0.8), _P([(0.22, 0.6), (0.28, 0.86), (0.4, 0.86), (0.36, 0.64)], 1)]
    p += _hatch([(0.3, 0.4), (0.7, 0.16), (0.7, 0.8), (0.3, 0.6)], 0.055, 135, stage=2)[::2]
    for i, r in enumerate([0.1, 0.18, 0.26]):
        p.append(_P(arc(0.76, 0.48, r, r * 1.6, -50, 50, 8), 3, "yellow", 0.9))
    return p


def chat() -> list[Part]:
    p = [_P([(0.06, 0.1), (0.7, 0.1), (0.7, 0.5), (0.34, 0.5), (0.2, 0.64), (0.22, 0.5), (0.06, 0.5), (0.06, 0.1)]),
         _P([(0.4, 0.58), (0.4, 0.74), (0.62, 0.74), (0.8, 0.9), (0.78, 0.74), (0.94, 0.74), (0.94, 0.34), (0.74, 0.34)], 0)]
    p += [_squig(0.14, 0.22, 0.6, n=8, stage=1), _squig(0.14, 0.34, 0.5, n=6, stage=1)]
    p += [_ell(0.52, 0.55, 0.01, 0.01, stage=1), _ell(0.67, 0.55, 0.01, 0.01, stage=3, color="yellow"), _ell(0.82, 0.55, 0.01, 0.01, stage=3, color="yellow")]
    p += _hatch([(0.06, 0.1), (0.2, 0.1), (0.06, 0.28)], 0.05, 45, stage=2)
    return p


def laptop() -> list[Part]:
    p = [_rect(0.16, 0.14, 0.68, 0.5), _P([(0.06, 0.78), (0.94, 0.78), (0.86, 0.64), (0.14, 0.64), (0.06, 0.78)], 0),
         _P([(0.06, 0.78), (0.06, 0.85), (0.94, 0.85), (0.94, 0.78)], 0), _P([(0.42, 0.82), (0.58, 0.82)], 1, weight=0.7)]
    p += [_squig(0.24, 0.26, 0.6, n=8, stage=1), _squig(0.24, 0.38, 0.5, n=6, stage=1), _squig(0.24, 0.5, 0.66, n=9, stage=1)]
    p += _hatch([(0.16, 0.14), (0.3, 0.14), (0.16, 0.34)], 0.05, 45, stage=2)
    p += [_P([(0.6, 0.22), (0.76, 0.22), (0.76, 0.34), (0.6, 0.34), (0.6, 0.22)], 3, "yellow", 0.9)]
    return p


def phone() -> list[Part]:
    p = [_rect(0.28, 0.04, 0.44, 0.92), _rect(0.33, 0.12, 0.34, 0.68, 1), _ell(0.5, 0.88, 0.03, 0.03, stage=1),
         _P([(0.44, 0.08), (0.56, 0.08)], 1, weight=0.8)]
    p += [_squig(0.37, 0.2, 0.62, n=6, stage=1), _squig(0.37, 0.3, 0.55, n=5, stage=1)]
    p += _hatch([(0.33, 0.12), (0.48, 0.12), (0.33, 0.3)], 0.05, 45, stage=2)
    p += [_rect(0.38, 0.45, 0.24, 0.16, 3, "yellow", 0.9)]
    return p


def globe() -> list[Part]:
    p = [_ell(0.5, 0.5, 0.42, 0.42), _P([(0.08, 0.5), (0.92, 0.5)], 1, weight=0.7), _P([(0.5, 0.08), (0.5, 0.92)], 1, weight=0.7),
         _ell(0.5, 0.5, 0.2, 0.42, stage=1, weight=0.7), _P(arc(0.5, 0.5, 0.42, 0.18, 0, 180, 12), 1, weight=0.6),
         _P(arc(0.5, 0.5, 0.42, 0.18, 180, 360, 12), 1, weight=0.6)]
    p += _hatch(arc(0.5, 0.5, 0.4, 0.4, 100, 260, 14), 0.05, 45, stage=2)
    p += [_ell(0.64, 0.32, 0.05, 0.05, stage=3, color="yellow", weight=1.0)]
    return p


def pin() -> list[Part]:
    p = [_P(arc(0.5, 0.36, 0.28, 0.28, 150, 390, 16) + [(0.5, 0.94), (0.26, 0.5)]), _ell(0.5, 0.36, 0.1, 0.1, stage=1)]
    p += _hatch(arc(0.5, 0.36, 0.24, 0.24, 0, 360, 14), 0.06, 45, stage=2)[::3]
    p += [_ell(0.5, 0.36, 0.05, 0.05, stage=3, color="yellow"), _P(arc(0.5, 0.96, 0.18, 0.04, 0, 360, 10), 1, weight=0.6)]
    return p


def domino() -> list[Part]:
    p: list[Part] = []
    for i, ang in enumerate([0, 0, 8, 22, 42, 66]):
        a = math.radians(ang)
        bx = 0.06 + i * 0.12 + (0.025 * i if i > 2 else 0)
        pts = [(0, 0), (0.1, 0), (0.1, -0.56), (0, -0.56), (0, 0)]
        rot = [(bx + x * math.cos(a) - y * math.sin(a) * 1.0, 0.92 + x * math.sin(a) + y * math.cos(a)) for x, y in pts]
        p.append(_P(rot, 0 if i < 4 else 1))
        p.append(_P([(bx + (0.0 + 0.1 * 0.5) * math.cos(a) + 0.28 * math.sin(a), 0.92 - 0.28 * math.cos(a)), (bx + 0.05 * math.cos(a) + 0.28 * math.sin(a) + 0.0, 0.92 - 0.28 * math.cos(a))], 1, weight=0.6))
        p.append(_ell(bx + 0.05 * math.cos(a) + 0.4 * math.sin(a), 0.92 - 0.4 * math.cos(a) + 0.05 * math.sin(a), 0.014, 0.014, stage=1, weight=0.7))
        p.append(_ell(bx + 0.05 * math.cos(a) + 0.16 * math.sin(a), 0.92 - 0.16 * math.cos(a) + 0.02 * math.sin(a), 0.014, 0.014, stage=1, weight=0.7))
    p += [_P([(0.06, 0.96), (0.98, 0.96)], 0, weight=0.7), _P([(0.2, 0.2), (0.16, 0.08), (0.28, 0.12)], 3, "yellow", 1.0)]
    return p


def bomb() -> list[Part]:
    p = [_ell(0.46, 0.6, 0.34, 0.34), _rect(0.38, 0.2, 0.16, 0.08, 1), _P(_bez((0.54, 0.22), (0.7, 0.14), (0.68, 0.04), (0.82, 0.08)), 1, weight=0.9)]
    p += [_P(arc(0.46, 0.6, 0.24, 0.24, 190, 250, 6), 1, weight=0.7)]
    p += _hatch(arc(0.46, 0.6, 0.33, 0.33, -20, 120, 14), 0.05, 45, stage=2)
    p += [_star(0.86, 0.08, 0.09, 8, 0.4, 3)]
    return p


def trap() -> list[Part]:
    p = [_rect(0.06, 0.52, 0.88, 0.38), _P([(0.12, 0.52), (0.12, 0.26), (0.7, 0.26), (0.7, 0.52)], 0, weight=1.1)]
    coil = [(0.12 + 0.012 * i, 0.4 + (0.07 if i % 2 else -0.07)) for i in range(0, 1)]
    p += [_P([(0.2, 0.52), (0.22, 0.44), (0.26, 0.52), (0.3, 0.44), (0.34, 0.52)], 1, weight=0.7)]
    p += [_P([(0.12, 0.26), (0.7, 0.26)], 1, weight=0.6), _ell(0.12, 0.52, 0.025, 0.025, stage=1)]
    p += _hatch([(0.06, 0.52), (0.94, 0.52), (0.94, 0.9), (0.06, 0.9)], 0.07, 45, stage=2)[::2]
    p += [_P([(0.62, 0.84), (0.76, 0.6), (0.9, 0.84), (0.62, 0.84)], 3, "yellow", 1.0), _ell(0.76, 0.76, 0.015, 0.015, stage=3, color="yellow", weight=0.8)]
    return p

def web() -> list[Part]:
    p: list[Part] = []
    n = 8
    for i in range(n):
        a = 2 * math.pi * i / n
        p.append(_P([(0.5, 0.5), (0.5 + 0.46 * math.cos(a), 0.5 + 0.46 * math.sin(a))], 0, weight=0.8))
    for k, r in enumerate([0.12, 0.22, 0.32, 0.43]):
        ring = [(0.5 + r * math.cos(2 * math.pi * i / n + 0.04 * (k % 2)), 0.5 + r * math.sin(2 * math.pi * i / n)) for i in range(n + 1)]
        p.append(_P(ring, 1 if k > 1 else 0, weight=0.7))
    p += [_ell(0.74, 0.7, 0.04, 0.04, stage=3, color="yellow", weight=1.2), _ell(0.74, 0.7, 0.012, 0.012, stage=3, color="yellow")]
    return p


def iceberg() -> list[Part]:
    wl = 0.36
    wave = [(0.0 + 0.065 * i, wl + (0.012 if i % 2 else -0.012)) for i in range(16)]
    tip = [(0.32, wl), (0.40, 0.2), (0.45, 0.25), (0.54, 0.06), (0.6, 0.2), (0.66, 0.15), (0.74, wl)]
    under = [(0.3, wl + 0.01), (0.16, 0.5), (0.1, 0.64), (0.2, 0.78), (0.15, 0.88), (0.36, 0.96), (0.58, 0.99),
             (0.78, 0.92), (0.9, 0.76), (0.86, 0.6), (0.92, 0.48), (0.76, wl + 0.01), (0.3, wl + 0.01)]
    p = [_P(wave, 0, weight=0.9), _P(tip), _P(under, 0, weight=1.1)]
    p += [_P([(0.54, 0.06), (0.5, 0.18), (0.56, wl)], 1, weight=0.6),
          _P([(0.54, 0.42), (0.48, 0.62), (0.56, 0.78)], 1, weight=0.6),
          _P([(0.3, 0.6), (0.4, 0.72)], 1, weight=0.6), _P([(0.74, 0.54), (0.66, 0.7)], 1, weight=0.6)]
    p += _hatch(under[:-1], 0.06, 45, stage=2)[::2]
    p += _hatch([(0.4, wl), (0.54, 0.1), (0.7, wl)], 0.07, 135, stage=2, weight=0.4)
    p += [_P([(0.34, 0.76), (0.44, 0.67), (0.54, 0.77), (0.64, 0.69)], 3, "yellow", 1.0), _star(0.5, 0.86, 0.05, 4, 0.26, 3)]
    return p


def rocket() -> list[Part]:
    body = _bez((0.5, 0.02), (0.74, 0.2), (0.74, 0.55), (0.66, 0.78)) + [(0.34, 0.78)] + _bez((0.34, 0.78), (0.26, 0.55), (0.26, 0.2), (0.5, 0.02))
    p = [_P(body), _ell(0.5, 0.34, 0.09, 0.09, stage=1), _P([(0.34, 0.6), (0.18, 0.84), (0.34, 0.76)], 0), _P([(0.66, 0.6), (0.82, 0.84), (0.66, 0.76)], 0)]
    p += _hatch(body[:12] + [(0.4, 0.78)], 0.05, 135, stage=2)[::2]
    p += [_P([(0.42, 0.82), (0.5, 0.98), (0.58, 0.82)], 3, "yellow", 1.1), _P([(0.46, 0.84), (0.5, 0.92), (0.54, 0.84)], 3, "yellow", 0.8)]
    return p


def trophy() -> list[Part]:
    cup = [(0.28, 0.1), (0.72, 0.1), (0.7, 0.4)] + arc(0.5, 0.4, 0.2, 0.22, 0, 180, 10) + [(0.28, 0.1)]
    p = [_P(cup), _P([(0.28, 0.16), (0.1, 0.16), (0.14, 0.34), (0.3, 0.42)], 1), _P([(0.72, 0.16), (0.9, 0.16), (0.86, 0.34), (0.7, 0.42)], 1),
         _P([(0.5, 0.62), (0.5, 0.78)], 0), _rect(0.3, 0.78, 0.4, 0.06, 0), _rect(0.22, 0.86, 0.56, 0.08, 0)]
    p += _hatch([(0.28, 0.1), (0.4, 0.1), (0.4, 0.5), (0.3, 0.4)], 0.05, 45, stage=2)
    p += [_star(0.5, 0.3, 0.09, 5, 0.45, 3)]
    return p


def briefcase() -> list[Part]:
    p = [_rect(0.06, 0.3, 0.88, 0.58), _P([(0.34, 0.3), (0.34, 0.16), (0.66, 0.16), (0.66, 0.3)]), _P([(0.06, 0.56), (0.94, 0.56)], 1, weight=0.8),
         _rect(0.44, 0.5, 0.12, 0.14, 1)]
    p += _hatch([(0.06, 0.3), (0.94, 0.3), (0.94, 0.56), (0.06, 0.56)], 0.06, 45, stage=2)[::2]
    p += [_ell(0.5, 0.57, 0.02, 0.02, stage=3, color="yellow")]
    return p


def cart() -> list[Part]:
    p = [_P([(0.04, 0.14), (0.18, 0.14), (0.3, 0.64), (0.82, 0.64), (0.92, 0.28), (0.2, 0.28)]), _ell(0.38, 0.82, 0.07, 0.07), _ell(0.74, 0.82, 0.07, 0.07)]
    for x in (0.36, 0.5, 0.64, 0.78):
        p.append(_P([(x, 0.28), (x - 0.02, 0.64)], 1, weight=0.6))
    p += _hatch([(0.2, 0.28), (0.92, 0.28), (0.82, 0.64), (0.3, 0.64)], 0.07, 45, stage=2)[::2]
    p += [_rect(0.5, 0.1, 0.1, 0.1, 3, "yellow", 0.9)]
    return p


def flag() -> list[Part]:
    p = [_P([(0.2, 0.04), (0.2, 0.98)], 0, weight=1.2)]
    wave = _bez((0.2, 0.1), (0.4, 0.0), (0.6, 0.2), (0.88, 0.08)) + [(0.8, 0.28)] + _bez((0.8, 0.28), (0.58, 0.4), (0.4, 0.2), (0.2, 0.34))
    p += [_P(wave), _P(_bez((0.2, 0.22), (0.4, 0.12), (0.6, 0.32), (0.84, 0.18)), 1, weight=0.6)]
    p += _hatch(wave[:], 0.06, 45, stage=2)[::2]
    p += [_ell(0.2, 0.03, 0.03, 0.03, stage=3, color="yellow")]
    return p


def ladder() -> list[Part]:
    p = [_P([(0.28, 0.02), (0.18, 0.98)]), _P([(0.72, 0.02), (0.82, 0.98)])]
    for i in range(7):
        t = 0.1 + i * 0.13
        p.append(_P([(0.28 - 0.1 * t + 0.0, 0.02 + 0.96 * t), (0.72 + 0.1 * t, 0.02 + 0.96 * t)], 1))
    p += [_P([(0.44, 0.1), (0.5, 0.04), (0.56, 0.1)], 3, "yellow", 1.1)]
    return p


def door() -> list[Part]:
    p = [_rect(0.2, 0.06, 0.6, 0.9), _rect(0.28, 0.14, 0.44, 0.36, 1), _rect(0.28, 0.58, 0.44, 0.3, 1), _ell(0.68, 0.54, 0.025, 0.025, stage=1)]
    p += _hatch([(0.2, 0.06), (0.8, 0.06), (0.8, 0.96), (0.2, 0.96)], 0.07, 45, stage=2)[::3]
    p += [_P([(0.8, 0.96), (0.98, 0.98), (0.98, 0.9)], 3, "yellow", 0.9), _P([(0.05, 0.2), (0.14, 0.34)], 3, "yellow", 0.9)]
    return p


def reportcard() -> list[Part]:
    p = [_rect(0.12, 0.06, 0.76, 0.88)]
    p += [_squig(0.2, 0.18, 0.6, n=7, stage=1), _squig(0.2, 0.28, 0.5, n=6, stage=1)]
    p += [_P([(0.2, 0.74 - i * 0.0), (0.8, 0.74)], 1, weight=0.6) for i in range(1)]
    p += _glyph("A+", 0.5, 0.52, 0.28, stage=3, weight=1.3)
    p += [_ell(0.5, 0.52, 0.36, 0.2, -20, 330, stage=3, color="teal", weight=0.8)]
    p += _hatch([(0.12, 0.06), (0.3, 0.06), (0.12, 0.24)], 0.045, 45, stage=2)
    return p


def check() -> list[Part]:
    return [_ell(0.5, 0.5, 0.42, 0.42), _P([(0.26, 0.52), (0.44, 0.72), (0.78, 0.28)], 1, "teal", 1.5), _ell(0.5, 0.5, 0.35, 0.35, 220, 330, stage=2, weight=0.5)]


def cross() -> list[Part]:
    return [_ell(0.5, 0.5, 0.42, 0.42), _P([(0.3, 0.3), (0.7, 0.7)], 1, "yellow", 1.5), _P([(0.7, 0.3), (0.3, 0.7)], 1, "yellow", 1.5),
            _ell(0.5, 0.5, 0.35, 0.35, 220, 330, stage=2, weight=0.5)]


def question() -> list[Part]:
    p = _glyph("?", 0.5, 0.5, 0.82, stage=0, color="chalk", weight=1.9, seed=4)
    p += [_star(0.8, 0.14, 0.07, 4, 0.26, 3), _star(0.16, 0.84, 0.05, 4, 0.26, 3)]
    return p


def dollar() -> list[Part]:
    p = [_ell(0.5, 0.5, 0.42, 0.42), _ell(0.5, 0.5, 0.35, 0.35, 0, 360, stage=1, weight=0.6)]
    p += _glyph("$", 0.5, 0.5, 0.42, stage=3, weight=1.3)
    return p


def crack() -> list[Part]:
    p = [_P([(0.5, 0.0), (0.44, 0.16), (0.56, 0.3), (0.42, 0.46), (0.58, 0.6), (0.46, 0.76), (0.52, 1.0)], 0, weight=1.2)]
    p += [_P([(0.56, 0.3), (0.74, 0.34), (0.82, 0.5)], 1, weight=0.8), _P([(0.42, 0.46), (0.24, 0.5), (0.16, 0.64)], 1, weight=0.8),
          _P([(0.58, 0.6), (0.72, 0.7), (0.78, 0.86)], 2, weight=0.7)]
    p += [_star(0.5, 0.46, 0.08, 4, 0.26, 3)]
    return p


def frown() -> list[Part]:
    p = [_ell(0.5, 0.5, 0.42, 0.42), _P([(0.34, 0.4), (0.34, 0.46)], 1, weight=1.3), _P([(0.66, 0.4), (0.66, 0.46)], 1, weight=1.3),
         _P(arc(0.5, 0.78, 0.2, 0.15, 205, 335, 10), 1, weight=1.0)]
    p += [_P([(0.26, 0.3), (0.42, 0.34)], 2, weight=0.7), _P([(0.74, 0.3), (0.58, 0.34)], 2, weight=0.7)]
    p += _hatch(arc(0.5, 0.5, 0.4, 0.4, 100, 260, 12), 0.05, 45, stage=2)
    p += [_P([(0.72, 0.52), (0.74, 0.62), (0.7, 0.68)], 3, "teal", 1.0)]
    return p

def gradebook() -> list[Part]:
    """An open teacher's gradebook: ruled name column, a column of marks with one circled, a pencil."""
    left = [(0.05, 0.24), (0.5, 0.3), (0.5, 0.9), (0.05, 0.84), (0.05, 0.24)]
    right = [(0.5, 0.3), (0.95, 0.24), (0.95, 0.84), (0.5, 0.9), (0.5, 0.3)]
    p = [_P(left), _P(right), _P([(0.5, 0.3), (0.5, 0.9)], 0, weight=1.2),
         _P([(0.05, 0.84), (0.07, 0.9), (0.5, 0.96), (0.93, 0.9), (0.95, 0.84)], 1, weight=0.8)]
    for i in range(5):                               # ruled lines follow the page perspective
        t = 0.2 + i * 0.12
        p.append(_P([(0.09, 0.24 + t * 0.6), (0.47, 0.3 + t * 0.6)], 1, weight=0.55))
    p.append(_P([(0.7, 0.285), (0.7, 0.87)], 1, weight=0.5))                       # one column rule
    for i, g in enumerate("ABC"):                    # marks, one per ruled row
        y = 0.4 + i * 0.15
        p += _glyph(g, 0.6, y, 0.08, stage=1, color="chalk", weight=0.85, seed=i)
    p += _hatch([(0.46, 0.3), (0.5, 0.3), (0.5, 0.9), (0.46, 0.89)], 0.02, 90, inset=0.0, stage=2, weight=0.5)   # gutter shadow
    p += _hatch([(0.5, 0.3), (0.55, 0.29), (0.55, 0.88), (0.5, 0.9)], 0.02, 90, inset=0.0, stage=2, weight=0.5)
    p += [_P([(0.84, 0.1), (0.97, 0.47), (0.995, 0.54), (1.03, 0.45), (0.9, 0.08), (0.87, 0.04), (0.84, 0.1)], 0, weight=1.1)]
    p += [_P([(0.99, 0.51), (0.995, 0.54), (1.01, 0.5)], 3, "yellow", 1.4)]
    p.append(_ell(0.6, 0.395, 0.075, 0.06, stage=3, color="yellow", weight=1.2))
    return p


def orgchart() -> list[Part]:
    """A tiny company hierarchy: the top box is the boss - circled, crowned and shaded."""
    p: list[Part] = []
    top = (0.34, 0.1, 0.32, 0.18)
    mids = [(0.04, 0.42, 0.26, 0.15), (0.37, 0.42, 0.26, 0.15), (0.7, 0.42, 0.26, 0.15)]
    lows = [(0.04 + i * 0.24, 0.74, 0.19, 0.12) for i in range(4)]
    p.append(_rect(*top))
    for b in mids:
        p.append(_rect(*b, stage=0, weight=0.9))
    for b in lows:
        p.append(_rect(*b, stage=1, weight=0.7))
    # connectors
    p.append(_P([(0.5, 0.28), (0.5, 0.35), (0.17, 0.35), (0.17, 0.42)], 0, weight=0.8))
    p.append(_P([(0.5, 0.35), (0.83, 0.35), (0.83, 0.42)], 0, weight=0.8))
    p.append(_P([(0.5, 0.35), (0.5, 0.42)], 0, weight=0.8))
    for i, b in enumerate(lows):
        m = mids[min(2, i * 3 // 4)]
        mx = m[0] + m[2] / 2
        p.append(_P([(mx, m[1] + m[3]), (mx, 0.66), (b[0] + b[2] / 2, 0.66), (b[0] + b[2] / 2, b[1])], 1, weight=0.6))
    # a person mark in each upper box
    for b in [top]:
        cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
        p.append(_ell(cx, cy - 0.025, 0.022, 0.024, stage=1, weight=0.7))
        p.append(_P(arc(cx, cy + 0.07, 0.045, 0.04, 200, 340, 6), 1, weight=0.7))
    p += _hatch([(top[0], top[1]), (top[0] + top[2], top[1]), (top[0] + top[2], top[1] + top[3]), (top[0], top[1] + top[3])], 0.03, 45, inset=0.0, stage=2, weight=0.6)
    p.append(_P([(0.38, 0.08), (0.42, 0.01), (0.46, 0.07), (0.5, 0.0), (0.54, 0.07), (0.58, 0.01), (0.62, 0.08)], 3, "yellow", 1.3))
    p.append(_ell(0.5, 0.19, 0.27, 0.16, -20, 330, stage=3, color="yellow", weight=1.0))
    return p


def score_dial() -> list[Part]:
    """A credit-score dial, 300 to 850, needle resting in the middle of the band."""
    return _dial(247, (205, 235), ("300", "850"))


def danger_gauge() -> list[Part]:
    """A danger gauge whose needle has swung into the crosshatched red zone."""
    return _dial(318, (290, 345), ("LOW", "HIGH"), label_stage=2)


def silhouette() -> list[Part]:
    """A figure seen from behind, shaded as a solid mass: someone standing in front of something."""
    body = [(0.1, 1.0), (0.12, 0.74)] + arc(0.5, 0.74, 0.38, 0.2, 180, 360, 12) + [(0.9, 1.0)]
    p = [_ell(0.5, 0.3, 0.17, 0.2), _P(body), _P([(0.44, 0.48), (0.44, 0.58)], 1, weight=0.8), _P([(0.56, 0.48), (0.56, 0.58)], 1, weight=0.8)]
    head = arc(0.5, 0.3, 0.17, 0.2, 0, 360, 18)
    p += _hatch(head, 0.045, 60, inset=0.004, stage=2, weight=0.65)
    p += _hatch(body + [(0.1, 1.0)], 0.05, 60, inset=0.004, stage=2, weight=0.65)
    p += _hatch(body + [(0.1, 1.0)], 0.1, 120, inset=0.01, stage=2, color="dim", weight=0.5)
    p.append(_P([(0.38, 0.12), (0.5, 0.08), (0.62, 0.12)], 1, weight=0.7))
    return p


def ration_valve() -> list[Part]:
    """A tank feeding a pipe through a half-closed valve; only a thin trickle comes out: rationed effort."""
    p: list[Part] = []
    top = arc(0.17, 0.2, 0.15, 0.05, 0, 360, 14)
    p += [_P(top), _P([(0.02, 0.2), (0.02, 0.78)]), _P([(0.32, 0.2), (0.32, 0.78)]), _P(arc(0.17, 0.78, 0.15, 0.05, 0, 180, 10))]
    p += [_P(arc(0.17, 0.5, 0.14, 0.04, 0, 180, 8), 1, weight=0.6)]                       # liquid surface
    p += _hatch([(0.04, 0.5), (0.3, 0.5), (0.3, 0.77), (0.04, 0.77)], 0.035, 45, inset=0.0, stage=2, weight=0.6)
    p += _glyph("MODEL", 0.17, 0.34, 0.05, stage=1, color="chalk", weight=0.8, seed=2)
    p += [_P([(0.31, 0.6), (0.78, 0.6)]), _P([(0.31, 0.68), (0.78, 0.68)]), _P([(0.78, 0.6), (0.78, 0.84)]), _P([(0.86, 0.68), (0.86, 0.84)]),
          _P([(0.78, 0.84), (0.86, 0.84)])]
    p += [_rect(0.5, 0.56, 0.08, 0.16, stage=0, weight=0.9)]
    p.append(_P([(0.54, 0.56), (0.54, 0.46)], 0, weight=0.9))
    for k in range(2):                                  # valve wheel, turned about 30 degrees: half closed
        a = math.radians(30 + k * 90)
        p.append(_P([(0.54 - 0.1 * math.cos(a), 0.36 - 0.1 * math.sin(a)), (0.54 + 0.1 * math.cos(a), 0.36 + 0.1 * math.sin(a))], 1, weight=0.9))
    p.append(_ell(0.54, 0.36, 0.1, 0.1, stage=0, weight=1.0))
    p += _hatch([(0.31, 0.68), (0.78, 0.68), (0.78, 0.72), (0.31, 0.72)], 0.025, 45, inset=0.0, stage=2, weight=0.5)
    p.append(_P([(0.82, 0.86), (0.82, 0.95)], 3, "yellow", 1.4))
    p += [_ell(0.82, 0.975, 0.012, 0.018, stage=3, color="yellow", weight=0.9),
          _P([(0.74, 0.985), (0.76, 1.04), (0.88, 1.04), (0.9, 0.985)], 0, weight=0.9)]       # the cup that catches it
    return p


def frosted_pane() -> list[Part]:
    """A window of frosted glass: you can tell there is something behind it, not what."""
    outer = [(0.06, 0.05), (0.94, 0.05), (0.94, 0.95), (0.06, 0.95)]
    inner = [(0.1, 0.09), (0.9, 0.09), (0.9, 0.91), (0.1, 0.91)]
    p = [_P(outer + [(0.06, 0.044)], 0, weight=1.2), _P(inner + [(0.1, 0.084)], 0, weight=0.8)]
    for x, y in ((0.08, 0.07), (0.92, 0.07), (0.08, 0.93), (0.92, 0.93)):
        p.append(_ell(x, y, 0.012, 0.012, stage=1, weight=0.7))
    p += _hatch(inner, 0.055, 52, inset=0.012, stage=1, color="dim", weight=0.5)
    p += _hatch([(0.1, 0.55), (0.55, 0.91), (0.1, 0.91)], 0.12, 128, inset=0.01, stage=2, color="dim", weight=0.45)
    p += [_P([(0.7, 0.16), (0.8, 0.26)], 2, weight=0.7), _P([(0.76, 0.14), (0.86, 0.24)], 2, weight=0.7)]       # glint
    return p


def sealed_card() -> list[Part]:
    """A report card as it exists behind frosted glass: dashed, dim, its grade slot empty."""
    card = [(0.14, 0.06), (0.86, 0.06), (0.86, 0.94), (0.14, 0.94), (0.14, 0.06)]
    p = [_dashed1(card, 0.05, 0.032, 0, "dim", 0.9)]
    for i in range(4):
        p.append(_dashed1([(0.22, 0.2 + i * 0.1), (0.7 - (0.12 if i % 2 else 0), 0.2 + i * 0.1)], 0.045, 0.03, 1, "dim", 0.5))
    p.append(_dashed1(arc(0.5, 0.68, 0.24, 0.17, 0, 360, 22), 0.05, 0.035, 1, "dim", 0.8))     # the empty grade slot
    return p


def stamp_final() -> list[Part]:
    """A rubber stamp that has landed crooked: FINAL."""
    a = math.radians(-9)
    def rot(pts):
        return [(0.5 + (x - 0.5) * math.cos(a) - (y - 0.5) * math.sin(a), 0.5 + (x - 0.5) * math.sin(a) + (y - 0.5) * math.cos(a)) for x, y in pts]
    outer = [(0.04, 0.3), (0.96, 0.3), (0.96, 0.7), (0.04, 0.7), (0.04, 0.296)]
    inner = [(0.08, 0.34), (0.92, 0.34), (0.92, 0.66), (0.08, 0.66), (0.08, 0.336)]
    p = [_P(rot(outer), 0, "yellow", 1.5), _P(rot(inner), 0, "yellow", 0.8)]
    for g in _glyph("FINAL", 0.5, 0.5, 0.2, stage=0, color="yellow", weight=1.5, seed=5):
        p.append(Part(np.array(rot([tuple(q) for q in g.pts]), np.float32), 0, "yellow", 1.5))
    return p


# ------------------------------------------------------------------ registry
LIBRARY: dict[str, Callable[[], list[Part]]] = {
    "person": person, "crowd": crowd, "robot": robot, "brain": brain, "network": network, "server": server,
    "chip": chip, "cloud": cloud, "document": document, "papers": papers, "magnifier": magnifier, "eye": eye,
    "lock": lock, "key": key, "shield": shield, "warning": warning, "clock": clock, "hourglass": hourglass,
    "gear": gear, "funnel": funnel, "scales": scales, "coins": coins, "moneybag": moneybag, "bank": bank,
    "factory": factory, "skyscraper": skyscraper, "truck": truck, "chart_down": chart_down, "chart_up": chart_up,
    "bars": bars, "pie": pie, "lightbulb": lightbulb, "megaphone": megaphone, "chat": chat, "laptop": laptop,
    "phone": phone, "globe": globe, "pin": pin, "domino": domino, "bomb": bomb, "trap": trap, "web": web,
    "iceberg": iceberg, "rocket": rocket, "trophy": trophy, "briefcase": briefcase, "cart": cart, "flag": flag,
    "ladder": ladder, "door": door, "reportcard": reportcard, "check": check, "cross": cross,
    "question": question, "dollar": dollar, "crack": crack, "frown": frown,
    # scene-specific drawings (no keyword triggers: they are used only by explicit plans / compositions)
    "gradebook": gradebook, "orgchart": orgchart, "score_dial": score_dial, "danger_gauge": danger_gauge,
    "silhouette": silhouette, "ration_valve": ration_valve, "frosted_pane": frosted_pane,
    "sealed_card": sealed_card, "stamp_final": stamp_final,
}

# word -> illustration lexicon used by the local director (first match wins ties by weight)
KEYWORDS: dict[str, tuple[str, ...]] = {
    "reportcard": ("graded", "grade", "grading", "score", "scored", "rating", "rated", "rank", "ranked", "ranking", "judged", "assess"),
    "robot": ("ai", "chatbot", "chatgpt", "assistant", "model", "llm", "machine", "bot", "claude", "gpt", "openai", "anthropic"),
    "brain": ("neural", "brain", "learn", "learning", "intelligence", "trained", "training", "reinforcement", "preference", "mind"),
    "person": ("teacher", "teachers", "student", "students", "professor", "you", "user", "users", "customer", "customers", "worker", "employee", "founder", "ceo", "executive", "contractor", "contractors", "rater", "raters", "human", "people", "person"),
    "crowd": ("crowd", "public", "population", "audience", "everyone", "millions", "consumers", "voters", "society", "workforce"),
    "network": ("network", "connected", "system", "systems", "platform", "ecosystem", "web", "infrastructure", "architecture", "scaffold", "pipeline"),
    "server": ("server", "servers", "datacenter", "cloud", "compute", "inference", "gpu", "hosting", "database", "data"),
    "chip": ("chip", "chips", "processor", "hardware", "silicon", "semiconductor", "algorithm", "software", "code", "digital", "neural network"),
    "document": ("document", "memo", "contract", "report", "filing", "email", "letter", "record", "policy", "paper", "papers", "prompt", "prompts", "paper trail", "terms"),
    "papers": ("documents", "files", "records", "evidence", "leaked", "leak", "emails", "disclosures", "dataset", "archive"),
    "magnifier": ("investigate", "investigation", "inspect", "examine", "audit", "discover", "found", "uncover", "analysis", "search", "look", "detail"),
    "eye": ("watch", "watching", "watched", "monitor", "surveillance", "observe", "tracking", "track", "invisible", "silent", "silently", "see", "saw", "hidden", "secretly", "quietly", "private"),
    "lock": ("secret", "secrets", "locked", "encrypted", "private", "security", "restricted", "gate", "gated", "gates", "access", "block", "blocked", "refuse", "refusal"),
    "key": ("key", "unlock", "password", "credential", "credentials", "keys"),
    "shield": ("safe", "safety", "protect", "protection", "protects", "harmless", "defense", "guard", "compliance", "liability", "insurance"),
    "warning": ("danger", "dangerous", "risk", "risks", "threat", "warn", "warning", "scandal", "fraud", "breach", "crisis", "disaster", "violation", "illegal"),
    "clock": ("time", "years", "year", "decade", "months", "hours", "deadline", "history", "long", "minutes", "later", "five minutes"),
    "hourglass": ("waiting", "delay", "running out", "countdown", "patience", "eventually", "slowly"),
    "gear": ("mechanism", "process", "engine", "operate", "works", "work", "machinery", "optimization", "optimize", "optimizes", "tuned", "tuning", "workings"),
    "funnel": ("filter", "filtered", "filters", "funnel", "pipeline", "narrow", "narrowing", "select", "selected", "cheap", "gate"),
    "scales": ("compare", "weigh", "balance", "fair", "fairness", "justice", "court", "trial", "law", "legal", "judge", "tradeoff", "trade-off", "ration", "ration", "rations"),
    "coins": ("money", "cost", "costs", "price", "profit", "revenue", "paid", "pay", "budget", "billion", "million", "dollar", "dollars", "cash", "expensive", "bill", "invest", "investors"),
    "moneybag": ("wealth", "rich", "fortune", "payout", "bonus", "salary", "business", "sell", "selling", "sold", "monetize", "model", "buy"),
    "bank": ("bank", "banks", "finance", "financial", "loan", "credit", "debt", "bureau", "wall street", "lender", "fund"),
    "factory": ("factory", "manufacturing", "production", "produce", "plant", "industrial", "industry", "built", "assembly", "supply"),
    "skyscraper": ("company", "corporation", "corporate", "headquarters", "office", "firm", "enterprise", "organization", "companies", "giant"),
    "truck": ("shipping", "delivery", "logistics", "freight", "transport", "supply chain", "route"),
    "chart_down": ("collapse", "collapsed", "crash", "crashed", "fell", "fall", "decline", "dropped", "plunge", "losses", "loss", "failure", "failed", "bankrupt", "withdraws", "withdraw", "withdrawn", "shorter"),
    "chart_up": ("growth", "grew", "rose", "rise", "surge", "boom", "soared", "increase", "increased", "scale", "scaling", "deeper", "longer"),
    "bars": ("statistics", "percent", "numbers", "data points", "comparison", "survey", "metrics", "measure", "measured", "quantify"),
    "pie": ("share", "market", "fraction", "portion", "slice", "split", "allocate", "allocation", "allocates", "allocation"),
    "lightbulb": ("idea", "ideas", "realize", "realization", "insight", "invention", "invented", "innovation", "clever", "genius", "understand"),
    "megaphone": ("announce", "announced", "announcement", "marketing", "advertising", "promise", "promised", "claim", "claims", "said", "told", "broadcast", "campaign", "pr"),
    "chat": ("conversation", "chat", "message", "messages", "ask", "asked", "asks", "question", "questions", "reply", "respond", "response", "responses", "answer", "answers", "typed", "type", "prompt", "follow-up", "correction"),
    "laptop": ("email", "computer", "online", "internet", "website", "app", "typing", "screen", "draft", "login", "opened"),
    "phone": ("phone", "mobile", "smartphone", "text", "notification", "notifications"),
    "globe": ("global", "world", "worldwide", "international", "country", "countries", "everywhere", "planet", "nation", "deployed", "across"),
    "pin": ("location", "city", "headquartered", "place", "town", "where"),
    "domino": ("chain", "chain reaction", "cascade", "domino", "collapse", "ripple", "spread", "contagion", "snowball", "effect", "result", "consequence"),
    "bomb": ("explosion", "explode", "bomb", "blast", "detonate", "ticking", "fuse", "time bomb", "explosive", "bubble", "burst"),
    "trap": ("trap", "trapped", "bait", "catch", "snare", "scam", "con", "deceive", "deception", "lure", "fooled", "manipulate", "manipulation"),
    "web": ("tangled", "web of", "entangled", "network of lies", "conspiracy", "connections", "interconnected", "intertwined", "spider"),
    "iceberg": ("iceberg", "tip", "surface", "beneath", "underneath", "below", "unseen", "never see", "hidden", "tip of the", "behind the scenes", "invisible", "cannot appeal", "prove"),
    "rocket": ("launch", "launched", "startup", "start-up", "takeoff", "skyrocket", "ambition", "moonshot", "space", "rapid", "fast", "faster"),
    "trophy": ("win", "won", "winner", "winning", "best", "champion", "success", "reward", "rewards", "prize", "victory", "top", "high"),
    "briefcase": ("boss", "bosses", "manager", "employer", "job", "career", "employment", "hire", "hired", "hiring", "work", "professional", "resume", "contractors", "contract work", "consultant"),
    "cart": ("shop", "shopping", "retail", "store", "purchase", "ecommerce", "amazon", "consumer", "checkout"),
    "flag": ("flag", "flagged", "milestone", "country's", "banner", "mission", "goal", "target", "targets", "label", "labeled", "classify", "classifies", "classification", "tag", "tagged"),
    "ladder": ("climb", "climbing", "promotion", "promoted", "tier", "tiers", "levels", "hierarchy", "rank", "ladder", "status", "step"),
    "door": ("door", "exit", "enter", "entrance", "opportunity", "closed", "shut", "leave", "left", "walk out", "quit", "open"),
    "check": ("approve", "approved", "correct", "verified", "accept", "accepted", "compliant", "passed", "pass", "pleases", "satisfied", "satisfaction", "helpful", "honest", "yes", "agree"),
    "cross": ("reject", "rejected", "denied", "deny", "banned", "wrong", "fail", "refused", "refuses", "rule", "forbid", "cannot", "can't", "no", "bypass"),
    "question": ("mystery", "unknown", "why", "how", "puzzle", "unanswered", "what if", "wonder", "unclear", "nobody knows", "who", "what"),
    "dollar": ("dollars", "$", "paying", "revenue", "billing", "invoice", "price tag", "fee", "fees", "subscription", "paid"),
    "crack": ("broken", "broke", "break", "cracks", "cracked", "fracture", "flaw", "flawed", "bug", "split", "shatter", "collapse", "unravel", "unraveled", "failure", "fault"),
    "frown": ("dislike", "unhappy", "dissatisfied", "frustrated", "complaint", "complain", "angry", "hated", "negative", "worse", "poor", "bad", "rude", "risk"),
}


def motifs_for_text(text: str, limit: int = 3, avoid: Iterable[str] = ()) -> list[str]:
    """Rank library illustrations by how well their keywords match ``text``."""
    s = " " + re.sub(r"[^a-z0-9$'’ -]+", " ", text.casefold()) + " "
    tokens = set(s.split())
    scores: dict[str, float] = {}
    for name, words in KEYWORDS.items():
        score = 0.0
        for w in words:
            if " " in w or "-" in w:
                if f" {w} " in s:
                    score += 2.0
            elif w in tokens:
                score += 1.0
        if score:
            scores[name] = score + 0.01 * (len(words) % 7)
    ranked = [n for n, _ in sorted(scores.items(), key=lambda kv: -kv[1]) if n in LIBRARY]
    avoid = set(avoid)
    fresh = [n for n in ranked if n not in avoid]
    ordered = fresh + [n for n in ranked if n in avoid]
    return ordered[:limit]


# ------------------------------------------------------------ SVG path import
_SVG_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|-?\d*\.?\d+(?:e-?\d+)?")


def svg_path_to_polylines(d: str, curve_steps: int = 14) -> list[list[Pt]]:
    """Parse SVG path data (M L H V C S Q T Z, absolute+relative) into polylines.

    Arcs (A) are not supported; flatten them to curves in your vector editor.
    """
    tokens = _SVG_TOKEN.findall(d)
    i = 0
    cmd = ""
    cur = (0.0, 0.0)
    start = (0.0, 0.0)
    last_ctrl: Pt | None = None
    lines: list[list[Pt]] = []
    line: list[Pt] = []

    def num() -> float:
        nonlocal i
        v = float(tokens[i])
        i += 1
        return v

    def flush() -> None:
        nonlocal line
        if len(line) >= 2:
            lines.append(line)
        line = []

    while i < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                if line:
                    line.append(start)
                flush()
                cur = start
                continue
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            flush()
            x, y = num(), num()
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            start = cur
            line = [cur]
            cmd = "l" if rel else "L"
            last_ctrl = None
        elif c == "L":
            x, y = num(), num()
            cur = (cur[0] + x, cur[1] + y) if rel else (x, y)
            line.append(cur)
            last_ctrl = None
        elif c == "H":
            x = num()
            cur = (cur[0] + x, cur[1]) if rel else (x, cur[1])
            line.append(cur)
            last_ctrl = None
        elif c == "V":
            y = num()
            cur = (cur[0], cur[1] + y) if rel else (cur[0], y)
            line.append(cur)
            last_ctrl = None
        elif c in "CS":
            if c == "C":
                x1, y1 = num(), num()
                p1 = (cur[0] + x1, cur[1] + y1) if rel else (x1, y1)
            else:
                p1 = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_ctrl else cur
            x2, y2, x, y = num(), num(), num(), num()
            p2 = (cur[0] + x2, cur[1] + y2) if rel else (x2, y2)
            p3 = (cur[0] + x, cur[1] + y) if rel else (x, y)
            line += _bez(cur, p1, p2, p3, curve_steps)[1:]
            cur, last_ctrl = p3, p2
        elif c in "QT":
            if c == "Q":
                x1, y1 = num(), num()
                q = (cur[0] + x1, cur[1] + y1) if rel else (x1, y1)
            else:
                q = (2 * cur[0] - last_ctrl[0], 2 * cur[1] - last_ctrl[1]) if last_ctrl else cur
            x, y = num(), num()
            p3 = (cur[0] + x, cur[1] + y) if rel else (x, y)
            p1 = (cur[0] + 2 / 3 * (q[0] - cur[0]), cur[1] + 2 / 3 * (q[1] - cur[1]))
            p2 = (p3[0] + 2 / 3 * (q[0] - p3[0]), p3[1] + 2 / 3 * (q[1] - p3[1]))
            line += _bez(cur, p1, p2, p3, curve_steps)[1:]
            cur, last_ctrl = p3, q
        else:  # unsupported command (A): skip its numbers defensively
            while i < len(tokens) and not re.fullmatch(r"[A-Za-z]", tokens[i]):
                i += 1
    flush()
    return lines


def register_svg(name: str, paths: Sequence[str | tuple[str, int, str]], *, keywords: Sequence[str] = ()) -> None:
    """Add an illustration from SVG path data (``d`` strings).

    ``paths`` items are either a ``d`` string or ``(d, stage, colour)``.  The
    drawing is fitted into the unit square, preserving its aspect ratio.
    """
    parsed: list[tuple[list[Pt], int, str]] = []
    for item in paths:
        d, stage, color = (item, 0, "chalk") if isinstance(item, str) else item
        for pl in svg_path_to_polylines(d):
            parsed.append((pl, stage, color))
    if not parsed:
        raise ValueError("SVG illustration has no drawable paths")
    allp = np.concatenate([np.asarray(pl, np.float32) for pl, _s, _c in parsed])
    lo, hi = allp.min(axis=0), allp.max(axis=0)
    scale = 1.0 / max(1e-6, float(max(hi - lo)))
    off = (1.0 - (hi - lo) * scale) / 2.0

    def build() -> list[Part]:
        return [Part(((np.asarray(pl, np.float32) - lo) * scale + off).astype(np.float32), st, col, 1.0) for pl, st, col in parsed]

    LIBRARY[name] = build
    if keywords:
        KEYWORDS[name] = tuple(keywords)


# --------------------------------------------------------------------- placing
def build_illustration(name: str, cx: float, cy: float, size: float, *, seed: int = 0, rotate: float | None = None,
                       detail: int = 3, flip: bool = False) -> list[Part]:
    """Place an illustration on the board (world px) with hand-drawn imperfection."""
    fn = LIBRARY.get(name)
    if fn is None:
        raise KeyError(f"unknown illustration {name!r}")
    rng = random.Random(seed * 131 + sum(map(ord, name)))
    rot = math.radians(rng.uniform(-3.5, 3.5) if rotate is None else rotate)
    out: list[Part] = []
    # detail 3 = everything; 2 = no hatching; 1 = outline + the accent that makes the point
    keep = {3: {0, 1, 2, 3}, 2: {0, 1, 3}, 1: {0, 3}}.get(max(1, min(3, int(detail))), {0, 1, 2, 3})
    for part in fn():
        if part.stage not in keep:
            continue
        pts = part.pts.copy()
        if flip:
            pts[:, 0] = 1.0 - pts[:, 0]
        x = (pts[:, 0] - 0.5) * size
        y = (pts[:, 1] - 0.5) * size
        xr = x * math.cos(rot) - y * math.sin(rot) + cx
        yr = x * math.sin(rot) + y * math.cos(rot) + cy
        arr = densify(np.stack([xr, yr], axis=1), 2.5)
        if len(arr) >= 4:
            # smooth low-frequency wobble: a hand never draws a perfect curve
            n = len(arr)
            phase = rng.uniform(0, 6.28)
            freq = rng.uniform(0.04, 0.09)
            amp = max(0.6, size * 0.0042)
            t = np.arange(n, dtype=np.float32)
            arr[:, 0] += amp * np.sin(t * freq + phase)
            arr[:, 1] += amp * np.cos(t * freq * 1.3 + phase * 0.7)
            # lines overshoot a little where a hand would not stop exactly
            d = arr[-1] - arr[-3]
            norm = float(np.hypot(*d))
            if norm > 1e-3 and rng.random() < 0.5:
                arr = np.vstack([arr, arr[-1] + d / norm * rng.uniform(1.5, 4.5)])
        out.append(Part(arr.astype(np.float32), part.stage, part.color, part.weight,
                        (part.dash[0] * size, part.dash[1] * size) if part.dash else None))
    return out
