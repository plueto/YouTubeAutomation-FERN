"""Turn a chalkboard scene plan into a timed :class:`~moduli.chalk_engine.Program`.

The plan (see ``chalkboard_director``) says *what* each narrated beat is about:
a layout, illustrations, short chalk labels and the spoken words that should
trigger them.  This module decides *how and when* it is drawn:

* every element is anchored to the moment its word is spoken (word-level
  timestamps when available, a clearly-labelled estimate otherwise)
* one pen: elements are drawn in sequence, with natural breathing room
* if a beat is too short for its drawing, detail is reduced before the pen is
  hurried (hatching and secondary details go first)
* the camera pushes in, closes up on the focal drawing and pans across the
  board between connected ideas; unrelated beats are wiped with an eraser
* impact / whoosh sound cues are placed on the reveals and the moves
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from moduli.chalk_engine import (
    PAGE_H, PAGE_W, CamKey, Erase, Program, Stroke, densify, path_length,
)
from moduli.chalk_illustrations import LIBRARY, build_illustration
from moduli.chalk_lettering import letter_strokes, text_width

LAYOUTS = ("hero", "duo", "chain", "versus", "iceberg", "bignumber", "network",
           "timeline", "evidence", "funnel", "statement", "question")
MARGIN_X = 130


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class Path:
    pts: np.ndarray
    color: str = "chalk"
    width: float = 6.0
    stage: int = 0
    dryness: float = 0.55
    speed: float = 800.0     # px/s along the path
    min_dur: float = 0.04


def path_time(p: "Path") -> float:
    """Seconds the pen needs for one stroke (shared by scheduling and emission)."""
    length = path_length(p.pts)
    if p.stage >= 2 and len(p.pts) < 60:       # hatching & small details are flicked in
        return max(0.035, length / (p.speed * 2.4))
    return max(p.min_dur, length / p.speed)


@dataclass(eq=False)
class Element:
    kind: str
    paths: list[Path]
    anchor: float | None = None      # absolute seconds at which this should *start*
    lead: float = 0.30               # start this long before the anchor word
    gap: float = 0.0                 # extra pause before starting
    impact: bool = False
    tag: str = ""
    focus: tuple[float, float] | None = None
    importance: int = 1              # 0 = decoration (dropped first when time is short)

    def nominal_duration(self) -> float:
        total = 0.0
        last_stage = 0
        for p in self.paths:
            total += path_time(p) + 0.012
            if p.stage != last_stage:
                total += 0.08
                last_stage = p.stage
        return total + self.gap


def _wobbly(p0, p1, rng: random.Random, sag: float = 0.012, overshoot: float = 6.0) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy) or 1.0
    ux, uy = dx / d, dy / d
    nx, ny = -uy, ux
    ex, ey = x1 + ux * rng.uniform(0, overshoot), y1 + uy * rng.uniform(0, overshoot)
    n = max(4, int(d / 40))
    bow = rng.uniform(-1, 1) * sag * d
    pts = []
    for i in range(n + 1):
        t = i / n
        off = bow * math.sin(math.pi * t) + rng.uniform(-0.6, 0.6)
        pts.append((x0 + (ex - x0) * t + nx * off, y0 + (ey - y0) * t + ny * off))
    return densify(pts, 2.5)


def E_text(text: str, x: float, y: float, cap: float, *, align: str = "center", color: str = "chalk",
           anchor: float | None = None, max_width: float | None = None, seed: int = 0, impact: bool = False,
           width: float | None = None, importance: int = 2, gap: float = 0.0) -> Element:
    strokes, bb = letter_strokes(text, x, y, cap, align=align, max_width=max_width, seed=seed)
    w = width if width is not None else max(3.6, min(8.5, cap * 0.088))
    paths = [Path(densify(s["pts"], 2.2), color, w, 0, 0.5, 700.0 + cap * 48.0, 0.03) for s in strokes]
    return Element("text", paths, anchor, 0.30, gap, impact, f"text:{text[:24]}",
                   focus=((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2), importance=importance)


def E_illus(name: str, cx: float, cy: float, size: float, *, anchor: float | None = None, seed: int = 0,
            detail: int = 3, flip: bool = False, rotate: float | None = None, impact: bool = False,
            importance: int = 2, lead: float = 0.65) -> Element:
    parts = build_illustration(name, cx, cy, size, seed=seed, detail=detail, rotate=rotate, flip=flip)
    base = max(3.6, min(9.0, size * 0.0158))
    sp = 1800.0 + size * 4.2
    paths = [Path(p.pts, p.color, max(2.6, base * p.weight), p.stage, 0.5 + 0.1 * (p.stage >= 2), sp) for p in parts]
    return Element("illus", paths, anchor, lead, 0.0, impact, f"illus:{name}", focus=(cx, cy), importance=importance)


def E_line(p0, p1, *, color="chalk", width=5.0, anchor=None, seed=0, importance=1, speed=1500.0) -> Element:
    rng = random.Random(seed)
    return Element("line", [Path(_wobbly(p0, p1, rng), color, width, 0, 0.5, speed)], anchor, 0.2, 0.0, False,
                   "line", focus=((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2), importance=importance)


def E_arrow(p0, p1, *, color="teal", width=5.4, anchor=None, seed=0, curve: float = 0.12, importance=1) -> Element:
    rng = random.Random(seed)
    x0, y0 = p0
    x1, y1 = p1
    d = math.hypot(x1 - x0, y1 - y0) or 1.0
    nx, ny = -(y1 - y0) / d, (x1 - x0) / d
    c = rng.choice((-1, 1)) * curve * d
    ctrl = ((x0 + x1) / 2 + nx * c, (y0 + y1) / 2 + ny * c)
    pts = []
    for i in range(21):
        t = i / 20
        pts.append(((1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl[0] + t * t * x1,
                    (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl[1] + t * t * y1))
    shaft = densify(pts, 2.5)
    ang = math.atan2(shaft[-1][1] - shaft[-4][1], shaft[-1][0] - shaft[-4][0])
    hs = min(46.0, 18 + d * 0.06)
    tip = shaft[-1]
    h1 = (tip[0] - hs * math.cos(ang - 0.5), tip[1] - hs * math.sin(ang - 0.5))
    h2 = (tip[0] - hs * math.cos(ang + 0.5), tip[1] - hs * math.sin(ang + 0.5))
    return Element("arrow", [Path(shaft, color, width, 0, 0.5, 1500.0),
                             Path(densify([h1, tuple(tip)], 2.5), color, width, 0, 0.5, 900.0),
                             Path(densify([h2, tuple(tip)], 2.5), color, width, 0, 0.5, 900.0)],
                   anchor, 0.2, 0.0, False, "arrow", focus=((x0 + x1) / 2, (y0 + y1) / 2), importance=importance)


def E_underline(x0, x1, y, *, color="yellow", width=5.6, anchor=None, seed=0, double=True, impact=True, importance=1) -> Element:
    rng = random.Random(seed)
    paths = [Path(_wobbly((x0, y), (x1, y + rng.uniform(-6, 6)), rng, 0.01, 10), color, width, 0, 0.5, 2300.0)]
    if double:
        paths.append(Path(_wobbly((x0 + 24, y + 15), (x1 - 12, y + 12 + rng.uniform(-4, 4)), rng, 0.01, 8), color, width * 0.85, 0, 0.55, 2400.0))
    return Element("underline", paths, anchor, 0.1, 0.0, impact, "underline", focus=((x0 + x1) / 2, y), importance=importance)


def E_loop(cx, cy, rx, ry, *, color="yellow", width=5.6, anchor=None, seed=0, impact=True, importance=1) -> Element:
    """A hand-drawn loop that circles something and overshoots its start."""
    rng = random.Random(seed)
    a0 = rng.uniform(-150, -100)
    pts = []
    n = 46
    for i in range(n + 1):
        u = i / n
        a = math.radians(a0 + 392 * u)
        k = 1.0 - 0.055 * u + 0.012 * math.sin(u * 9)
        pts.append((cx + rx * k * math.cos(a), cy + ry * k * math.sin(a)))
    return Element("loop", [Path(densify(pts, 3.0), color, width, 0, 0.5, 2600.0)], anchor, 0.12, 0.0, impact,
                   "loop", focus=(cx, cy), importance=importance)


def E_scribble_burst(cx, cy, r, *, color="yellow", anchor=None, seed=0) -> Element:
    rng = random.Random(seed)
    paths = []
    for i in range(9):
        a = rng.uniform(0, 6.28)
        r0, r1 = r * rng.uniform(0.55, 0.7), r * rng.uniform(0.95, 1.25)
        paths.append(Path(densify([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], 2.5), color, 4.6, 0, 0.5, 1700.0))
    return Element("burst", paths, anchor, 0.1, 0.0, True, "burst", focus=(cx, cy), importance=0)


# ---------------------------------------------------------------------------
# layouts. Each returns (elements, focus point, focal element index)
# ---------------------------------------------------------------------------
@dataclass
class SceneSpec:
    index: int
    t0: float
    t1: float
    layout: str
    title: str
    motifs: list[str]
    labels: list[str]
    anchors: dict[str, float]          # key -> absolute seconds (see _anchor)
    number: str = ""
    callout: str = ""
    detail: int = 3
    mirror: bool = False
    seed: int = 0
    words: list[tuple[str, float, float]] = field(default_factory=list)


def _fit_cap(text: str, preferred: float, max_w: float, min_cap: float = 30.0) -> float:
    w = text_width(text.upper(), preferred)
    if w <= max_w:
        return preferred
    return max(min_cap, preferred * max_w / max(1.0, w))


def _label(spec: SceneSpec, i: int) -> str | None:
    return spec.labels[i] if i < len(spec.labels) and spec.labels[i] else None


def _motif(spec: SceneSpec, i: int, default: str = "document") -> str:
    if i < len(spec.motifs):
        return spec.motifs[i]
    return spec.motifs[-1] if spec.motifs else default


def _an(spec: SceneSpec, key: str) -> float | None:
    return spec.anchors.get(key)


def _caption(spec: SceneSpec, i: int, x: float, y: float, cap: float, max_w: float, *, color="chalk", align="center", impact=False, importance=1) -> Element | None:
    text = _label(spec, i)
    if not text:
        return None
    c = _fit_cap(text, cap, max_w * 1.9 if " " in text else max_w)
    return E_text(text, x, y, c, align=align, color=color, anchor=_an(spec, f"label{i}"), max_width=max_w,
                  seed=spec.seed * 5 + i, impact=impact, importance=importance)


def L_hero(spec: SceneSpec):
    s = spec
    m = -1 if s.mirror else 1
    cx = 560 if not s.mirror else PAGE_W - 560
    ox = PAGE_W - cx
    els: list[Element] = []
    hero = E_illus(_motif(s, 0), cx, 585, 600, anchor=_an(s, "motif0"), seed=s.seed, detail=s.detail, impact=True, importance=3)
    if s.title:
        cap = _fit_cap(s.title, 78, 1000)
        tx = MARGIN_X if not s.mirror else PAGE_W - MARGIN_X
        els.append(E_text(s.title, tx, 190, cap, align="left" if not s.mirror else "right", color="chalk",
                          anchor=_an(s, "title"), seed=s.seed, importance=3))
    els.append(hero)
    l0 = _caption(s, 0, ox, 430, 64, 640, color="chalk", importance=3)
    if l0:
        els.append(l0)
        els.append(E_arrow((ox - 330 * m, 470), (cx + 330 * m, 470), color="teal", anchor=_an(s, "label0"), seed=s.seed + 1))
    l1 = _caption(s, 1, ox, 700, 58, 640, color="yellow" if s.callout else "chalk", importance=2, impact=True)
    if l1:
        els.append(l1)
        bbox_w = text_width(_label(s, 1).upper(), _fit_cap(_label(s, 1), 58, 640 * 1.9))
        if s.callout or True:
            els.append(E_underline(ox - min(330, bbox_w / 2), ox + min(330, bbox_w / 2), 770, anchor=_an(s, "label1"), seed=s.seed + 2, impact=False, importance=0))
    if len(s.motifs) > 1:
        els.append(E_illus(s.motifs[1], ox + 10, 905, 220, anchor=_an(s, "motif1"), seed=s.seed + 3, detail=min(2, s.detail), importance=1))
    return els, (cx, 585), els.index(hero)


def L_duo(spec: SceneSpec):
    s = spec
    els: list[Element] = []
    a = E_illus(_motif(s, 0), 470, 500, 430, anchor=_an(s, "motif0"), seed=s.seed, detail=s.detail, importance=3)
    b = E_illus(_motif(s, 1, "question"), PAGE_W - 470, 500, 430, anchor=_an(s, "motif1"), seed=s.seed + 1, detail=s.detail, impact=True, importance=3)
    els.append(a)
    ca = _caption(s, 0, 470, 850, 56, 560)
    if ca:
        els.append(ca)
    verb = _label(s, 2)
    els.append(E_arrow((720, 520), (PAGE_W - 720, 520), color="teal", anchor=_an(s, "arrow"), seed=s.seed + 2, curve=0.1))
    if verb:
        els.append(E_text(verb, PAGE_W / 2, 440, _fit_cap(verb, 54, 440), color="yellow", anchor=_an(s, "label2"), seed=s.seed + 4, max_width=460, importance=1))
    els.append(b)
    cb = _caption(s, 1, PAGE_W - 470, 850, 56, 560, color="yellow", impact=True)
    if cb:
        els.append(cb)
    if s.title:
        els.insert(0, E_text(s.title, PAGE_W / 2, 150, _fit_cap(s.title, 66, 1300), anchor=_an(s, "title"), seed=s.seed, importance=2))
    return els, (PAGE_W / 2, 520), els.index(b)


def L_chain(spec: SceneSpec):
    s = spec
    n = max(3, min(4, len(s.motifs))) if len(s.motifs) >= 3 else 3
    xs = [PAGE_W / 2 + (i - (n - 1) / 2) * (PAGE_W - 2 * 300) / max(1, n - 1) for i in range(n)]
    els: list[Element] = []
    if s.title:
        els.append(E_text(s.title, PAGE_W / 2, 175, _fit_cap(s.title, 66, 1300), anchor=_an(s, "title"), seed=s.seed, importance=2))
    focal = 0
    for i, x in enumerate(xs):
        e = E_illus(_motif(s, i, "document"), x, 470, 290, anchor=_an(s, f"motif{i}"), seed=s.seed + i, detail=min(2, s.detail), impact=(i == n - 1), importance=3 if i == n - 1 else 2)
        if i > 0:
            els.append(E_arrow((xs[i - 1] + 175, 470), (x - 175, 470), anchor=_an(s, f"motif{i}"), seed=s.seed + 10 + i, curve=0.06))
        els.append(e)
        if i == n - 1:
            focal = len(els) - 1
        c = _caption(s, i, x, 760, 48, 420, color="yellow" if i == n - 1 else "chalk", importance=2)
        if c:
            els.append(c)
    return els, (PAGE_W / 2, 480), focal


def L_versus(spec: SceneSpec):
    s = spec
    els: list[Element] = [E_line((PAGE_W / 2, 170), (PAGE_W / 2, 960), color="dim", width=5, anchor=_an(s, "motif0"), seed=s.seed, importance=1, speed=1100)]
    els.append(E_text("VS", PAGE_W / 2, 150, 54, color="yellow", anchor=_an(s, "motif1"), seed=s.seed + 1, importance=1, impact=True))
    a = E_illus(_motif(s, 0), 470, 500, 420, anchor=_an(s, "motif0"), seed=s.seed + 2, detail=s.detail, importance=3)
    b = E_illus(_motif(s, 1, "cross"), PAGE_W - 470, 500, 420, anchor=_an(s, "motif1"), seed=s.seed + 3, detail=s.detail, importance=3, impact=True)
    els += [a]
    c0 = _caption(s, 0, 470, 850, 54, 640)
    if c0:
        els.append(c0)
    els += [b]
    c1 = _caption(s, 1, PAGE_W - 470, 850, 54, 640, color="yellow")
    if c1:
        els.append(c1)
    return els, (PAGE_W / 2, 520), els.index(b)


def L_iceberg(spec: SceneSpec):
    s = spec
    els: list[Element] = []
    ice = E_illus("iceberg", 880 if not s.mirror else PAGE_W - 880, 560, 880, anchor=_an(s, "motif0"), seed=s.seed, detail=s.detail, impact=True, importance=3)
    els.append(ice)
    top = _label(s, 0) or "WHAT YOU SEE"
    bot = _label(s, 1) or "WHAT YOU DON'T"
    x_text = 1560 if not s.mirror else 360
    els.append(E_text(top, x_text, 240, _fit_cap(top, 54, 420), color="chalk", anchor=_an(s, "label0"), seed=s.seed + 1, max_width=440, importance=2))
    els.append(E_arrow((x_text - 150 if not s.mirror else x_text + 150, 270), (1060 if not s.mirror else PAGE_W - 1060, 270), color="teal", anchor=_an(s, "label0"), seed=s.seed + 2))
    els.append(E_text(bot, x_text, 760, _fit_cap(bot, 56, 440), color="yellow", anchor=_an(s, "label1"), seed=s.seed + 3, max_width=440, importance=3, impact=True))
    els.append(E_arrow((x_text - 150 if not s.mirror else x_text + 150, 800), (1020 if not s.mirror else PAGE_W - 1020, 760), color="yellow", anchor=_an(s, "label1"), seed=s.seed + 4))
    if len(s.motifs) > 1 and s.motifs[1] != "iceberg":
        els.append(E_illus(s.motifs[1], x_text, 520, 200, anchor=_an(s, "motif1"), seed=s.seed + 5, detail=2, importance=1))
    return els, (880 if not s.mirror else PAGE_W - 880, 700), 0


def L_bignumber(spec: SceneSpec):
    s = spec
    num = (s.number or (s.labels[0] if s.labels else "?")).upper()
    cap = _fit_cap(num, 300, 1500, 90)
    els: list[Element] = []
    pre = _label(s, 1)
    if pre:
        els.append(E_text(pre, PAGE_W / 2, 250, _fit_cap(pre, 62, 1300), color="dim", anchor=_an(s, "title"), seed=s.seed, importance=1))
    n = E_text(num, PAGE_W / 2, 700, cap, color="chalk", anchor=_an(s, "number"), seed=s.seed + 1, impact=True, importance=3, width=max(7.0, cap * 0.075))
    els.append(n)
    w = text_width(num, cap)
    els.append(E_underline(PAGE_W / 2 - w / 2, PAGE_W / 2 + w / 2, 700 + cap * 0.16, anchor=_an(s, "number"), seed=s.seed + 2, impact=True))
    post = _label(s, 2) or _label(s, 0)
    if post and post.upper() != num:
        els.append(E_text(post, PAGE_W / 2, 930, _fit_cap(post, 60, 1400), color="teal", anchor=_an(s, "label0"), seed=s.seed + 3, importance=2))
    els.append(E_illus(_motif(s, 0, "clock"), 1700, 210, 230, anchor=_an(s, "motif0"), seed=s.seed + 4, detail=2, importance=1))
    return els, (PAGE_W / 2, 560), els.index(n)


def L_network(spec: SceneSpec):
    s = spec
    els: list[Element] = []
    c = E_illus(_motif(s, 0, "network"), PAGE_W / 2, 540, 380, anchor=_an(s, "motif0"), seed=s.seed, detail=s.detail, importance=3, impact=True)
    sats = [(430, 400), (PAGE_W - 430, 400), (430, 760), (PAGE_W - 430, 760)]
    n_sat = max(2, min(4, len(s.motifs) - 1)) if len(s.motifs) > 1 else 3
    els.append(c)
    for i in range(n_sat):
        x, y = sats[i]
        name = _motif(s, i + 1, "document")
        ang = math.atan2(y - 540, x - PAGE_W / 2)
        p0 = (PAGE_W / 2 + math.cos(ang) * 230, 540 + math.sin(ang) * 190)
        p1 = (x - math.cos(ang) * 150, y - math.sin(ang) * 130)
        els.append(E_line(p0, p1, color="teal", width=4.6, anchor=_an(s, f"motif{i + 1}"), seed=s.seed + 20 + i, speed=1000))
        els.append(E_illus(name, x, y, 200, anchor=_an(s, f"motif{i + 1}"), seed=s.seed + 30 + i, detail=2, importance=2))
        cap = _caption(s, i, x, y + 175 if y > 540 else y - 130, 40, 380, color="yellow" if i == 0 else "chalk", importance=1)
        if cap:
            els.append(cap)
    if s.title:
        els.insert(0, E_text(s.title, PAGE_W / 2, 135, _fit_cap(s.title, 60, 900), anchor=_an(s, "title"), seed=s.seed, importance=2))
    return els, (PAGE_W / 2, 540), 0


def L_timeline(spec: SceneSpec):
    s = spec
    n = max(3, min(4, len(s.motifs) if len(s.motifs) >= 3 else 3))
    xs = [260 + i * (PAGE_W - 520) / (n - 1) for i in range(n)]
    y = 650
    els: list[Element] = [E_arrow((150, y), (PAGE_W - 120, y), color="chalk", width=6, anchor=_an(s, "motif0"), seed=s.seed, curve=0.01)]
    focal = 0
    for i, x in enumerate(xs):
        els.append(E_line((x, y - 24), (x, y + 24), color="yellow" if i == n - 1 else "chalk", width=6, anchor=_an(s, f"motif{i}"), seed=s.seed + i, speed=500))
        e = E_illus(_motif(s, i, "clock"), x, 410, 260, anchor=_an(s, f"motif{i}"), seed=s.seed + 3 + i, detail=2 if i < n - 1 else s.detail, importance=2, impact=(i == n - 1))
        els.append(e)
        if i == n - 1:
            focal = len(els) - 1
        c = _caption(s, i, x, 790, 46, 400, color="yellow" if i == n - 1 else "chalk")
        if c:
            els.append(c)
    if s.title:
        els.insert(0, E_text(s.title, PAGE_W / 2, 160, _fit_cap(s.title, 62, 1300), anchor=_an(s, "title"), seed=s.seed, importance=2))
    return els, (PAGE_W / 2, 500), focal


def L_evidence(spec: SceneSpec):
    s = spec
    names = [_motif(s, 0, "document"), _motif(s, 1, "papers"), _motif(s, 2, "document")]
    if names[0] == names[2]:
        names[2] = "papers" if names[0] != "papers" else "document"
    spots = [(500, 470, 430, -5), (990, 520, 400, 4), (1440, 450, 400, -3)]
    els: list[Element] = []
    pins = []
    for i, ((x, y, sz, rot), nm) in enumerate(zip(spots, names)):
        els.append(E_illus(nm, x, y, sz, anchor=_an(s, f"motif{i}"), seed=s.seed + i, detail=s.detail, rotate=rot, importance=3, impact=(i == 2)))
        pins.append((x, y - sz * 0.36))
    for i in range(2):
        els.append(E_line(pins[i], pins[i + 1], color="yellow", width=3.6, anchor=_an(s, f"motif{i + 1}"), seed=s.seed + 9 + i, speed=1100))
    els.append(E_illus("magnifier", 1130, 830, 250, anchor=_an(s, "motif3"), seed=s.seed + 7, detail=2, importance=1))
    for i in range(3):
        c = _caption(s, i, spots[i][0], 800 if i != 1 else 870, 42, 440, color="yellow" if i == 2 else "chalk")
        if c:
            els.append(c)
    if s.title:
        els.insert(0, E_text(s.title, PAGE_W / 2, 140, _fit_cap(s.title, 62, 1100), anchor=_an(s, "title"), seed=s.seed, importance=2))
    return els, (PAGE_W / 2, 500), len(els) - 1 if False else 0


def L_funnel(spec: SceneSpec):
    s = spec
    els: list[Element] = []
    xs = [560, 960, 1360]
    for i, x in enumerate(xs):
        els.append(E_illus(_motif(s, i, "document"), x, 230, 230, anchor=_an(s, f"motif{i}"), seed=s.seed + i, detail=2, importance=2))
        els.append(E_arrow((x, 370), (960 + (x - 960) * 0.28, 470), color="dim", width=4.4, anchor=_an(s, f"motif{i}"), seed=s.seed + 5 + i, curve=0.05))
    f = E_illus("funnel", 960, 640, 440, anchor=_an(s, "label0"), seed=s.seed + 9, detail=s.detail, importance=3, impact=True)
    els.append(f)
    out_name = s.motifs[-1] if len(s.motifs) > 3 else "check"
    els.append(E_arrow((1200, 780), (1480, 860), color="yellow", anchor=_an(s, "label1"), seed=s.seed + 12))
    els.append(E_illus(out_name, 1640, 860, 280, anchor=_an(s, "label1"), seed=s.seed + 13, detail=2, importance=2, impact=True))
    c0 = _caption(s, 0, 560, 920, 46, 520)
    if c0:
        els.append(c0)
    c1 = _caption(s, 1, 1640, 1010, 44, 420, color="yellow")
    if c1:
        els.append(c1)
    return els, (960, 620), els.index(f)


def L_statement(spec: SceneSpec):
    s = spec
    text = (s.title or s.callout or _label(s, 0) or "").upper()
    cap = _fit_cap(text, 170, 1500, 90)
    lines_est = max(1, math.ceil(text_width(text, cap) / 1500))
    els: list[Element] = []
    t = E_text(text, PAGE_W / 2, 500 if lines_est == 1 else 430, cap, color="chalk", anchor=_an(s, "title"), seed=s.seed, max_width=1500, impact=True, importance=3, width=max(7.5, cap * 0.07))
    els.append(t)
    w = min(1500, text_width(text, cap))
    ybase = (500 if lines_est == 1 else 430) + (lines_est - 1) * cap * 1.42 + cap * 0.2
    els.append(E_underline(PAGE_W / 2 - w / 2, PAGE_W / 2 + w / 2, ybase, anchor=_an(s, "emph"), seed=s.seed + 1))
    sub = _label(s, 0)
    if sub and sub.upper() not in text and text not in sub.upper():
        els.append(E_text(sub, PAGE_W / 2, ybase + 190, _fit_cap(sub, 58, 1300), color="dim", anchor=_an(s, "label0"), seed=s.seed + 2, importance=1))
    if s.motifs:
        els.append(E_illus(s.motifs[0], PAGE_W - 330, 800, 340, anchor=_an(s, "emph"), seed=s.seed + 4, detail=2, importance=1, impact=True))
    return els, (PAGE_W / 2, 520), 0


def L_question(spec: SceneSpec):
    s = spec
    els: list[Element] = []
    q = E_illus("question", 1300 if not s.mirror else 620, 520, 640, anchor=_an(s, "motif0"), seed=s.seed, detail=s.detail, impact=True, importance=3)
    p = E_illus(_motif(s, 1, "person") if _motif(s, 1) != "question" else "person", 520 if not s.mirror else 1400, 600, 460, anchor=_an(s, "motif1"), seed=s.seed + 1, detail=s.detail, importance=2)
    els += [p, q]
    c = _caption(s, 0, 520 if not s.mirror else 1400, 920, 52, 700, color="yellow")
    if c:
        els.append(c)
    if s.title:
        els.insert(0, E_text(s.title, PAGE_W / 2, 150, _fit_cap(s.title, 62, 1300), anchor=_an(s, "title"), seed=s.seed, importance=2))
    return els, (1300 if not s.mirror else 620, 520), els.index(q)


LAYOUT_FN: dict[str, Callable[[SceneSpec], tuple[list[Element], tuple[float, float], int]]] = {
    "hero": L_hero, "duo": L_duo, "chain": L_chain, "versus": L_versus, "iceberg": L_iceberg,
    "bignumber": L_bignumber, "network": L_network, "timeline": L_timeline, "evidence": L_evidence,
    "funnel": L_funnel, "statement": L_statement, "question": L_question,
}


# ---------------------------------------------------------------------------
# anchoring words to time
# ---------------------------------------------------------------------------
def _norm(w: str) -> str:
    return re.sub(r"[^a-z0-9$%]+", "", w.casefold())


def find_phrase_time(words: list[tuple[str, float, float]], phrase: str) -> float | None:
    """Start time of the first occurrence of ``phrase`` (or of its first word)."""
    toks = [_norm(t) for t in str(phrase).split() if _norm(t)]
    if not toks or not words:
        return None
    norm = [_norm(w[0]) for w in words]
    for i in range(len(norm) - len(toks) + 1):
        if norm[i:i + len(toks)] == toks:
            return words[i][1]
    for i, w in enumerate(norm):
        if w == toks[0]:
            return words[i][1]
    # tolerate plural/tense: prefix match on a long token
    if len(toks[0]) >= 5:
        for i, w in enumerate(norm):
            if w.startswith(toks[0][:5]):
                return words[i][1]
    return None


def scene_words(row: dict[str, Any]) -> tuple[list[tuple[str, float, float]], str]:
    """(word, start, end) for a scene row, and a provenance label."""
    text = str(row.get("narration_excerpt", ""))
    toks = text.split()
    wt = row.get("word_times")
    t0, t1 = float(row["start_seconds"]), float(row["end_seconds"])
    if isinstance(wt, list) and len(wt) == len(toks) and toks:
        out = []
        for tok, pair in zip(toks, wt):
            if pair is None:
                out.append((tok, None, None))
            else:
                out.append((tok, float(pair[0]), float(pair[1])))
        # fill any gaps by interpolation
        for i, (tok, a, b) in enumerate(out):
            if a is None:
                prev = next((out[j][2] for j in range(i - 1, -1, -1) if out[j][2] is not None), t0)
                nxt = next((out[j][1] for j in range(i + 1, len(out)) if out[j][1] is not None), t1)
                out[i] = (tok, prev, max(prev, nxt))
        return out, str(row.get("word_times_source", "measured"))
    if not toks:
        return [], "none"
    weights = [max(1, len(re.sub(r"\W", "", w))) + (4 if re.search(r"[.!?]$", w) else 2 if re.search(r"[,;:]$", w) else 0) for w in toks]
    total = float(sum(weights))
    out = []
    t = t0
    span = max(0.1, t1 - t0)
    for tok, w in zip(toks, weights):
        d = span * w / total
        out.append((tok, t, t + d))
        t += d
    return out, "estimated"


# ---------------------------------------------------------------------------
# scheduling
# ---------------------------------------------------------------------------
def _reduce_detail(spec: SceneSpec, detail: int) -> list[Element]:
    sp = SceneSpec(**{**spec.__dict__, "detail": detail})
    els, _focus, _fi = LAYOUT_FN[spec.layout](sp)
    return els


def _order(els: list[Element], t_first: float, k: float) -> list[Element]:
    """Pen order. Anchored elements are drawn in the order their words are spoken; unanchored
    elements (secondary drawings, decoration) fill the pen's idle gaps instead of blocking a word
    that is about to be said. Without this, a big unanchored drawing placed first by the layout
    pushed every later caption seconds behind its word (artifact A-002)."""
    anchored = sorted((e for e in els if e.anchor is not None), key=lambda e: e.anchor - e.lead)
    free = [e for e in els if e.anchor is None]
    out: list[Element] = []
    pen = t_first
    while anchored or free:
        if anchored:
            deadline = anchored[0].anchor - anchored[0].lead
            if deadline <= pen + 0.02:
                e = anchored.pop(0)
            elif free and (pen + (free[0].nominal_duration() + 0.04) / k <= deadline + 0.35
                           or free[0].importance > anchored[0].importance):
                # fits in the gap before the next word (a 0.35 s slip is tolerated), or is simply more
                # important than the decoration waiting for its word - main content is never deferred
                e = free.pop(0)
            else:
                e = anchored.pop(0)                  # nothing fits: wait for the word
        else:
            e = free.pop(0)
        out.append(e)
        start = max(pen + e.gap / k, (e.anchor - e.lead) if e.anchor is not None else pen)
        pen = start + (e.nominal_duration() + 0.04) / k
    return out


def _schedule(els: list[Element], t_first: float, k: float = 1.0, t_last: float | None = None,
              max_pull: float = 0.0) -> tuple[list[tuple[Element, float]], float]:
    """One-pen schedule in word order (see ``_order``). ``k`` is the pen speed-up factor. With
    ``max_pull`` > 0 an element anchored too late to finish may start up to that many seconds *ahead*
    of its word - a hand that draws just before the narrator says it. Returns ([(element, start)],
    time the pen lifts)."""
    els = _order(els, t_first, k)
    durs = [e.nominal_duration() / k + 0.04 / k for e in els]
    latest = [1e9] * len(els)
    if max_pull > 0 and t_last is not None:
        nxt = t_last
        for i in range(len(els) - 1, -1, -1):
            nxt = nxt - durs[i]
            latest[i] = nxt
    placed: list[tuple[Element, float]] = []
    pen = t_first
    for i, e in enumerate(els):
        want = (e.anchor - e.lead) if e.anchor is not None else pen
        if max_pull > 0 and want > latest[i]:
            want = max(latest[i], want - max_pull)
        start = max(pen + e.gap / k, want)
        placed.append((e, start))
        pen = start + durs[i]
    return placed, pen


def _fit(els: list[Element], t_first: float, t_last: float, kmax: float) -> tuple[list[tuple[Element, float]], float, bool, float]:
    """Gentlest schedule that finishes before ``t_last``: first a natural pen (k<=1.3) honouring
    anchors, then drawing a little ahead of the words, then a faster pen. Returns (..., pull)."""
    for pull, kcap in ((0.0, 1.3), (0.9, 1.4), (1.8, kmax)):
        k = 1.0
        while True:
            placed, end = _schedule(els, t_first, k, t_last, pull)
            if end <= t_last:
                return placed, k, True, pull
            if k >= min(kmax, kcap) - 1e-9:
                break
            k = min(kmax, kcap, k + 0.1)
    return placed, kmax, False, 0.0


def compose_program(plan: dict[str, Any], *, seed: int = 0, max_pages: int = 3) -> Program:
    rows = plan.get("scenes") or []
    if not rows:
        raise ValueError("Plan has no scenes")
    duration = float(plan.get("duration_seconds") or rows[-1]["end_seconds"])
    n = len(rows)
    # --- decide pages / transitions
    pages_n = max(1, min(max_pages, n))
    page_of: list[int] = []
    trans: list[str] = ["start"]
    page = 0
    page_of.append(0)
    for i in range(1, n):
        ch = rows[i].get("chalk") or {}
        want = str(ch.get("transition") or "pan")
        if pages_n == 1:
            want = "erase"
        if want == "pan":
            page = (page + 1) % pages_n
        page_of.append(page)
        trans.append(want)

    prog = Program(world_pages=pages_n, duration=duration)
    cam: list[CamKey] = []
    scene_meta: list[dict[str, Any]] = []
    rng = random.Random(seed or 17)
    prev_layout = ""
    erase_free_at = 0.0

    def center(p: int) -> tuple[float, float]:
        return p * PAGE_W + PAGE_W / 2, PAGE_H / 2

    for i, row in enumerate(rows):
        ch = dict(row.get("chalk") or {})
        t0, t1 = float(row["start_seconds"]), float(row["end_seconds"])
        words, wsrc = scene_words(row)
        layout = str(ch.get("layout") or "hero")
        if layout not in LAYOUT_FN:
            layout = "hero"
        # anchors: dict key -> phrase; resolve to times
        anchors: dict[str, float] = {}
        for key, phrase in (ch.get("anchors") or {}).items():
            tt = find_phrase_time(words, str(phrase))
            if tt is not None:
                anchors[str(key)] = float(tt)
        spec = SceneSpec(
            index=i, t0=t0, t1=t1, layout=layout, title=str(ch.get("title") or "").strip()[:40],
            motifs=[m for m in (ch.get("motifs") or []) if m in LIBRARY] or ["document"],
            labels=[str(x).strip() for x in (ch.get("labels") or []) if str(x).strip()],
            anchors=anchors, number=str(ch.get("number") or ""), callout=str(ch.get("callout") or ""),
            detail=int(ch.get("detail", 3)), mirror=bool(ch.get("mirror", i % 2 == 1)), seed=seed * 101 + i * 13 + 3, words=words,
        )
        page = page_of[i]
        ox = page * PAGE_W
        lead_in = 0.35 if trans[i] == "pan" else (0.4 if trans[i] == "erase" else 0.25)
        hold = max(0.25, min(0.8, 0.10 * (t1 - t0)))
        t_first = max(t0 + lead_in, (erase_free_at + 0.05) if trans[i] == "erase" else 0.0)
        t_last = t1 - hold
        pacing: list[str] = []
        els, focus, focal_idx = LAYOUT_FN[layout](spec)
        sched, k, ok, pull = _fit(els, t_first, t_last, 1.5)
        detail = spec.detail

        def _relayout(d: int) -> None:
            nonlocal els, focus, focal_idx
            spec.detail = d
            els, focus, focal_idx = LAYOUT_FN[layout](spec)

        # pacing decisions, gentlest first: no hatching -> drop decoration -> drop secondary
        # captions -> bare outlines -> drop the costliest extras -> only then hurry the pen
        if not ok and detail > 2:
            detail = 2
            _relayout(2)
            sched, k, ok, pull = _fit(els, t_first, t_last, 1.5)
            pacing.append("detail->2")
        for imp in (0, 1):
            if ok:
                break
            keep = [e for e in els if e.importance > imp]
            if 2 <= len(keep) < len(els):
                focal_el = els[focal_idx] if focal_idx < len(els) else None
                els = keep
                focal_idx = els.index(focal_el) if focal_el in els else 0
                sched, k, ok, pull = _fit(els, t_first, t_last, 1.5)
                pacing.append(f"dropped importance<={imp}")
        if not ok and detail > 1:
            detail = 1
            keep_els = [e for e in els]
            _relayout(1)
            els = [e for e in els if e.importance > (1 if "dropped importance<=1" in pacing else 0 if "dropped importance<=0" in pacing else -1)] or els
            focal_idx = min(focal_idx, len(els) - 1)
            sched, k, ok, pull = _fit(els, t_first, t_last, 1.5)
            pacing.append("detail->1")
        while not ok and len(els) > 2:
            focal_el = els[focal_idx] if focal_idx < len(els) else None
            cands = [e for e in els if e is not focal_el and e.importance <= 2 and e.kind in ("text", "illus", "arrow", "line", "underline")]
            if not cands:
                break
            victim = max(cands, key=lambda e: e.nominal_duration())
            els = [e for e in els if e is not victim]
            focal_idx = els.index(focal_el) if focal_el in els else 0
            sched, k, ok, pull = _fit(els, t_first, t_last, 1.5)
            pacing.append(f"dropped {victim.tag}")
        if not ok:
            sched, k, ok, pull = _fit(els, t_first, t_last, 2.0)
            pacing.append(f"pen x{k:.1f}")
        first_t = t_first
        n_strokes0 = len(prog.strokes)
        last_end = t_first
        focal_time = None
        el_times: list[tuple[Element, float, float]] = []
        for e, st in sched:
            en = _emit(prog, e, st, ox, k, rng)
            el_times.append((e, st, en))
            last_end = max(last_end, en)
            if focal_idx < len(els) and e is els[focal_idx]:
                focal_time = st
            if e.impact:
                prog.cues.append({"t": float(en), "kind": "impact", "gain": 1.0 if e.importance >= 3 else 0.6})
        # ---- camera for this scene
        cx0, cy0 = center(page)
        fx, fy = ox + focus[0], focus[1]
        scene_pts = [st_.pts for st_ in prog.strokes[n_strokes0:]]
        if scene_pts:
            allp = np.concatenate(scene_pts)
            bx0, by0 = float(allp[:, 0].min()) - 40, float(allp[:, 1].min()) - 40
            bx1, by1 = float(allp[:, 0].max()) + 40, float(allp[:, 1].max()) + 40
        else:
            bx0, by0, bx1, by1 = cx0 - PAGE_W / 2, 0.0, cx0 + PAGE_W / 2, float(PAGE_H)
        fit_zoom = max(1.0, min(PAGE_W / max(1.0, bx1 - bx0), PAGE_H / max(1.0, by1 - by0)) * 0.97)

        def frame_center(zoom: float, want_x: float, want_y: float) -> tuple[float, float]:
            """Move the view toward (want_x, want_y) but never so far that content is cropped."""
            hw, hh = PAGE_W / zoom / 2, PAGE_H / zoom / 2
            lo_x, hi_x = bx1 - hw, bx0 + hw
            lo_y, hi_y = by1 - hh, by0 + hh
            x = (bx0 + bx1) / 2 if lo_x > hi_x else min(max(want_x, lo_x), hi_x)
            y = (by0 + by1) / 2 if lo_y > hi_y else min(max(want_y, lo_y), hi_y)
            return x, y

        push = min(1.045 + 0.03 * ((i % 3) / 2), fit_zoom)
        style = str(ch.get("camera") or ("closeup" if i % 4 == 2 else "push"))
        arrive = t0 + (0.55 if trans[i] == "pan" else 0.0)
        cam.append(CamKey(max(0.0, arrive if i else 0.0), cx0, cy0, 1.0))
        if style == "closeup" and focal_time is not None and fit_zoom > 1.12:
            tz = max(arrive + 0.2, focal_time - 0.25)
            zoom = min(1.28 + 0.07 * (i % 2), fit_zoom)
            px, py = frame_center(zoom, fx, fy)
            cam.append(CamKey(tz, cx0, cy0, 1.0))
            cam.append(CamKey(min(t1 - 0.9, tz + 1.3), px, py, zoom))
            cam.append(CamKey(max(tz + 1.4, t1 - 0.35), px, py, zoom * 1.02))
        elif style == "hold":
            cam.append(CamKey(max(arrive + 0.1, t1 - 0.25), cx0, cy0, 1.0))
        else:
            px, py = frame_center(push, cx0 + (fx - cx0) * 0.12, cy0 + (fy - cy0) * 0.12)
            cam.append(CamKey(max(arrive + 0.1, t1 - 0.25), px, py, push))
        # ---- transition out
        if i + 1 < n:
            nxt = trans[i + 1]
            if nxt == "pan":
                ncx, ncy = center(page_of[i + 1])
                dist = abs(page_of[i + 1] - page)
                dur = min(1.6, 0.8 + 0.3 * max(0, dist - 1))
                prog.cues.append({"t": float(t1 - 0.20), "kind": "whoosh", "gain": 0.8, "dur": dur})
                # page we are about to enter must be clean; clear it unseen while we're still looking away
                nxt_page = page_of[i + 1]
                if any(rr_page == nxt_page for rr_page in page_of[:i + 1]):
                    prog.erases.append(Erase(max(0.0, t1 - 1.6), 0.0, (nxt_page * PAGE_W, 0, (nxt_page + 1) * PAGE_W, PAGE_H), visible=False, seed=i))
                cam.append(CamKey(t1 - 0.20, cam[-1].cx, cam[-1].cy, cam[-1].zoom))
                cam.append(CamKey(t1 - 0.20 + dur, ncx, ncy, 1.0))
            else:  # erase in place; the next scene waits until the board is clean
                e_start = max(last_end + 0.2, t1 - 1.0)
                e_end = max(e_start + 0.5, min(t1 + 0.3, e_start + 1.25))
                prog.erases.append(Erase(e_start, e_end - e_start, (ox, 0, ox + PAGE_W, PAGE_H), True, seed + i))
                prog.cues.append({"t": float(e_start), "kind": "erase", "gain": 1.0, "dur": float(e_end - e_start)})
                erase_free_at = e_end
                cam.append(CamKey(e_start, cam[-1].cx, cam[-1].cy, cam[-1].zoom))
                cam.append(CamKey(e_end, cx0, cy0, 1.0))
        scene_meta.append({
            "scene": i + 1, "start": round(t0, 3), "end": round(t1, 3), "page": page, "transition_in": trans[i],
            "layout": layout, "motifs": spec.motifs, "labels": spec.labels, "title": spec.title,
            "strokes": len(prog.strokes) - n_strokes0, "pen_busy_until": round(last_end, 3),
            "pen_speedup": round(k, 2), "anchor_pull": round(pull, 2), "fits": bool(ok), "pacing": pacing, "detail": detail, "word_timing": wsrc, "camera": style,
            "first_stroke": round(prog.strokes[n_strokes0].t0, 3) if len(prog.strokes) > n_strokes0 else None,
        })
        prev_layout = layout

    # keep camera keys strictly ordered and unique in time
    cam.sort(key=lambda k: k.t)
    cleaned: list[CamKey] = []
    for k in cam:
        if cleaned and k.t <= cleaned[-1].t + 1e-3:
            k = CamKey(cleaned[-1].t + 0.002, k.cx, k.cy, k.zoom, k.ease)
        cleaned.append(k)
    prog.camera = cleaned
    prog.meta["scenes"] = scene_meta
    prog.meta["pages"] = pages_n
    return prog


def _emit(prog: Program, e: Element, start: float, ox: float, k: float, rng: random.Random) -> float:
    """Append the strokes of one element; returns the time the pen lifts."""
    t = start + e.gap / k
    last_stage = 0
    sp_var = rng.uniform(0.9, 1.12)
    for p in e.paths:
        if p.stage != last_stage:
            t += 0.08 / k
            last_stage = p.stage
        d = path_time(p) / sp_var / k
        pts = p.pts + np.array([ox, 0.0], np.float32)
        prog.strokes.append(Stroke(pts, t, max(0.025, d), p.color, p.width, dryness=p.dryness,
                                   ease=0.45 if e.kind == "text" else 0.7, tag=e.tag))
        t += d + 0.012 / k
    return t
