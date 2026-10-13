"""FERN chalk engine: a persistent, incremental chalkboard renderer.

The legacy renderer (``chalkboard.py``) redraws every frame from scratch with
Pillow.  This engine instead keeps a *persistent* chalk layer on a large world
board and only paints the newly revealed part of each stroke per frame.  That
makes it cheap enough to afford the things that make a board feel real:

* grainy, pressure-varying chalk that breaks up on the board's tooth
* a camera (pans between "pages" of the board, push-ins, close-ups, drift)
* an eraser that wipes the board and leaves ghost smudges behind
* chalk dust falling from the drawing tip
* subtle glow, vignette and film grain

Everything is deterministic for a given seed, so a render can be reproduced.

Coordinates are *world pixels*.  One "page" of the board is ``PAGE_W`` x
``PAGE_H`` (1920 x 1080); pages sit side by side on a horizontal strip.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

try:  # OpenCV is a hard requirement of the new engine (see requirements.txt)
    import cv2
except ImportError as exc:  # pragma: no cover - exercised only without OpenCV
    raise ImportError(
        "The FERN chalk engine needs OpenCV: pip install opencv-python-headless"
    ) from exc

PAGE_W = 1920
PAGE_H = 1080

# Restrained palette: off-white chalk dominates, teal and yellow are accents.
PALETTE: dict[str, tuple[int, int, int]] = {
    "chalk": (240, 238, 226),
    "dim": (160, 176, 170),
    "teal": (139, 214, 197),
    "yellow": (242, 199, 106),
}
BOARD_DARK = np.array((15, 23, 24), np.float32)
BOARD_LIGHT = np.array((38, 51, 48), np.float32)


# --------------------------------------------------------------------------
# small math helpers
# --------------------------------------------------------------------------
def smoothstep(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    x = 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def densify(points: Iterable[Iterable[float]], spacing: float = 2.5) -> np.ndarray:
    """Resample a polyline so consecutive points are ~``spacing`` px apart."""
    pts = np.asarray(list(points), np.float32)
    if len(pts) < 2:
        return pts.reshape(-1, 2)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    total = float(cum[-1])
    if total < 1e-3:
        return pts[:1].copy()
    n = max(2, int(math.ceil(total / spacing)) + 1)
    s = np.linspace(0.0, total, n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1).astype(np.float32)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.hypot(*np.diff(points, axis=0).T).sum())


# --------------------------------------------------------------------------
# program model
# --------------------------------------------------------------------------
@dataclass
class Stroke:
    """One continuous chalk stroke that is revealed over [t0, t0+dur]."""

    pts: np.ndarray
    t0: float
    dur: float
    color: str = "chalk"
    width: float = 6.0
    dryness: float = 0.55      # 0 = solid, 1 = very dry/broken chalk
    pressure: float = 1.0
    ease: float = 0.6          # 0 = constant speed, 1 = strong ease in/out
    tag: str = ""
    cum: np.ndarray = field(init=False, repr=False)
    length: float = field(init=False)

    def __post_init__(self) -> None:
        self.pts = np.asarray(self.pts, np.float32).reshape(-1, 2)
        if len(self.pts) == 1:
            self.pts = np.repeat(self.pts, 2, axis=0) + np.array([[0, 0], [0.01, 0]], np.float32)
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.cum = np.concatenate(([0.0], np.cumsum(seg))).astype(np.float64)
        self.length = float(self.cum[-1])
        self.dur = max(0.02, float(self.dur))

    @property
    def t1(self) -> float:
        return self.t0 + self.dur

    def progress(self, t: float) -> float:
        u = (t - self.t0) / self.dur
        if u <= 0.0:
            return 0.0
        if u >= 1.0:
            return 1.0
        return u + self.ease * (smoothstep(u) - u)


@dataclass
class Erase:
    """A (usually visible) eraser sweep over a rectangle of the world."""

    t0: float
    dur: float
    rect: tuple[int, int, int, int]
    visible: bool = True
    seed: int = 0


@dataclass
class CamKey:
    t: float
    cx: float
    cy: float
    zoom: float = 1.0
    ease: str = "smooth"  # smooth | hold


@dataclass
class Program:
    """A complete, timed drawing program for one video."""

    world_pages: int
    duration: float
    strokes: list[Stroke] = field(default_factory=list)
    erases: list[Erase] = field(default_factory=list)
    camera: list[CamKey] = field(default_factory=list)
    cues: list[dict[str, Any]] = field(default_factory=list)   # {"t","kind","gain"}
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def world_w(self) -> int:
        return PAGE_W * self.world_pages

    @property
    def world_h(self) -> int:
        return PAGE_H


# --------------------------------------------------------------------------
# textures
# --------------------------------------------------------------------------
def _fbm(h: int, w: int, rng: np.random.Generator, octaves: Iterable[tuple[int, float]]) -> np.ndarray:
    """Multi-scale noise in roughly [0,1]. ``octaves`` = (cell_px, weight)."""
    out = np.zeros((h, w), np.float32)
    wsum = 0.0
    for cell, weight in octaves:
        gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
        small = rng.random((gh, gw), dtype=np.float32)
        big = cv2.resize(small, (gw * cell, gh * cell), interpolation=cv2.INTER_CUBIC)[:h, :w]
        out += weight * big
        wsum += weight
    out /= max(1e-6, wsum)
    return out


def make_board(world_w: int, world_h: int, seed: int = 1984) -> np.ndarray:
    """A dark slate board: mottling, tooth, old eraser streaks and scratches."""
    rng = np.random.default_rng(seed)
    n = _fbm(world_h, world_w, rng, [(480, 1.0), (140, 0.8), (36, 0.5), (9, 0.25)])
    n = (n - n.min()) / max(1e-6, float(n.max() - n.min()))
    fine = rng.random((world_h, world_w), dtype=np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 0.9)
    t = np.clip(0.28 + 0.52 * (n - 0.5) + 0.16 * (fine - 0.5), 0.0, 1.0)
    board = BOARD_DARK[None, None, :] + (BOARD_LIGHT - BOARD_DARK)[None, None, :] * t[..., None]

    # Old eraser streaks: broad, low-contrast, mostly horizontal smears.
    streaks = np.zeros((world_h, world_w), np.float32)
    for _ in range(max(10, world_w // 260)):
        x = int(rng.integers(-200, world_w))
        y = int(rng.integers(40, world_h - 40))
        length = int(rng.integers(300, 900))
        pts = np.array([[x + i * length / 6, y + math.sin(i * 0.9 + x) * rng.uniform(8, 40)] for i in range(7)], np.float32)
        cv2.polylines(streaks, [np.round(pts * 8).astype(np.int32)], False, float(rng.uniform(0.04, 0.10)),
                      thickness=int(rng.integers(60, 170)), lineType=cv2.LINE_AA, shift=3)
    streaks = cv2.GaussianBlur(streaks, (0, 0), 28)
    board += streaks[..., None] * np.array((46, 60, 56), np.float32)[None, None, :]

    # A handful of faint scratches.
    scr = np.zeros((world_h, world_w), np.float32)
    for _ in range(max(14, world_w // 200)):
        x, y = int(rng.integers(0, world_w)), int(rng.integers(0, world_h))
        a = float(rng.uniform(0, math.pi))
        l = int(rng.integers(40, 220))
        cv2.line(scr, (x, y), (int(x + math.cos(a) * l), int(y + math.sin(a) * l)), float(rng.uniform(0.05, 0.13)), 1, cv2.LINE_AA)
    board += scr[..., None] * np.array((70, 84, 80), np.float32)[None, None, :]
    return np.clip(board, 0, 255).astype(np.uint8)


def make_grain(world_w: int, world_h: int, seed: int = 77) -> np.ndarray:
    """Chalk tooth: where the surface catches chalk (uint8, 0..255)."""
    rng = np.random.default_rng(seed)
    fine = cv2.GaussianBlur(rng.random((world_h, world_w), dtype=np.float32), (0, 0), 0.75)
    mid = _fbm(world_h, world_w, rng, [(5, 1.0), (13, 0.8), (31, 0.4)])
    g = 0.5 * fine + 0.5 * mid
    g = (g - g.min()) / max(1e-6, float(g.max() - g.min()))
    # Stretch contrast so some pixels reliably miss and some reliably catch.
    g = np.clip((g - 0.5) * 1.55 + 0.5, 0, 1)
    return (g * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------
class ChalkEngine:
    """Renders a :class:`Program` frame by frame (strictly sequential in time)."""

    def __init__(self, program: Program, out_w: int, out_h: int, fps: int, *, seed: int = 7,
                 dust: bool = True, glow: float = 0.22, grain_noise: float = 2.2) -> None:
        self.p = program
        self.out_w, self.out_h, self.fps = int(out_w), int(out_h), int(fps)
        self.dust_on = dust
        self.glow = glow
        self.grain_noise = grain_noise
        self.rng = np.random.default_rng(seed)
        ww, wh = program.world_w, program.world_h
        self.board = make_board(ww, wh, 1984 + seed)
        self.grain = make_grain(ww, wh, 77 + seed)
        self.ink_pm = np.zeros((wh, ww, 3), np.uint8)     # premultiplied colour
        self.ink_a = np.zeros((wh, ww), np.uint8)
        self.strokes = sorted(program.strokes, key=lambda s: s.t0)
        self.erases = sorted(program.erases, key=lambda e: e.t0)
        self.cam = sorted(program.camera, key=lambda k: k.t) or [CamKey(0.0, PAGE_W / 2, PAGE_H / 2, 1.0)]
        self.cues = sorted(program.cues, key=lambda c: c["t"])
        self._next_stroke = 0
        self._active: list[dict[str, Any]] = []
        self._next_erase = 0
        self._erasing: list[dict[str, Any]] = []
        self._particles: list[list[float]] = []
        self._last_t = -1.0
        # screen-space overlays
        yy, xx = np.mgrid[0:self.out_h, 0:self.out_w].astype(np.float32)
        r = np.hypot((xx - self.out_w / 2) / (self.out_w / 2), (yy - self.out_h / 2) / (self.out_h / 2))
        vig = (1.0 - 0.34 * np.clip(r - 0.35, 0, 1.2) ** 1.6)
        self.vignette_u8 = cv2.merge([np.round(vig * 255).astype(np.uint8)] * 3)
        pool = np.random.default_rng(seed + 991).normal(0.0, self.grain_noise, (6, self.out_h, self.out_w)).astype(np.float32)
        self._noise_pos = [cv2.merge([np.clip(np.round(n), 0, 255).astype(np.uint8)] * 3) for n in pool]
        self._noise_neg = [cv2.merge([np.clip(np.round(-n), 0, 255).astype(np.uint8)] * 3) for n in pool]
        self._frame_i = 0

    # ----- camera ---------------------------------------------------------
    def camera_at(self, t: float) -> tuple[float, float, float]:
        keys = self.cam
        if t <= keys[0].t:
            cx, cy, z = keys[0].cx, keys[0].cy, keys[0].zoom
        elif t >= keys[-1].t:
            cx, cy, z = keys[-1].cx, keys[-1].cy, keys[-1].zoom
        else:
            i = int(np.searchsorted([k.t for k in keys], t, side="right")) - 1
            a, b = keys[i], keys[i + 1]
            u = (t - a.t) / max(1e-6, b.t - a.t)
            e = 0.0 if b.ease == "hold" and u < 1 else smootherstep(u)
            cx = a.cx + (b.cx - a.cx) * e
            cy = a.cy + (b.cy - a.cy) * e
            # zoom interpolates geometrically so a push-in feels even
            z = math.exp(math.log(a.zoom) + (math.log(b.zoom) - math.log(a.zoom)) * e)
        # gentle handheld drift (kept sub-pixel-smooth) + decaying impact shake
        dx = 5.0 * math.sin(t * 0.61 + 0.4) + 2.2 * math.sin(t * 1.37)
        dy = 3.5 * math.sin(t * 0.47 + 1.7) + 1.6 * math.sin(t * 1.13 + 0.3)
        for c in self.cues:
            if c["t"] > t:
                break
            if c.get("kind") == "impact" and t - c["t"] < 0.55:
                age = t - c["t"]
                amp = 9.0 * float(c.get("gain", 1.0)) * math.exp(-age * 9.0)
                dx += amp * math.sin(age * 58.0)
                dy += amp * 0.7 * math.cos(age * 49.0)
        return cx + dx, cy + dy, z

    def _view(self, t: float) -> tuple[float, float, float]:
        cx, cy, z = self.camera_at(t)
        z = max(0.9, min(3.2, z))
        vw, vh = PAGE_W / z, PAGE_H / z
        x0 = min(max(0.0, cx - vw / 2), self.p.world_w - vw)
        y0 = min(max(0.0, cy - vh / 2), self.p.world_h - vh)
        return x0, y0, self.out_w / vw

    # ----- chalk painting -------------------------------------------------
    def _paint(self, path: np.ndarray, st: Stroke, s_start: float, s_end: float) -> None:
        """Paint ``path`` (k>=2 world points) as chalk. s_* are 0..1 along the stroke."""
        if len(path) < 2:
            return
        wpx = max(1, int(round(st.width)))
        pad = wpx + 3
        x0 = int(max(0, math.floor(path[:, 0].min()) - pad))
        y0 = int(max(0, math.floor(path[:, 1].min()) - pad))
        x1 = int(min(self.p.world_w, math.ceil(path[:, 0].max()) + pad + 1))
        y1 = int(min(self.p.world_h, math.ceil(path[:, 1].max()) + pad + 1))
        if x1 <= x0 or y1 <= y0:
            return
        h, w = y1 - y0, x1 - x0
        mask = np.zeros((h, w), np.uint8)
        local = (path - np.array([x0, y0], np.float32)) * 16.0
        # pressure: lighter at both ends of a stroke, heaviest through the middle
        n = len(path)
        step = 4 if n > 8 else max(1, n - 1)
        for i in range(0, n - 1, step):
            j = min(n - 1, i + step)
            s = s_start + (s_end - s_start) * ((i + j) / 2) / max(1, n - 1)
            taper = 0.62 + 0.38 * min(1.0, math.sin(math.pi * min(1.0, max(0.0, s))) ** 0.5 * 1.15)
            val = int(255 * min(1.0, st.pressure * taper))
            cv2.polylines(mask, [np.round(local[i:j + 1]).astype(np.int32)], False, val,
                          thickness=wpx, lineType=cv2.LINE_AA, shift=4)
        g = self.grain[y0:y1, x0:x1].astype(np.float32) / 255.0
        m = mask.astype(np.float32) / 255.0
        dry = float(st.dryness)
        # Chalk catches on the surface tooth. Perturb the mask with the grain and
        # threshold it: edges turn ragged, the core gets speckled, and drier
        # chalk (higher ``dryness``) drops out more.
        amp = 0.70 + 0.85 * dry
        x = m + (g - 0.5) * amp
        lo, hi = 0.30 + 0.12 * dry, 0.80
        cov = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        cov = cov * cov * (3.0 - 2.0 * cov)
        cov *= np.clip(m * 1.6, 0.0, 1.0)   # nothing outside the pen footprint
        # a slightly lighter, tighter core makes the stroke read as pressed chalk
        a_old = self.ink_a[y0:y1, x0:x1]
        a_new = (cov * 255.0).astype(np.uint8)
        take = a_new > a_old
        if not take.any():
            return
        color = np.array(PALETTE.get(st.color, PALETTE["chalk"]), np.float32)
        pm_roi = self.ink_pm[y0:y1, x0:x1]
        pm_roi[take] = (cov[take][:, None] * color[None, :]).astype(np.uint8)
        a_old[take] = a_new[take]

    def _advance_stroke(self, rec: dict[str, Any], t: float) -> bool:
        """Paint the newly revealed part of a stroke. Returns True when finished."""
        st: Stroke = rec["stroke"]
        f = st.progress(t)
        target = f * st.length
        drawn = rec["drawn"]
        if target > drawn + 1e-3 or (f >= 1.0 and not rec["done"]):
            cum, pts = st.cum, st.pts
            j0 = int(np.searchsorted(cum, drawn, side="right"))
            j1 = int(np.searchsorted(cum, target, side="right"))
            middle = pts[j0:j1]
            if j1 >= len(pts):
                tip = pts[-1]
            else:
                i = max(0, j1 - 1)
                span = max(1e-6, cum[i + 1] - cum[i])
                tip = pts[i] + (pts[i + 1] - pts[i]) * float((target - cum[i]) / span)
            pieces = [rec["tip"]] + [m for m in middle] + [tip]
            path = np.asarray(pieces, np.float32)
            if len(path) >= 2 and path_length(path) > 0.05:
                self._paint(path, st, drawn / max(1e-6, st.length), target / max(1e-6, st.length))
            rec["drawn"], rec["tip"] = target, tip
            if self.dust_on and f < 1.0:
                self._spawn_dust(tip, st, target - drawn)
        if f >= 1.0:
            rec["done"] = True
        return rec["done"]

    # ----- eraser ---------------------------------------------------------
    @staticmethod
    def _eraser_path(rect: tuple[int, int, int, int], seed: int) -> np.ndarray:
        x0, y0, x1, y1 = rect
        rng = np.random.default_rng(seed + 11)
        band = 150.0
        rows = max(2, int(math.ceil((y1 - y0) / band)))
        pts: list[tuple[float, float]] = []
        for r in range(rows):
            y = y0 + (r + 0.5) * (y1 - y0) / rows
            jitter = lambda: float(rng.uniform(-18, 18))
            a, b = (x0 - 40, x1 + 40) if r % 2 == 0 else (x1 + 40, x0 - 40)
            pts += [(a, y + jitter()), ((a + b) / 2, y + jitter() * 1.4), (b, y + jitter())]
        return densify(pts, 8.0)

    def _start_erase(self, e: Erase) -> None:
        x0, y0, x1, y1 = e.rect
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.p.world_w, x1), min(self.p.world_h, y1)
        if not e.visible or e.dur < 0.05:
            self._clear_rect((x0, y0, x1, y1))
            return
        path = self._eraser_path((x0, y0, x1, y1), e.seed)
        cum = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))))
        self._erasing.append({"e": e, "path": path, "cum": cum, "drawn": 0.0, "rect": (x0, y0, x1, y1)})

    def _clear_rect(self, rect: tuple[int, int, int, int]) -> None:
        """Wipe a region instantly, leaving a faint ghost of what was there."""
        x0, y0, x1, y1 = rect
        a = self.ink_a[y0:y1, x0:x1]
        if a.size and a.any():
            ghost = cv2.GaussianBlur(a, (0, 0), 7).astype(np.float32) / 255.0 * 0.04
            dim = np.array(PALETTE["dim"], np.float32)
            roi = self.board[y0:y1, x0:x1].astype(np.float32)
            roi += (dim[None, None, :] - roi) * ghost[..., None]
            self.board[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
        self.ink_a[y0:y1, x0:x1] = 0
        self.ink_pm[y0:y1, x0:x1] = 0

    def _advance_erase(self, rec: dict[str, Any], t: float) -> bool:
        e: Erase = rec["e"]
        u = smoothstep((t - e.t0) / max(1e-6, e.dur))
        total = float(rec["cum"][-1])
        target = u * total
        if target <= rec["drawn"] + 0.5 and u < 1.0:
            return False
        j0 = int(np.searchsorted(rec["cum"], rec["drawn"], side="right"))
        j1 = int(np.searchsorted(rec["cum"], target, side="right"))
        seg = rec["path"][max(0, j0 - 1):min(len(rec["path"]), j1 + 1)]
        rec["drawn"] = target
        if len(seg) >= 2:
            thick = 160
            pad = thick
            rx0, ry0, rx1, ry1 = rec["rect"]
            x0 = int(max(0, seg[:, 0].min() - pad)); x1 = int(min(self.p.world_w, seg[:, 0].max() + pad))
            y0 = int(max(0, seg[:, 1].min() - pad)); y1 = int(min(self.p.world_h, seg[:, 1].max() + pad))
            x0, y0, x1, y1 = max(x0, rx0), max(y0, ry0), min(x1, rx1), min(y1, ry1)
            if x1 > x0 and y1 > y0:
                m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                loc = (seg - np.array([x0, y0], np.float32)) * 16.0
                cv2.polylines(m, [np.round(loc).astype(np.int32)], False, 255, thickness=thick, lineType=cv2.LINE_AA, shift=4)
                m = cv2.GaussianBlur(m, (0, 0), 9)
                E = m.astype(np.float32) / 255.0
                a = self.ink_a[y0:y1, x0:x1]
                if a.any():
                    ghost = cv2.GaussianBlur(a, (0, 0), 7).astype(np.float32) / 255.0 * 0.05 * E
                    dim = np.array(PALETTE["dim"], np.float32)
                    roi = self.board[y0:y1, x0:x1].astype(np.float32)
                    roi += (dim[None, None, :] - roi) * ghost[..., None]
                    self.board[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
                    keep = (1.0 - np.clip(E * 1.35, 0, 1))
                    self.ink_a[y0:y1, x0:x1] = (a.astype(np.float32) * keep).astype(np.uint8)
                    self.ink_pm[y0:y1, x0:x1] = (self.ink_pm[y0:y1, x0:x1].astype(np.float32) * keep[..., None]).astype(np.uint8)
        if u >= 1.0:
            self._clear_rect(rec["rect"])
            return True
        return False

    # ----- dust -----------------------------------------------------------
    def _spawn_dust(self, tip: np.ndarray, st: Stroke, moved: float) -> None:
        if moved <= 0.5:
            return
        count = min(3, 1 + int(moved / 14.0))
        for _ in range(count):
            if self.rng.random() < 0.55:
                self._particles.append([
                    float(tip[0] + self.rng.normal(0, st.width * 0.45)),
                    float(tip[1] + self.rng.normal(0, st.width * 0.45)),
                    float(self.rng.normal(0, 7.0)), float(self.rng.uniform(-3.0, 9.0)),
                    float(self.rng.uniform(0.45, 1.1)), 0.0,
                    float(self.rng.uniform(0.8, 1.9)),
                ])
        if len(self._particles) > 220:
            del self._particles[:len(self._particles) - 220]

    def _step_particles(self, dt: float) -> None:
        alive = []
        for p in self._particles:
            p[5] += dt
            if p[5] >= p[4]:
                continue
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[3] += 46.0 * dt   # gravity
            p[2] *= 0.97
            alive.append(p)
        self._particles = alive

    # ----- frame ----------------------------------------------------------
    def render(self, t: float) -> np.ndarray:
        """Advance the simulation to time ``t`` and return an RGB uint8 frame."""
        dt = max(0.0, t - self._last_t) if self._last_t >= 0 else 0.0
        self._last_t = t
        # start things that are due
        while self._next_erase < len(self.erases) and self.erases[self._next_erase].t0 <= t:
            self._start_erase(self.erases[self._next_erase])
            self._next_erase += 1
        while self._next_stroke < len(self.strokes) and self.strokes[self._next_stroke].t0 <= t:
            st = self.strokes[self._next_stroke]
            self._active.append({"stroke": st, "drawn": 0.0, "tip": st.pts[0], "done": False})
            self._next_stroke += 1
        # erase before painting new ink so a fresh stroke is never wiped by an old sweep
        self._erasing = [r for r in self._erasing if not self._advance_erase(r, t)]
        self._active = [r for r in self._active if not self._advance_stroke(r, t)]
        if self.dust_on:
            self._step_particles(dt)

        x0, y0, s = self._view(t)
        M = np.array([[s, 0.0, -x0 * s], [0.0, s, -y0 * s]], np.float32)
        size = (self.out_w, self.out_h)
        flags = cv2.INTER_LINEAR
        board = cv2.warpAffine(self.board, M, size, flags=flags, borderMode=cv2.BORDER_REPLICATE)
        pm = cv2.warpAffine(self.ink_pm, M, size, flags=flags, borderMode=cv2.BORDER_CONSTANT)
        al = cv2.warpAffine(self.ink_a, M, size, flags=flags, borderMode=cv2.BORDER_CONSTANT)

        # fixed-point composite: board * (1 - alpha) + premultiplied chalk
        inv = cv2.cvtColor(cv2.convertScaleAbs(al, alpha=-0.93, beta=255), cv2.COLOR_GRAY2RGB)
        out = cv2.multiply(board, inv, scale=1.0 / 255.0)
        out = cv2.addWeighted(out, 1.0, pm, 0.93, 0)
        if self.glow > 0:
            halo = cv2.GaussianBlur(pm, (0, 0), 2.6 * max(0.6, s))
            out = cv2.addWeighted(out, 1.0, halo, self.glow, 0)
        if self._particles:
            layer = np.zeros((self.out_h, self.out_w, 3), np.uint8)
            dust = PALETTE["chalk"]
            for px, py, _vx, _vy, life, age, size_px in self._particles:
                sx, sy = (px - x0) * s, (py - y0) * s
                if 0 <= sx < self.out_w and 0 <= sy < self.out_h:
                    fade = (1.0 - age / life) ** 1.4 * 0.8
                    cv2.circle(layer, (int(sx), int(sy)), max(1, int(round(size_px * s * 0.8))),
                               tuple(int(c * fade) for c in dust), -1, cv2.LINE_AA)
            out = cv2.add(out, layer)
        out = cv2.multiply(out, self.vignette_u8, scale=1.0 / 255.0)
        if self.grain_noise > 0:
            k = self._frame_i % len(self._noise_pos)
            self._frame_i += 1
            out = cv2.subtract(cv2.add(out, self._noise_pos[k]), self._noise_neg[k])
        return out
