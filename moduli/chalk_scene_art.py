"""Hand-composed, scene-specific layouts (opt-in via ``chalk["art"]``).

The generic layouts (hero, chain, duo, ...) place stock icons. These compositions are written *for one
sentence*: they use drawings made for that idea, layer them (a figure in front of glass, a card behind it),
shade with purpose (hatching marks shadow, frost or danger, never filler) and hang every element on the
spoken word that triggers it.

They are never chosen automatically; a scene plan selects one with ``"art": "<name>"`` and supplies the
anchor phrases it names. Anchor keys are listed in each function's docstring.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from moduli.chalk_composer import (
    ART, PAGE_H, PAGE_W, Element, Path, SceneSpec, E_line, E_text, _an, _fit_cap,
)
from moduli.chalk_engine import densify
from moduli.chalk_illustrations import Part, _dashed1, _occlude, arc, build_illustration


def _parts_element(parts: Iterable[Part], size: float, anchor: float | None, tag: str, *, importance: int = 3,
                   impact: bool = False, lead: float = 0.65, focus: tuple[float, float] = (0, 0), speed_scale: float = 1.0) -> Element:
    base = max(3.6, min(9.0, size * 0.0158))
    sp = (1800.0 + size * 4.2) * speed_scale
    paths = [Path(p.pts, p.color, max(2.6, base * p.weight), p.stage, 0.5 + 0.1 * (p.stage >= 2), sp, dash=p.dash) for p in parts]
    return Element("illus", paths, anchor, lead, 0.0, impact, tag, focus=focus, importance=importance)


def _later(t: float | None, dt: float) -> float | None:
    return None if t is None else t + dt


def _strike(cx: float, cy: float, w: float, h: float, anchor: float | None, seed: int) -> Element:
    """A bold yellow diagonal 'no' across a drawing - a second stroke follows close behind the first."""
    e = E_line((cx - w / 2, cy + h / 2), (cx + w / 2, cy - h / 2), color="yellow", width=9.5, anchor=anchor, seed=seed, importance=2, speed=2600.0)
    e.impact = True
    e.kind = "strike"
    e.tag = "strike"
    return e


def rejected_judges(spec: SceneSpec):
    """'Not by a teacher, not by a boss, not by a credit bureau.'

    Three *specific* judges at different sizes and heights - an open gradebook, an org chart with its top box
    circled, a credit-score dial - each struck out in yellow as it is named.
    Anchors: judge0, judge1, judge2 (the nouns); optional label0-2 are skipped when time is short.
    """
    s = spec
    props = [("gradebook", 440, 520, 500), ("orgchart", 960, 600, 430), ("score_dial", 1480, 530, 500)]
    els: list[Element] = []
    names = ["TEACHER", "BOSS", "CREDIT BUREAU"]
    focal = 0
    for i, (name, cx, cy, size) in enumerate(props):
        t = _an(s, f"judge{i}")
        parts = build_illustration(name, cx, cy, size, seed=s.seed + i, detail=s.detail, rotate=(-2.5, 1.5, -1.0)[i])
        els.append(_parts_element(parts, size, t, f"illus:{name}", importance=3, focus=(cx, cy)))
        els.append(_strike(cx, cy, size * 0.74, size * 0.62, _later(t, 0.15), s.seed + 20 + i))
        lab = names[i]
        els.append(E_text(lab, cx, cy + size * 0.5 + 70, _fit_cap(lab, 46, 440), color="yellow" if i == 2 else "chalk",
                          anchor=_later(t, 0.1), seed=s.seed + 40 + i, importance=1))
        if i == 2:
            focal = len(els) - 3
    return els, (PAGE_W / 2, 520), focal


def two_dials(spec: SceneSpec):
    """'...how dangerous you might be, and how much effort the model should spend on you.'

    Danger: a gauge whose needle has swung into a crosshatched zone. Effort: a tank labelled MODEL feeding a
    half-closed valve that lets only a thin trickle reach a figure at the end of the pipe - rationing.
    Anchors: gauge ('dangerous'), valve ('effort'), person ('you', last word).
    """
    s = spec
    els: list[Element] = []
    gauge = build_illustration("danger_gauge", 540, 450, 600, seed=s.seed, detail=s.detail, rotate=-2.0)
    els.append(_parts_element(gauge, 600, _an(s, "gauge"), "illus:danger_gauge", importance=3, focus=(540, 450)))
    els.append(E_text("DANGEROUS?", 540, 790, _fit_cap("DANGEROUS?", 62, 640), color="yellow", anchor=_later(_an(s, "gauge"), 0.3), seed=s.seed + 3, importance=1))
    valve = build_illustration("ration_valve", 1290, 520, 760, seed=s.seed + 1, detail=s.detail, rotate=1.0)
    els.append(_parts_element(valve, 760, _an(s, "valve"), "illus:ration_valve", importance=3, impact=True, focus=(1290, 520)))
    els.append(E_text("EFFORT", 1290, 190, _fit_cap("EFFORT", 62, 520), color="chalk", anchor=_later(_an(s, "valve"), 0.2), seed=s.seed + 4, importance=1))
    fig = build_illustration("silhouette", 1650, 800, 250, seed=s.seed + 2, detail=2, rotate=0.0)
    els.append(_parts_element(fig, 250, _an(s, "person"), "illus:silhouette", importance=2, focus=(1650, 800)))
    return els, (1000, 540), 2


def frosted_score(spec: SceneSpec):
    """'And the score it assigns you? You will never see it. You cannot appeal it. You can barely even prove it exists.'

    A sealed, dashed report card sits behind a pane of frosted glass; a figure stands in front of the glass
    (the pane's frost stops at the figure: it really is in front); an eye above watches the card; a crooked
    FINAL stamp lands on the glass; the grade shows through only as a yellow question mark.
    Anchors: card ('score'), eye ('assigns'), pane ('never'), person ('see'), stamp ('appeal'), grade ('exists').
    """
    s = spec
    els: list[Element] = []
    pane_c, pane_size = (1090, 520), 780
    card_c, card_size = (1090, 540), 540
    fig_c, fig_size = (610, 690), 560
    # footprint of the figure in world px, padded a little so frost never touches its outline
    def foot(u, v):
        return (fig_c[0] + (u - 0.5) * fig_size, fig_c[1] + (v - 0.5) * fig_size)
    head = [foot(0.5 + 0.17 * math.cos(math.radians(a)) * 1.12, 0.3 + 0.2 * math.sin(math.radians(a)) * 1.12) for a in range(0, 360, 12)]
    body = [foot(0.07, 1.02), foot(0.09, 0.74)] + [foot(0.5 + 0.41 * math.cos(math.radians(a)), 0.74 + 0.23 * math.sin(math.radians(a))) for a in range(180, 361, 12)] + [foot(0.93, 1.02)]

    card = build_illustration("sealed_card", *card_c, card_size, seed=s.seed, detail=3, rotate=-1.5)
    els.append(_parts_element(card, card_size, _an(s, "card"), "illus:sealed_card", importance=3, focus=card_c))
    eye = build_illustration("eye", 1640, 190, 300, seed=s.seed + 1, detail=2, rotate=4.0)
    els.append(_parts_element(eye, 300, _an(s, "eye"), "illus:eye", importance=2, focus=(1640, 190)))
    # the watcher's line of sight to the card: two dashed lines (a drawing that is felt, not read)
    for k, (tx, ty) in enumerate(((1330, 335), (1480, 420))):
        gaze = [Part(densify([(1600, 290), (tx, ty)], 3.0), 2, "dim", 0.7, (28.0, 20.0))]
        els.append(_parts_element(gaze, 300, _later(_an(s, "eye"), 0.3), "gaze", importance=0, speed_scale=1.5))
    pane = build_illustration("frosted_pane", *pane_c, pane_size, seed=s.seed + 2, detail=3, rotate=0.5)
    pane = _occlude(pane, head, 3.0)
    pane = _occlude(pane, body, 3.0)
    els.append(_parts_element(pane, pane_size, _an(s, "pane"), "illus:frosted_pane", importance=3, focus=pane_c, speed_scale=1.15))
    fig = build_illustration("silhouette", *fig_c, fig_size, seed=s.seed + 3, detail=3, rotate=0.0)
    els.append(_parts_element(fig, fig_size, _an(s, "person"), "illus:silhouette", importance=3, focus=fig_c, speed_scale=1.2))
    stamp = build_illustration("stamp_final", 1150, 400, 380, seed=s.seed + 4, detail=3, rotate=0.0)
    els.append(_parts_element(stamp, 380, _an(s, "stamp"), "illus:stamp_final", importance=3, impact=True, focus=(1150, 400), lead=0.3))
    els.append(E_text("?", card_c[0] + 8, card_c[1] + 0.22 * card_size + 40, 130, color="yellow", anchor=_an(s, "grade"), seed=s.seed + 5, impact=True, importance=3, width=10.0))
    return els, (1000, 540), len(els) - 3


ART["rejected_judges"] = rejected_judges
ART["two_dials"] = two_dials
ART["frosted_score"] = frosted_score
