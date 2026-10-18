"""Tests for the chalk engine (moduli/chalk_*.py).

Several tests are regressions for bugs found while building the engine; each docstring says which.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import wave
from pathlib import Path

import numpy as np

from moduli import chalk_illustrations as ill
from moduli import chalk_lettering as lt
from moduli import chalk_story as story
from moduli import chalk_voice as voice
from moduli.chalk_composer import LAYOUTS, compose_program, find_phrase_time, scene_words
from moduli.chalk_engine import (
    PAGE_H, PAGE_W, CamKey, ChalkEngine, Erase, Program, Stroke, densify,
)
from tools.chalk_art_demo import apply_illustration_overrides
from tools.chalk_qa import classify_checks, overall_status, qa_exit_code


def _have_ffmpeg() -> bool:
    from moduli.ffmpeg_utils import ffmpeg_path, ffprobe_path
    def available(value: str) -> bool:
        if value in {"ffmpeg", "ffprobe"}:
            return shutil.which(value) is not None
        return Path(value).is_file()
    try:
        return available(ffmpeg_path()) and available(ffprobe_path())
    except (OSError, FileNotFoundError):
        return False


# --------------------------------------------------------------------------- lettering
class LetteringTests(unittest.TestCase):
    def test_every_letter_and_digit_has_a_glyph(self):
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;-'\"()/+=%$&#*":
            self.assertIn(ch, lt.GLYPHS, ch)
            adv, strokes = lt.GLYPHS[ch]
            self.assertGreater(adv, 0)
            self.assertTrue(strokes, ch)
            for s in strokes:
                self.assertGreaterEqual(len(s), 2, ch)

    def test_lowercase_and_typographic_quotes_are_written(self):
        strokes, _ = lt.letter_strokes("Don’t stop", 0, 100, 60)
        chars = {s["char"] for s in strokes}
        self.assertIn("'", chars)
        self.assertIn("D", chars)

    def test_layout_is_deterministic_and_inside_its_box(self):
        a, bb1 = lt.letter_strokes("FOLLOW THE MONEY", 100, 300, 80, seed=3)
        b, bb2 = lt.letter_strokes("FOLLOW THE MONEY", 100, 300, 80, seed=3)
        self.assertEqual(bb1, bb2)
        self.assertTrue(all(np.array_equal(x["pts"], y["pts"]) for x, y in zip(a, b)))
        self.assertAlmostEqual(bb1[0], 100, delta=15)
        self.assertLess(bb1[3], 300 + 0.3 * 80)

    def test_wrapping_respects_max_width(self):
        lines = lt.wrap_text("ONE TWO THREE FOUR FIVE SIX SEVEN", 60, 400)
        self.assertGreater(len(lines), 1)
        for ln in lines:
            self.assertLessEqual(lt.text_width(ln, 60), 400 + 1e-6 if " " in ln else 1e9)

    def test_center_alignment_is_centered(self):
        _, bb = lt.letter_strokes("MIDDLE", 960, 500, 100, align="center", seed=1)
        self.assertAlmostEqual((bb[0] + bb[2]) / 2, 960, delta=40)


# ------------------------------------------------------------------------ illustrations
class IllustrationTests(unittest.TestCase):
    def test_every_illustration_builds_with_valid_parts(self):
        self.assertGreaterEqual(len(ill.LIBRARY), 50)
        for name, fn in ill.LIBRARY.items():
            parts = fn()
            self.assertTrue(parts, name)
            for p in parts:
                self.assertIn(p.stage, (0, 1, 2, 3), name)
                self.assertGreaterEqual(len(p.pts), 2, name)
                self.assertTrue(np.isfinite(p.pts).all(), name)
                self.assertTrue(((p.pts > -0.5) & (p.pts < 1.5)).all(), f"{name} escapes the unit box")
            self.assertTrue(any(p.stage == 0 for p in parts), f"{name} has no outline")

    def test_detail_levels_keep_the_accent(self):
        """Regression: simplifying a drawing used to drop its yellow point (the reportcard's A+)."""
        full = ill.build_illustration("reportcard", 500, 500, 400, detail=3)
        light = ill.build_illustration("reportcard", 500, 500, 400, detail=1)
        self.assertLess(len(light), len(full))
        self.assertTrue(any(p.stage == 3 for p in light))
        self.assertFalse(any(p.stage == 2 for p in light))

    def test_placement_is_deterministic_and_positioned(self):
        a = ill.build_illustration("robot", 900, 500, 300, seed=4)
        b = ill.build_illustration("robot", 900, 500, 300, seed=4)
        self.assertTrue(all(np.array_equal(x.pts, y.pts) for x, y in zip(a, b)))
        allp = np.concatenate([p.pts for p in a])
        self.assertAlmostEqual(float(allp[:, 0].mean()), 900, delta=120)

    def test_no_six_pointed_star_accents(self):
        """Artifact A-001: six-pointed accents read as a hexagram; they are refused at the source."""
        import re
        src = Path(ill.__file__).read_text(encoding="utf-8")
        self.assertEqual(re.findall(r"_star\([^)]*?,\s*6\s*,", src), [])
        with self.assertRaises(ValueError):
            ill._star(0.5, 0.5, 0.1, 6)
        for pts in (4, 5, 8):
            self.assertGreaterEqual(len(ill._star(0.5, 0.5, 0.1, pts).pts), 2 * pts)

    def test_svg_path_parser(self):
        lines = ill.svg_path_to_polylines("M0 0 L10 0 L10 10 Z")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0][0], lines[0][-1])
        rel = ill.svg_path_to_polylines("m5 5 l5 0 v5 h-5 z")
        self.assertEqual(rel[0][1], (10.0, 5.0))
        curve = ill.svg_path_to_polylines("M0 0 C 0 10 10 10 10 0")
        self.assertGreater(len(curve[0]), 8)
        quad = ill.svg_path_to_polylines("M0 0 Q 5 10 10 0 T 20 0")
        self.assertGreater(len(quad[0]), 10)

    def test_register_svg_adds_a_usable_illustration(self):
        ill.register_svg("test_triangle", ["M0 0 L100 0 L50 80 Z", ("M20 10 L80 10", 1, "yellow")], keywords=("triangulate",))
        try:
            parts = ill.build_illustration("test_triangle", 500, 500, 300)
            self.assertEqual(len(parts), 2)
            self.assertEqual({p.color for p in parts}, {"chalk", "yellow"})
            self.assertIn("test_triangle", ill.motifs_for_text("we triangulate the data", 3))
        finally:
            ill.LIBRARY.pop("test_triangle", None)
            ill.KEYWORDS.pop("test_triangle", None)

    def test_motifs_follow_the_words(self):
        self.assertEqual(ill.motifs_for_text("the money and the profit", 1), ["coins"])
        self.assertIn("iceberg", ill.motifs_for_text("what lies beneath the surface", 3))
        self.assertEqual(ill.motifs_for_text("zzzz qqqq", 3), [])


# --------------------------------------------------- scene-specific compositions (opt-in)
class SceneArtTests(unittest.TestCase):
    SCENES = {"rejected_judges": {"judge0": "teacher", "judge1": "boss", "judge2": "credit"},
              "two_dials": {"gauge": "dangerous", "valve": "effort", "person": "on you"},
              "frosted_score": {"card": "score", "eye": "assigns", "pane": "never", "person": "see", "stamp": "appeal", "grade": "exists"}}
    TEXT = {"rejected_judges": "Not by a teacher, not by a boss, not by a credit bureau.",
            "two_dials": "how dangerous you might be, and how much effort the model should spend on you.",
            "frosted_score": "And the score it assigns you? You will never see it. You cannot appeal it. You can barely even prove it exists."}

    def _plan(self, art, seconds):
        row = _row(1, 0.0, seconds, self.TEXT[art], {"layout": "hero", "title": "", "motifs": ["document"], "labels": [],
                                                      "anchors": self.SCENES[art], "art": art, "camera": "hold", "transition": "pan", "detail": 3})
        return {"duration_seconds": seconds, "scenes": [row]}

    def test_each_composition_builds_inside_the_page_and_fits_its_beat(self):
        from moduli.chalk_composer import ART
        for art, secs in (("rejected_judges", 4.5), ("two_dials", 5.1), ("frosted_score", 8.8)):
            self.assertIn(art, ART)
            prog = compose_program(self._plan(art, secs), seed=3)
            meta = prog.meta["scenes"][0]
            self.assertTrue(meta["fits"], (art, meta["pacing"]))
            self.assertLessEqual(meta["pen_speedup"], 1.5, art)
            allp = np.concatenate([s.pts for s in prog.strokes])
            self.assertGreaterEqual(float(allp[:, 0].min()), 0.04 * PAGE_W, art)
            self.assertLessEqual(float(allp[:, 0].max()), 0.96 * PAGE_W, art)
            self.assertLessEqual(float(allp[:, 1].max()), 0.96 * PAGE_H, art)
            self.assertLessEqual(max(s.t1 for s in prog.strokes), secs - 0.05, art)

    def test_compositions_are_never_chosen_automatically(self):
        """Broad application is out of scope: the director and the lexicon must not select them."""
        for name in ("gradebook", "orgchart", "score_dial", "danger_gauge", "silhouette", "ration_valve", "frosted_pane", "sealed_card", "stamp_final"):
            self.assertIn(name, ill.LIBRARY)
            self.assertNotIn(name, ill.KEYWORDS)
        rows = [_row(1, 0, 6.0, self.TEXT["two_dials"])]
        story.plan_chalk_scenes(rows)
        self.assertNotIn("art", rows[0]["chalk"])

    def test_occlusion_hides_only_what_is_behind_the_foreground(self):
        line = ill.Part(np.array([[0.0, 0.5], [1.0, 0.5]], np.float32))
        box = [(0.4, 0.3), (0.6, 0.3), (0.6, 0.7), (0.4, 0.7)]
        pieces = ill._occlude([line], box)
        self.assertEqual(len(pieces), 2)
        self.assertLess(float(pieces[0].pts[:, 0].max()), 0.41)
        self.assertGreater(float(pieces[1].pts[:, 0].min()), 0.59)
        self.assertEqual(len(ill._occlude([ill.Part(np.array([[0.0, 0.1], [1.0, 0.1]], np.float32))], box)), 1)

    def test_foreground_figure_really_hides_the_frost(self):
        from moduli.chalk_scene_art import frosted_score
        from moduli.chalk_composer import SceneSpec
        spec = SceneSpec(index=0, t0=0, t1=8.8, layout="hero", title="", motifs=["document"], labels=[], anchors={}, seed=5)
        els, _focus, _fi = frosted_score(spec)
        pane = next(e for e in els if e.tag == "illus:frosted_pane")
        fig = next(e for e in els if e.tag == "illus:silhouette")
        head_c = np.array([610 + 0 * 560, 690 + (0.3 - 0.5) * 560])      # centre of the figure's head, world px
        pts = np.concatenate([p.pts for p in pane.paths])
        self.assertEqual(int((np.hypot(pts[:, 0] - head_c[0], pts[:, 1] - head_c[1]) < 70).sum()), 0, "frost must stop at the head")
        self.assertTrue(fig.paths)


# ------------------------------------------------------------------------------ engine
def _tiny_program(seconds: float = 2.0) -> Program:
    prog = Program(world_pages=1, duration=seconds)
    pts = densify([(300, 300), (900, 340), (1500, 300)], 2.5)
    prog.strokes.append(Stroke(pts, 0.2, 1.0, "chalk", 7))
    prog.strokes.append(Stroke(densify([(300, 700), (1500, 700)], 2.5), 0.4, 0.8, "yellow", 6))
    prog.camera = [CamKey(0, PAGE_W / 2, PAGE_H / 2, 1.0)]
    return prog


class EngineTests(unittest.TestCase):
    def test_frames_have_the_requested_shape(self):
        eng = ChalkEngine(_tiny_program(), 320, 180, 12)
        f = eng.render(0.0)
        self.assertEqual(f.shape, (180, 320, 3))
        self.assertEqual(f.dtype, np.uint8)

    def test_render_is_deterministic(self):
        a = ChalkEngine(_tiny_program(), 320, 180, 12, seed=3)
        b = ChalkEngine(_tiny_program(), 320, 180, 12, seed=3)
        for i in range(12):
            self.assertTrue(np.array_equal(a.render(i / 12), b.render(i / 12)))

    def test_ink_is_revealed_progressively(self):
        eng = ChalkEngine(_tiny_program(), 640, 360, 12, dust=False)
        ink = []
        for i in range(24):
            f = eng.render(i / 12)
            ink.append(float((f.mean(axis=2) > 95).mean()))
        self.assertAlmostEqual(ink[0], 0.0, places=3)
        self.assertTrue(all(b >= a - 1e-4 for a, b in zip(ink, ink[1:])), "ink must only grow while drawing")
        self.assertGreater(ink[-1], 0.004)
        # mid-stroke the line is partly there: strictly between nothing and everything
        self.assertTrue(ink[3] < ink[-1])

    def test_chalk_is_grainy_not_a_flat_vector(self):
        eng = ChalkEngine(_tiny_program(), 1280, 720, 12, dust=False, glow=0.0, grain_noise=0)
        for i in range(24):
            f = eng.render(i / 12)
        alpha = eng.ink_a[280:320, 400:1400]
        covered = alpha[alpha > 0]
        self.assertGreater(len(covered), 100)
        self.assertGreater(float(covered.std()), 12.0, "stroke density should vary (grain)")
        self.assertLess(float((covered > 250).mean()), 0.9, "a chalk line is never perfectly solid")

    def test_dashed_stroke_lifts_the_pen_and_keeps_its_phase_across_frames(self):
        """A 60/40 px pattern painted over many frames must stay a regular pattern (rounded caps add ~7 px per dash)."""
        def runs(dash):
            prog = Program(world_pages=1, duration=2.0)
            prog.strokes.append(Stroke(densify([(200, 400), (1700, 400)], 2.5), 0.1, 1.2, "chalk", 7, dryness=0.15, dash=dash))
            prog.camera = [CamKey(0, PAGE_W / 2, PAGE_H / 2, 1.0)]
            eng = ChalkEngine(prog, 320, 180, 12, dust=False)
            t = 0.0
            while t < 1.6:                                        # many small increments, like a real render
                eng.render(t)
                t += 1 / 24
            row = eng.ink_a[398:403, 600:1700].max(axis=0) > 40  # skip the stroke's soft start
            out, cur, n = [], bool(row[0]), 0
            for v in row:
                if bool(v) == cur:
                    n += 1
                else:
                    out.append((cur, n))
                    cur, n = bool(v), 1
            return [(c, k) for c, k in out]                      # the final partial run is dropped
        dashed = runs((60.0, 40.0))
        on = [k for c, k in dashed if c]
        off = [k for c, k in dashed if not c]
        self.assertGreaterEqual(len(on), 9)
        self.assertAlmostEqual(float(np.median(on)), 67, delta=9)
        self.assertAlmostEqual(float(np.median(off)), 33, delta=9)
        self.assertLess(max(on) - min(on), 25, "dash lengths must be regular")
        solid = runs(None)
        self.assertEqual(sum(1 for c, _ in solid if not c), 0, "a solid stroke has no gaps")

    def test_erase_removes_ink_and_leaves_only_a_faint_ghost(self):
        prog = _tiny_program(4.0)
        prog.erases.append(Erase(2.0, 1.0, (0, 0, PAGE_W, PAGE_H), True, 1))
        eng = ChalkEngine(prog, 640, 360, 12, dust=False, grain_noise=0)
        before = None
        for i in range(48):
            f = eng.render(i / 12)
            if i == 22:
                before = float((f.mean(axis=2) > 95).mean())
        after = float((f.mean(axis=2) > 95).mean())
        self.assertGreater(before, 0.004)
        self.assertLess(after, before * 0.1)
        self.assertEqual(int(eng.ink_a.max()), 0)

    def test_new_strokes_survive_a_finished_erase(self):
        """Regression: an eraser that finished after the next scene started wiped its first strokes."""
        prog = _tiny_program(4.0)
        prog.erases.append(Erase(1.5, 0.8, (0, 0, PAGE_W, PAGE_H), True, 1))
        prog.strokes.append(Stroke(densify([(300, 500), (1500, 500)], 2.5), 2.4, 0.6, "chalk", 7))
        eng = ChalkEngine(prog, 640, 360, 12, dust=False)
        for i in range(48):
            eng.render(i / 12)
        self.assertGreater(int(eng.ink_a[490:510, 400:1400].max()), 100)

    def test_camera_pans_between_pages_and_never_leaves_the_board(self):
        prog = Program(world_pages=2, duration=4.0)
        prog.camera = [CamKey(0, PAGE_W / 2, PAGE_H / 2, 1.0), CamKey(1.0, PAGE_W / 2, PAGE_H / 2, 1.0),
                       CamKey(2.0, PAGE_W * 1.5, PAGE_H / 2, 1.0)]
        eng = ChalkEngine(prog, 320, 180, 12)
        x_start = eng._view(0.5)[0]
        x_end = eng._view(3.5)[0]
        self.assertAlmostEqual(x_start, 0, delta=20)
        self.assertAlmostEqual(x_end, PAGE_W, delta=20)
        for t in np.linspace(0, 4, 17):
            x0, y0, s = eng._view(float(t))
            self.assertGreaterEqual(x0, 0)
            self.assertLessEqual(x0 + 320 / s, prog.world_w + 1e-3)

    def test_fast_pans_are_motion_blurred_and_static_frames_are_untouched(self):
        """Artifact A-005: pans moved 10-21% of the frame per frame with razor-sharp lines (strobing)."""
        def program():
            prog = Program(world_pages=2, duration=4.0)
            for x in range(200, 3700, 180):                      # lots of vertical chalk lines to smear
                prog.strokes.append(Stroke(densify([(x, 200), (x, 900)], 2.5), 0.0, 0.2, "chalk", 6, dryness=0.2))
            prog.camera = [CamKey(0, PAGE_W / 2, PAGE_H / 2, 1.0), CamKey(1.5, PAGE_W / 2, PAGE_H / 2, 1.0),
                           CamKey(2.1, PAGE_W * 1.5, PAGE_H / 2, 1.0)]
            return prog

        def sharpness(frame):
            g = frame.mean(axis=2).astype(np.float32)
            return float(np.abs(np.diff(g, axis=1)).mean())

        blur = ChalkEngine(program(), 640, 360, 24, seed=2, grain_noise=0)
        sharp = ChalkEngine(program(), 640, 360, 24, seed=2, grain_noise=0, motion_blur=False)
        frames = {}
        for i in range(60):
            t = i / 24
            a, b = blur.render(t), sharp.render(t)
            frames[i] = (a, b)
        self.assertTrue(np.array_equal(*frames[12]), "a still camera must render identically with or without blur")
        mid = 1.8                                                # the fastest part of the pan
        a, b = frames[int(mid * 24)]
        self.assertLess(sharpness(a), sharpness(b) * 0.75, "moving frames should be visibly smoothed")
        self.assertGreater(float(a.mean()), 8.0, "blur must not black out the frame")

    def test_dust_particles_fall_and_expire(self):
        eng = ChalkEngine(_tiny_program(4.0), 320, 180, 12)
        for i in range(12):
            eng.render(i / 12)
        self.assertGreater(len(eng._particles), 0)
        for i in range(12, 60):
            eng.render(i / 12)
        self.assertEqual(len(eng._particles), 0)


# ---------------------------------------------------------------------------- planning
def _row(i, t0, t1, text, chalk=None):
    row = {"scene": i, "start_seconds": t0, "end_seconds": t1, "narration_excerpt": text}
    if chalk is not None:
        row["chalk"] = chalk
    return row


class StoryTests(unittest.TestCase):
    SCRIPT = ("You are being graded right now. Not by a teacher, not by a boss, not by a credit bureau. "
              "By the large language model you opened five minutes ago to help draft an email. "
              "Every prompt you type, every follow-up question, every frustrated correction feeds a silent calculation. "
              "This isn't science fiction. And the score it assigns you? You will never see it.")

    def test_beats_follow_sentences_and_keep_the_hook_alone(self):
        beats = story.segment_beats(self.SCRIPT, 18)
        self.assertEqual(" ".join(beats).split(), self.SCRIPT.split())
        self.assertEqual(" ".join(beats).casefold(), " ".join(self.SCRIPT.split()).casefold())
        self.assertEqual(beats[0], "You are being graded right now.")
        for b in beats:
            self.assertLessEqual(len(b.split()), 28)

    def test_scene_plan_falls_back_if_story_segmentation_drops_script_text(self):
        from moduli.chalkboard_director import build_scene_plan

        with patch.object(story, "segment_beats", return_value=["You are being graded"]):
            plan = build_scene_plan(
                self.SCRIPT, 20.0, target_seconds=7.5, use_tts_timestamps=False,
            )
        reconstructed = " ".join(row["narration_excerpt"] for row in plan["scenes"])
        self.assertEqual(reconstructed.casefold(), " ".join(self.SCRIPT.split()).casefold())

    def test_enumerations_and_numbers_are_detected(self):
        self.assertEqual([i.lower() for i in story.list_items("Not by a teacher, not by a boss, not by a credit bureau.")],
                         ["teacher", "boss", "credit bureau"])
        self.assertEqual(story.list_items("It was a dark and stormy night, and nothing happened."), [])
        self.assertEqual(story.find_number("opened five minutes ago")[0], "5 MINUTES")
        self.assertEqual(story.find_number("lost $40 million overnight")[0].replace(" ", ""), "$40MILLION")
        self.assertIsNone(story.find_number("nothing numeric here"))

    def test_captions_are_short_noun_phrases(self):
        caps = story.captions("It all feeds a silent calculation that decides how much effort the model should spend", 3)
        self.assertTrue(caps)
        for c in caps:
            self.assertLessEqual(len(c.split()), 3)
            self.assertNotIn(c.split()[0].casefold(), {"it", "all", "a", "that", "how", "much", "the"})

    def test_captions_do_not_bridge_sentences_or_repeat_terms(self):
        text = "A preference becomes feedback. Feedback changes the system again."
        captions = story.captions(text, 3)
        self.assertNotIn("FEEDBACK FEEDBACK", captions)
        self.assertEqual(len({caption.casefold() for caption in captions}), len(captions))

        # The final word before a period must not combine with the next
        # sentence's first word to create a false visual concept.
        captions = story.captions("The hidden score appears. The system changes the result.", 6)
        self.assertNotIn("APPEARS SYSTEM", captions)
        self.assertEqual(len({caption.casefold() for caption in captions}), len(captions))

    def test_plan_has_valid_varied_layouts(self):
        rows = [_row(i + 1, i * 6.0, (i + 1) * 6.0, t) for i, t in enumerate(story.segment_beats(self.SCRIPT, 18))]
        story.plan_chalk_scenes(rows)
        layouts = [r["chalk"]["layout"] for r in rows]
        for r in rows:
            ch = r["chalk"]
            self.assertIn(ch["layout"], LAYOUTS)
            self.assertTrue(all(m in ill.LIBRARY for m in ch["motifs"]))
            self.assertTrue(ch["motifs"])
        self.assertTrue(all(a != b for a, b in zip(layouts, layouts[1:])), layouts)

    def test_cue_driven_layouts_need_their_cue(self):
        """Regression: 'bignumber' was chosen for a beat that contained no number."""
        rows = [_row(1, 0, 3.0, "You can barely even prove it exists.")]
        story.plan_chalk_scenes(rows)
        self.assertNotEqual(rows[0]["chalk"]["layout"], "bignumber")

    def test_statement_text_is_short_enough_to_write(self):
        self.assertLessEqual(len(story.shorten_statement("You are being graded right now.", 14)), 14)
        self.assertEqual(story.shorten_statement("This isn't science fiction.", 24), "NOT SCIENCE FICTION")

    def test_ai_chalk_object_is_validated_field_by_field(self):
        fallback = {"layout": "hero", "motifs": ["robot"], "labels": ["MODEL"], "title": "", "anchors": {"motif0": "model"}}
        text = "the model decides everything"
        out = story.sanitize_chalk({"layout": "nonsense", "motifs": ["robot", "dragon"], "labels": ["WAY TOO MANY WORDS IN THIS ONE", "OK"],
                                    "anchors": {"motif0": "model", "label0": "absent", "bogus": "model"}, "camera": "spin"}, text, fallback)
        self.assertEqual(out["layout"], "hero")
        self.assertEqual(out["motifs"], ["robot"])
        self.assertEqual(out["labels"], ["OK"])
        self.assertEqual(out["anchors"], {"motif0": "model"})
        self.assertNotIn("camera", {k for k in out if out[k] == "spin"})
        self.assertEqual(story.sanitize_chalk("garbage", text, fallback), fallback)


class ComposerTests(unittest.TestCase):
    def _plan(self):
        text = "You are being graded right now. Not by a teacher, not by a boss, not by a credit bureau. By the large language model you opened five minutes ago."
        beats = story.segment_beats(text, 18)
        rows, t = [], 0.0
        for i, b in enumerate(beats):
            d = max(2.0, len(b.split()) * 0.42)
            rows.append(_row(i + 1, round(t, 3), round(t + d, 3), b))
            t += d
        story.plan_chalk_scenes(rows)
        rows[1]["chalk"]["transition"] = "erase"
        return {"duration_seconds": t, "scenes": rows}

    def test_program_invariants(self):
        plan = self._plan()
        prog = compose_program(plan, seed=1)
        self.assertTrue(prog.strokes)
        dur = plan["duration_seconds"]
        for st in prog.strokes:
            self.assertGreaterEqual(st.t0, 0.0)
            self.assertLessEqual(st.t1, dur + 3.0)
            self.assertTrue(np.isfinite(st.pts).all())
            self.assertGreaterEqual(float(st.pts[:, 0].min()), -5)
            self.assertLessEqual(float(st.pts[:, 0].max()), prog.world_w + 5)
        # each scene draws only inside its own time window
        for meta, row in zip(prog.meta["scenes"], plan["scenes"]):
            self.assertGreaterEqual(meta["first_stroke"], row["start_seconds"] - 1e-6)
        ts = [k.t for k in prog.camera]
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])), "camera keys must be strictly increasing")

    def test_erase_finishes_before_the_next_scene_draws(self):
        """Regression: eraser finished after the next scene's first strokes and deleted them."""
        plan = self._plan()
        prog = compose_program(plan, seed=1)
        visible = [e for e in prog.erases if e.visible]
        self.assertTrue(visible)
        for e in visible:
            later = [s.t0 for s in prog.strokes if s.t0 >= e.t0 + e.dur - 1e-6 and s.pts[:, 0].min() >= e.rect[0]
                     and s.pts[:, 0].max() <= e.rect[2]]
            started_during = [s for s in prog.strokes if e.t0 < s.t0 < e.t0 + e.dur and s.pts[:, 0].min() >= e.rect[0]
                              and s.pts[:, 0].max() <= e.rect[2] and s.t0 > plan["scenes"][1]["start_seconds"]]
            self.assertFalse(started_during, "strokes begin while the board is still being wiped")

    def test_pages_are_cleaned_before_reuse(self):
        plan = self._plan()
        prog = compose_program(plan, seed=1)
        pages_used = [m["page"] for m in prog.meta["scenes"]]
        for i, p in enumerate(pages_used):
            if p in pages_used[:i] and plan["scenes"][i]["chalk"]["transition"] == "pan":
                self.assertTrue(any(not e.visible and e.rect[0] == p * PAGE_W for e in prog.erases), (i, p))

    def test_anchored_elements_start_near_their_word(self):
        plan = self._plan()
        row = plan["scenes"][1]
        row["chalk"]["anchors"] = {"label2": "bureau"}
        words, _ = scene_words(row)
        t_word = find_phrase_time(words, "bureau")
        self.assertIsNotNone(t_word)
        self.assertGreaterEqual(t_word, row["start_seconds"])
        self.assertLessEqual(t_word, row["end_seconds"])

    def test_captions_are_not_blocked_by_unanchored_drawings(self):
        """Artifact A-002: a big unanchored drawing placed first by the layout delayed every caption by seconds."""
        from moduli.chalk_composer import Element, Path as CPath, _order

        def el(tag, anchor, secs):
            pts = np.stack([np.linspace(0, 1000, 60), np.zeros(60)], axis=1).astype(np.float32)
            return Element("illus", [CPath(pts, speed=1000 / secs)], anchor, 0.3, 0.0, False, tag)

        big, cap, later = el("big", None, 2.0), el("cap", 1.0, 0.5), el("later", 4.0, 0.5)
        self.assertGreater(big.nominal_duration(), 1.5)          # fixture sanity: it cannot fit before the word
        self.assertEqual([e.tag for e in _order([big, cap, later], 0.0, 1.0)], ["cap", "big", "later"])
        # a short filler that does fit before the first word is still drawn first
        tiny = el("tiny", None, 0.2)
        self.assertEqual([e.tag for e in _order([tiny, cap], 0.0, 1.0)], ["tiny", "cap"])
        # main content (importance 3) is not deferred behind decoration (importance 1) anchored to a late word
        title = el("title", None, 1.2)
        title.importance = 3
        deco = el("deco", 1.0, 0.5)             # deadline 0.7 s: the 1.2 s title cannot fit before it
        deco.importance = 1
        self.assertEqual([e.tag for e in _order([title, deco], 0.0, 1.0)], ["title", "deco"])
        # anchored elements keep spoken order whatever the layout order was
        self.assertEqual([e.tag for e in _order([later, cap], 0.0, 1.0)], ["cap", "later"])

    def test_close_up_never_crops_the_scene(self):
        """Regression: a close-up pushed in so far that captions and half a drawing were cut off."""
        plan = self._plan()
        for r in plan["scenes"]:
            r["chalk"]["camera"] = "closeup"
        prog = compose_program(plan, seed=2)
        eng = ChalkEngine(prog, 320, 180, 12)
        for meta, row in zip(prog.meta["scenes"], plan["scenes"]):
            sc = [s for s in prog.strokes if row["start_seconds"] <= s.t0 < row["end_seconds"]]
            if not sc:
                continue
            allp = np.concatenate([s.pts for s in sc])
            t = row["end_seconds"] - 0.5
            x0, y0, s = eng._view(t)
            if abs(eng.camera_at(t)[2] - 1.0) < 1e-3:
                continue
            self.assertLessEqual(x0, float(allp[:, 0].min()) + 60)
            self.assertGreaterEqual(x0 + 320 / s, float(allp[:, 0].max()) - 60)

    def test_short_statement_cards_become_text_only_instead_of_hurrying_the_pen(self):
        """A 2 s statement card cannot hold a title *and* an icon; text-only at a natural pen beats a hurried pen."""
        rows = [_row(1, 0.0, 2.1, "You are being graded right now.")]
        story.plan_chalk_scenes(rows)
        self.assertEqual(rows[0]["chalk"]["layout"], "statement")
        prog = compose_program({"duration_seconds": 2.1, "scenes": rows})
        meta = prog.meta["scenes"][0]
        self.assertLessEqual(meta["pen_speedup"], 1.3)
        self.assertTrue(all(s.tag.startswith("text:") for s in prog.strokes), "icon should have been dropped")
        self.assertTrue(meta["fits"])

    def test_short_beats_lose_detail_not_sync(self):
        rows = [_row(1, 0.0, 2.2, "This isn't science fiction.")]
        story.plan_chalk_scenes(rows)
        prog = compose_program({"duration_seconds": 2.2, "scenes": rows})
        last = max(s.t1 for s in prog.strokes)
        self.assertLessEqual(last, 2.2 + 0.05)


# ----------------------------------------------------------------------- voice + audio
class VoiceTests(unittest.TestCase):
    def test_alignment_survives_numbers_and_contractions(self):
        script = "It cost $40 million, didn't it? Yes."
        side = {"source": "t", "precision": "word", "words": [
            {"w": "It", "s": 0.0, "e": 0.2}, {"w": "cost", "s": 0.2, "e": 0.5},
            {"w": "forty", "s": 0.5, "e": 0.8}, {"w": "million", "s": 0.8, "e": 1.2}, {"w": "dollars", "s": 1.2, "e": 1.6},
            {"w": "didn't", "s": 1.7, "e": 2.0}, {"w": "it", "s": 2.0, "e": 2.2}, {"w": "Yes", "s": 2.6, "e": 3.0}]}
        res = voice.align_to_script(side, script, 3.2)
        self.assertTrue(res["ok"])
        times = res["word_timings"]
        self.assertEqual(len(times), len(script.split()))
        self.assertTrue(all(b[0] >= a[0] - 1e-9 for a, b in zip(times, times[1:])), "timings must be monotonic")
        self.assertAlmostEqual(times[-1][0], 2.6, places=2)
        self.assertEqual(res["timing_source"], "tts_word_boundaries")

    def test_sidecar_roundtrip_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "n.mp3"
            audio.write_bytes(b"x")
            self.assertIsNone(voice.read_sidecar(audio))
            voice.write_sidecar(audio, [{"w": "hi", "s": 0.0, "e": 0.3}], source="unit", precision="word")
            self.assertEqual(voice.read_sidecar(audio)["words"][0]["w"], "hi")
            self.assertIsNone(voice.sidecar_alignment(audio, "completely different words entirely here", 5.0))
            voice.sidecar_path(audio).write_text("{not json")
            self.assertIsNone(voice.read_sidecar(audio))

    def test_edge_tts_worker_writes_word_timestamps(self):
        """The real pipeline captures Edge TTS WordBoundary events; verify with a stand-in edge_tts."""
        from moduli import audio

        class FakeComm:
            def __init__(self, text, voice_, boundary="SentenceBoundary"):
                self.boundary = boundary

            async def stream(self):
                yield {"type": "audio", "data": b"ID3fake"}
                yield {"type": "WordBoundary", "offset": 5_000_000, "duration": 3_000_000, "text": "Hello"}
                yield {"type": "WordBoundary", "offset": 9_000_000, "duration": 4_000_000, "text": "world"}

        fake = types.ModuleType("edge_tts")
        fake.Communicate = FakeComm
        old = sys.modules.get("edge_tts")
        sys.modules["edge_tts"] = fake
        try:
            with tempfile.TemporaryDirectory() as d:
                out = str(Path(d) / "c.mp3")
                audio._tts_worker("Hello world", out, "en-US-GuyNeural")
                words = json.loads(Path(out + ".words.json").read_text())
                self.assertEqual([w["w"] for w in words], ["Hello", "world"])
                self.assertAlmostEqual(words[0]["s"], 0.5, places=3)
                self.assertAlmostEqual(words[1]["e"], 1.3, places=3)
                self.assertEqual(Path(out).read_bytes(), b"ID3fake")
        finally:
            if old is not None:
                sys.modules["edge_tts"] = old
            else:
                sys.modules.pop("edge_tts", None)

    @unittest.skipUnless(voice.espeak_available(), "espeak-ng/ffmpeg not installed")
    def test_local_synthesis_writes_audio_and_timings(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "n.mp3"
            info = voice.synthesize_espeak("This is a test. It has two sentences, and a comma.", out)
            self.assertGreater(info["duration"], 2.0)
            side = voice.read_sidecar(out)
            self.assertEqual(side["precision"], "phrase")
            self.assertEqual(len(side["words"]), 11)
            ends = [w["e"] for w in side["words"]]
            self.assertAlmostEqual(ends[-1], info["duration"], delta=0.8)


class DirectorIntegrationTests(unittest.TestCase):
    SCRIPT = ("You are being graded right now. Not by a teacher, not by a boss, not by a credit bureau. "
              "By the large language model you opened five minutes ago to help draft an email.")

    def test_plan_uses_sidecar_word_times_and_carries_chalk_plans(self):
        from moduli.chalkboard_director import build_scene_plan
        words = self.SCRIPT.split()
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "n.mp3"
            audio.write_bytes(b"x")
            voice.write_sidecar(audio, [{"w": w, "s": round(i * 0.4, 3), "e": round(i * 0.4 + 0.35, 3)} for i, w in enumerate(words)],
                                source="unit", precision="word")
            plan = build_scene_plan(self.SCRIPT, len(words) * 0.4 + 0.5, audio_path=str(audio), target_seconds=7.5)
        self.assertEqual(plan["timing_source"], "tts_word_boundaries")
        for row in plan["scenes"]:
            self.assertIn("chalk", row)
            self.assertEqual(len(row["word_times"]), len(row["narration_excerpt"].split()))
            self.assertAlmostEqual(row["word_times"][0][0], row["start_seconds"], delta=0.45)

    def test_legacy_beats_are_unchanged_without_a_target(self):
        from moduli.chalkboard_director import build_scene_plan
        plan = build_scene_plan(self.SCRIPT, 20.0)
        self.assertEqual(plan["planning_target_seconds"], 4.2)
        self.assertEqual(" ".join(r["narration_excerpt"] for r in plan["scenes"]).split(), self.SCRIPT.split())


# --------------------------------------------------------------------------------- sound
class SoundTests(unittest.TestCase):
    def test_effects_follow_the_drawing_and_are_finite(self):
        from moduli import chalk_sound as cs
        prog = _tiny_program(3.0)
        prog.cues = [{"t": 1.2, "kind": "impact", "gain": 1.0}, {"t": 1.8, "kind": "erase", "dur": 0.8},
                     {"t": 2.2, "kind": "whoosh", "dur": 0.6, "gain": 0.8}]
        sfx, rep = cs.render_sfx(prog, 3.0)
        self.assertEqual(len(sfx), 3 * cs.SR)
        self.assertTrue(np.isfinite(sfx).all(), "regression: sin() ** fractional produced NaN at the shape edge")
        loud = np.abs(sfx)
        self.assertGreater(float(loud[int(0.5 * cs.SR):int(1.0 * cs.SR)].mean()), float(loud[int(2.9 * cs.SR):].mean()) * 3)
        self.assertEqual(rep["impacts"], 1)

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not installed")
    def test_soundtrack_mix_is_clean(self):
        from moduli import chalk_sound as cs
        with tempfile.TemporaryDirectory() as d:
            narr = np.sin(2 * np.pi * 220 * np.arange(int(3 * cs.SR)) / cs.SR).astype(np.float32) * 0.3
            narr[int(1.5 * cs.SR):int(2.0 * cs.SR)] = 0
            cs.write_wav(Path(d) / "narr.wav", narr)
            rep = cs.build_soundtrack(_tiny_program(3.0), Path(d) / "narr.wav", None, 3.0, Path(d) / "mix.wav")
            with wave.open(str(Path(d) / "mix.wav")) as w:
                data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
            self.assertEqual(len(data), 3 * cs.SR)
            self.assertLess(int(np.abs(data).max()), 32767)
            self.assertTrue(math.isfinite(rep["sfx_rms_vs_voice"]))
            self.assertLess(rep["sfx_rms_vs_voice"], 0.5, "effects must sit under the narrator")


# ------------------------------------------------------------ sound stems (tuning pass 1)
GOLDEN_SFX = {"len": 176400, "peak": 1.5905708074569702, "window_energy": [144.0740654606755, 663.1351439024963, 682.8821693342309, 679.00758827799, 1226.9534809199213, 628.9660840054911, 0.6784715511179321, 869.9152532495273, 1196.8319233952852, 357.1601927927975, 189.19367156413696, 72.56208407181656, 304.3427761337052, 1.0381837927381166, 9.879656820437938e-10, 1.1978663530968765e-09], "report": {"chalk_strokes": 3, "chalk_seconds": 2.3, "erase_events": 1, "impacts": 2, "whooshes": 1}}


def _golden_program() -> Program:
    prog = Program(world_pages=1, duration=4.0)
    prog.strokes = [Stroke(densify([(300, 300), (900, 340), (1500, 300)], 2.5), 0.2, 1.0, "chalk", 7),
                    Stroke(densify([(300, 700), (1500, 700)], 2.5), 0.6, 0.8, "yellow", 6),
                    Stroke(densify([(300, 500), (1500, 520)], 2.5), 1.8, 0.5, "chalk", 5)]
    prog.cues = [{"t": 1.2, "kind": "impact", "gain": 1.0}, {"t": 1.8, "kind": "erase", "dur": 0.8},
                 {"t": 2.4, "kind": "whoosh", "dur": 0.6, "gain": 0.8}, {"t": 3.0, "kind": "impact", "gain": 0.6}]
    return prog


class SoundStemTests(unittest.TestCase):
    def test_default_gains_reproduce_the_v1_effects_track(self):
        """Per-effect stems with 0 dB offsets must equal the v1 single-track output (frozen from commit 6adc4d9)."""
        from moduli import chalk_sound as cs
        sfx, rep = cs.render_sfx(_golden_program(), 4.0)
        self.assertEqual(len(sfx), GOLDEN_SFX["len"])
        self.assertAlmostEqual(float(np.abs(sfx).max()), GOLDEN_SFX["peak"], places=5)
        windows = [float((x.astype(np.float64) ** 2).sum()) for x in np.array_split(sfx.astype(np.float64), 16)]
        for got, want in zip(windows, GOLDEN_SFX["window_energy"]):
            self.assertAlmostEqual(got, want, delta=max(abs(want) * 1e-6, 1e-6))
        self.assertEqual(rep, GOLDEN_SFX["report"])

    def test_gains_are_db_offsets_on_individual_stems(self):
        from moduli import chalk_sound as cs
        stems, _ = cs.render_sfx_stems(_golden_program(), 4.0)
        g = cs.SfxGains(chalk=-6.0, tap=-3.0)
        self.assertAlmostEqual(g.linear("chalk"), 0.5012, places=3)
        quiet = cs.sum_stems(stems, g)
        loud = cs.sum_stems(stems)
        # only the chalk stem and the tap changed; the eraser stretch (1.8-2.6 s) is untouched
        a, b = int(2.0 * cs.SR), int(2.4 * cs.SR)
        mask = (np.abs(stems["chalk"][a:b]) < 1e-9) & (np.abs(stems["tap"][a:b]) < 1e-9)
        self.assertTrue(np.allclose(quiet[a:b][mask], loud[a:b][mask]))
        self.assertLess(float(np.abs(quiet).max()), float(np.abs(loud).max()))

    def test_shipped_default_is_minus_6_db_and_never_boosts(self):
        """Adopted from docs/qa/sound_lab_1 (B2); FERN_CHALK_SFX_DB may lower it further but can never raise it."""
        from moduli import chalk_sound as cs
        old = os.environ.pop("FERN_CHALK_SFX_DB", None)
        try:
            g = cs.default_gains()
            for k in cs.STEM_ORDER:
                self.assertAlmostEqual(g.as_dict()[k], -7.0 if k in ("tick", "tap") else -6.0, places=9, msg=k)
            self.assertEqual(g.music, 0.0)
            os.environ["FERN_CHALK_SFX_DB"] = "0"
            self.assertEqual(cs.default_gains().chalk, 0.0)
            os.environ["FERN_CHALK_SFX_DB"] = "+4"
            self.assertEqual(cs.default_gains().chalk, 0.0, "a positive value must not boost an effect")
            os.environ["FERN_CHALK_SFX_DB"] = "-9"
            self.assertEqual(cs.default_gains().tap, -9.0)
            os.environ["FERN_CHALK_SFX_DB"] = "nonsense"
            self.assertEqual(cs.default_gains().tap, -7.0)             # falls back to the shipped default
            self.assertEqual(cs.default_gains().chalk, -6.0)
        finally:
            os.environ.pop("FERN_CHALK_SFX_DB", None)
            if old is not None:
                os.environ["FERN_CHALK_SFX_DB"] = old

    @unittest.skipUnless(_have_ffmpeg(), "ffmpeg not installed")
    def test_gain_offsets_are_not_undone_by_the_reference_normalisation(self):
        """Regression guard for the experiment design: -6 dB on an effect must really lower the mix."""
        from moduli import chalk_sound as cs
        with tempfile.TemporaryDirectory() as d:
            t = np.arange(int(4 * cs.SR)) / cs.SR
            narr = (0.25 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
            cs.write_wav(Path(d) / "n.wav", narr)
            base = cs.build_soundtrack(_golden_program(), Path(d) / "n.wav", None, 4.0, Path(d) / "a.wav", gains=cs.SfxGains())
            quiet = cs.build_soundtrack(_golden_program(), Path(d) / "n.wav", None, 4.0, Path(d) / "b.wav",
                                        gains=cs.SfxGains(chalk=-6, body=-6, tick=-6, erase=-6, whoosh=-6))
            self.assertLess(quiet["sfx_rms_vs_voice"], base["sfx_rms_vs_voice"] * 0.85)
            self.assertEqual(quiet["gains_db"]["chalk"], -6)
            self.assertEqual(base["gains_db"]["chalk"], 0)


# --------------------------------------------------------------------------- end to end
@unittest.skipUnless(voice.espeak_available(), "espeak-ng/ffmpeg not installed")
class EndToEndTests(unittest.TestCase):
    def test_small_render_has_audio_video_and_a_clean_report(self):
        from moduli.chalk_render import render_chalk_video
        script = "You are being graded right now. Not by a teacher, not by a boss. The score is invisible."
        with tempfile.TemporaryDirectory() as d:
            audio = Path(d) / "n.mp3"
            info = voice.synthesize_espeak(script, audio)
            out = Path(d) / "v.mp4"
            rep = render_chalk_video(str(audio), script, str(out), width=320, height=180, fps=12)
            self.assertTrue(out.is_file() and out.stat().st_size > 5000)
            from moduli.ffmpeg_utils import ffprobe_path
            probe_result = subprocess.run(
                [ffprobe_path(), "-v", "error", "-show_entries",
                 "stream=codec_type,width,height:format=duration", "-of", "json", str(out)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            probe = json.loads(probe_result.stdout)
            kinds = {s["codec_type"] for s in probe["streams"]}
            self.assertEqual(kinds, {"video", "audio"})
            self.assertAlmostEqual(float(probe["format"]["duration"]), info["duration"], delta=0.35)
            self.assertTrue(math.isfinite(rep["sound"]["peak"]))
            self.assertTrue(all(s["fits"] for s in rep["scenes"]), rep["scenes"])
            self.assertTrue(out.with_suffix(".scene_plan.json").is_file())


# --------------------------------------------------------------------------- integration regression tests for the review pass
class IllustrationDemoIntegrationTests(unittest.TestCase):
    @staticmethod
    def _row(start, end, excerpt):
        return {
            "start_seconds": float(start),
            "end_seconds": float(end),
            "narration_excerpt": excerpt,
            "word_count": len(excerpt.split()),
            "chalk": {"layout": "duo", "title": "", "labels": ["GENERIC"]},
        }

    def test_demo_selects_anchored_speech_windows_not_fixed_scene_numbers(self):
        plan = {
            "duration_seconds": 31.0,
            "scenes": [
                self._row(0, 2, "Your assistant is judging you, not by a teacher."),
                self._row(2, 8, "Not by a boss, and not by a credit bureau."),
                self._row(8, 16, "It estimates how dangerous you might be and how much effort the model should spend on you."),
                self._row(16, 24, "And the score it assigns you; you will never see it."),
                self._row(24, 31, "You cannot appeal it; you can barely prove it exists."),
            ],
        }
        placements = apply_illustration_overrides(plan)
        by_art = {p["art"]: p for p in placements}
        self.assertEqual(set(by_art), {"rejected_judges", "two_dials", "frosted_score"})
        self.assertEqual(by_art["rejected_judges"]["source_scenes"], [1, 2])
        self.assertEqual(by_art["two_dials"]["source_scenes"], [3])
        self.assertEqual(by_art["two_dials"]["anchors"]["person"], "on you")
        self.assertEqual(by_art["frosted_score"]["source_scenes"], [4, 5])
        self.assertEqual(plan["scene_count"], len(plan["scenes"]))

        art_by_excerpt = {
            row.get("illustration_demo_art"): row
            for row in plan["scenes"] if row.get("illustration_demo_art")
        }
        self.assertEqual(set(art_by_excerpt), set(by_art))
        for name, row in art_by_excerpt.items():
            excerpt = row["narration_excerpt"].casefold()
            self.assertIn("art", row["chalk"])
            self.assertEqual(row["chalk"]["art"], name)
            for phrase in row["chalk"]["anchors"].values():
                self.assertIn(phrase.casefold(), excerpt)
        self.assertEqual(" ".join(r["narration_excerpt"] for r in plan["scenes"]).split(),
                         " ".join([
                             "Your assistant is judging you, not by a teacher.",
                             "Not by a boss, and not by a credit bureau.",
                             "It estimates how dangerous you might be and how much effort the model should spend on you.",
                             "And the score it assigns you; you will never see it.",
                             "You cannot appeal it; you can barely prove it exists.",
                         ]).split())

        # Exercise the actual composer too: each opt-in art must resolve to its registered builder.
        program = compose_program(plan, seed=11)
        self.assertGreater(len(program.strokes), 0)
        emitted_tags = {stroke.tag for stroke in program.strokes}
        self.assertIn("illus:gradebook", emitted_tags)
        self.assertIn("illus:danger_gauge", emitted_tags)
        self.assertIn("illus:sealed_card", emitted_tags)

    def test_demo_fails_closed_if_anchor_phrases_are_missing(self):
        plan = {
            "duration_seconds": 4.0,
            "scenes": [self._row(0, 4, "A generic line with none of the requested concepts.")],
        }
        with self.assertRaisesRegex(ValueError, "rejected_judges"):
            apply_illustration_overrides(plan)

    def test_demo_does_not_modify_plan_when_anchor_selection_fails(self):
        row = self._row(0, 4, "A generic line with none of the requested concepts.")
        plan = {"duration_seconds": 4.0, "scenes": [row]}
        original = json.loads(json.dumps(plan))
        with self.assertRaises(ValueError):
            apply_illustration_overrides(plan)
        self.assertEqual(plan, original)


class QaClassificationTests(unittest.TestCase):
    def test_blocked_checks_are_never_pass_and_strict_mode_rejects_them(self):
        catalog = [
            ("content", "every scene has visible content", False, "no scene plan"),
            ("audio_present", "video has audio", True, ""),
        ]
        checks = classify_checks(catalog, [])
        self.assertEqual([c["status"] for c in checks], ["BLOCKED", "PASS"])
        self.assertEqual(overall_status(checks), "BLOCKED")
        self.assertEqual(qa_exit_code(checks, strict=False), 0)
        self.assertEqual(qa_exit_code(checks, strict=True), 1)

    def test_fail_always_fails_even_when_not_strict(self):
        catalog = [("sfx", "effects are finite", True, "")]
        checks = classify_checks(catalog, [("FAIL", "sfx", "ratio is infinite")])
        self.assertEqual(checks[0]["status"], "FAIL")
        self.assertEqual(qa_exit_code(checks, strict=False), 1)
        self.assertEqual(qa_exit_code(checks, strict=True), 1)

    def test_warning_is_not_a_pass_or_a_failure(self):
        catalog = [("audio_len", "audio matches video length", True, "")]
        checks = classify_checks(catalog, [("WARNING", "audio_len", "duration differs")])
        self.assertEqual(checks[0]["status"], "WARNING")
        self.assertEqual(overall_status(checks), "WARNING")
        self.assertEqual(qa_exit_code(checks, strict=False), 0)


if __name__ == "__main__":
    unittest.main()
