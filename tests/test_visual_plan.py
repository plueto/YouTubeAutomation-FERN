import unittest

from moduli.visual_plan import (
    build_shot_timeline,
    deterministic_visual_plan,
    parse_srt_cues,
    validate_visual_plan,
)


class VisualPlanTests(unittest.TestCase):
    def test_srt_parser_keeps_valid_times_and_ignores_malformed_blocks(self):
        cues = parse_srt_cues(
            "1\n00:00:00,000 --> 00:00:02,500\nA company decision.\n\n"
            "broken\nnot a timestamp\nignored\n\n"
            "2\n00:00:02,500 --> 00:00:04,000\nWorkers found records."
        )
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0], {"start": 0.0, "end": 2.5, "text": "A company decision."})
        self.assertEqual(cues[1]["start"], 2.5)

    def test_missing_or_malformed_plan_uses_stable_script_fallback(self):
        script = (
            "A company hid the losses. Workers found the missing records. "
            "The discovery changed the investigation."
        )
        first = validate_visual_plan(script, None)
        second = validate_visual_plan(script, [{"primary_query": "missing anchor"}])
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 1)
        self.assertTrue(all(beat["narration_anchor"] for beat in first))
        self.assertTrue(all(beat["primary_query"] for beat in first))

    def test_model_fields_are_bounded_and_optional_fields_have_defaults(self):
        beat = validate_visual_plan("A company hid the losses.", [{
            "id": "first",
            "narration_anchor": "A company hid the losses.",
            "primary_query": "company financial records",
            "duration_hint": 90,
            "music_intensity": -3,
            "alternate_queries": [" office archive ", "office archive", "records"],
            "visual_type": "unknown-source",
            "transition": "flash",
        }])[0]
        self.assertEqual(beat["duration_hint"], 12.0)
        self.assertEqual(beat["music_intensity"], 0.0)
        self.assertEqual(beat["alternate_queries"], ["office archive", "records"])
        self.assertEqual(beat["visual_type"], "illustration")
        self.assertEqual(beat["transition"], "cut")
        self.assertEqual(beat["sound_cue"], "")

    def test_anchor_alignment_preserves_story_order_and_audio_duration(self):
        script = "A company hid the losses. Workers found the missing records."
        beats = [
            {"id": "hook", "narration_anchor": "A company hid the losses.",
             "primary_query": "company financial records"},
            {"id": "reveal", "narration_anchor": "Workers found the missing records.",
             "primary_query": "workers archive documents", "dramatic_function": "reveal",
             "hold_for_reveal": True},
        ]
        events = build_shot_timeline(script, beats, 20.0)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0]["beat"]["id"], "hook")
        self.assertEqual(events[-1]["beat"]["id"], "reveal")
        self.assertAlmostEqual(events[0]["start"], 0.0)
        self.assertAlmostEqual(events[-1]["end"], 20.0)
        for previous, current in zip(events, events[1:]):
            self.assertAlmostEqual(previous["end"], current["start"])
        self.assertTrue(any(event["shot_count"] > 1 for event in events))

    def test_preview_limit_trims_timeline_without_changing_full_audio(self):
        script = "First the board approved the plan. Then investigators found evidence. Finally the company changed course."
        beats = deterministic_visual_plan(script, words_per_beat=5)
        full = build_shot_timeline(script, beats, 90.0)
        preview = build_shot_timeline(script, beats, 90.0, max_duration=35.0)
        self.assertEqual(round(full[-1]["end"], 5), 90.0)
        self.assertEqual(round(preview[-1]["end"], 5), 35.0)
        self.assertLessEqual(max(event["end"] for event in preview), 35.0)

    def test_matching_subtitle_cues_control_beat_boundary(self):
        script = "The company hid the losses. Workers found the missing records."
        beats = [
            {"narration_anchor": "The company hid the losses.", "primary_query": "company accounts"},
            {"narration_anchor": "Workers found the missing records.", "primary_query": "workers records"},
        ]
        cues = [
            {"start": 0.0, "end": 2.0, "text": "The company hid the losses."},
            {"start": 2.0, "end": 9.0, "text": "Workers found the missing records."},
        ]
        events = build_shot_timeline(script, beats, 10.0, subtitle_cues=cues)
        first_beat_end = next(event["end"] for event in events if event["beat"]["narration_anchor"].startswith("The company"))
        self.assertAlmostEqual(first_beat_end, 2.0)


if __name__ == "__main__":
    unittest.main()