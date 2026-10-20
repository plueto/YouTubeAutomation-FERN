import json
import tempfile
import unittest
from pathlib import Path

from moduli.credible_footage import (
    MANIFEST_SCHEMA,
    apply_footage_manifest,
    script_fingerprint,
    select_footage_for_scenes,
)


class CredibleFootageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.clip_a = self.root / "openai_launch.mp4"
        self.clip_a.write_bytes(b"fake video bytes for selection-only test")
        self.clip_b = self.root / "unrelated.mp4"
        self.clip_b.write_bytes(b"fake video bytes for selection-only test")
        self.manifest = self.root / "footage_candidates.json"

    def candidate(self, **overrides):
        candidate = {
            "title": "OpenAI GPT-5 product launch demo",
            "description": "Official product demonstration and launch footage",
            "channel_title": "OpenAI",
            "source_url": "https://www.youtube.com/watch?v=example123",
            "path": str(self.clip_a),
            "source_type": "product_demo",
            "source_verified": True,
            "verification_method": "official_channel",
            "license": "CC BY",
            "attribution": "OpenAI",
            "review_status": "approved",
        }
        candidate.update(overrides)
        return candidate

    def test_selects_relevant_licensed_primary_footage(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "GPT-5 launch",
                "labels": ["OpenAI product", "live demonstration"],
                "narration_excerpt": "OpenAI reveals the GPT-5 product in a live launch demonstration.",
            }],
            [self.candidate()],
            self.manifest,
        )
        self.assertEqual(report["scenes_with_footage"], 1)
        self.assertEqual(scenes[0]["footage"]["path"], str(self.clip_a.resolve()))
        self.assertEqual(scenes[0]["footage"]["source_type"], "product_demo")

    def test_rejects_uncleared_rights_even_if_source_is_official(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "GPT-5 launch",
                "labels": ["OpenAI product", "live demonstration"],
                "narration_excerpt": "OpenAI reveals the GPT-5 product in a live launch demonstration.",
            }],
            [self.candidate(license="YouTube Standard License")],
            self.manifest,
        )
        self.assertNotIn("footage", scenes[0])
        self.assertEqual(report["scenes_chalk_fallback"], 1)
        self.assertIn("license", report["candidates_rejected"][0]["reason"])

    def test_rejects_ai_storytelling_channels_as_source_kind(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "OpenAI GPT-5 launch",
                "labels": ["OpenAI", "GPT-5"],
                "narration_excerpt": "The OpenAI GPT-5 product launch demonstration.",
            }],
            [self.candidate(source_type="ai_storytelling")],
            self.manifest,
        )
        self.assertNotIn("footage", scenes[0])
        self.assertEqual(report["scenes_chalk_fallback"], 1)
        self.assertEqual(report["candidates_usable_before_relevance"], 0)

    def test_rejects_irrelevant_clip_and_keeps_chalk_fallback(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "OpenAI product demo",
                "labels": ["GPT-5 launch"],
                "narration_excerpt": "OpenAI reveals a new model with a live product demonstration.",
            }],
            [self.candidate(
                title="Mint's AI future explained",
                description="Commentary about artificial intelligence trends",
                channel_title="Mint",
                source_type="news_original_reporting",
                path=str(self.clip_b),
            )],
            self.manifest,
        )
        self.assertNotIn("footage", scenes[0])
        self.assertEqual(report["scenes_chalk_fallback"], 1)
        self.assertEqual(scenes[0]["footage_decision"], "chalk_fallback_no_suitable_verified_clip")

    def test_rejects_abbreviated_noncommercial_and_no_derivatives_even_with_permission_metadata(self):
        scene = {
            "id": "beat-001",
            "title": "OpenAI GPT-5 launch",
            "labels": ["OpenAI product", "live demonstration"],
            "narration_excerpt": "OpenAI reveals the GPT-5 product in a live launch demonstration.",
        }
        for license_name in ("CC BY-NC", "CC BY-ND 4.0", "CC BY-NC-SA 4.0"):
            with self.subTest(license=license_name):
                scenes, report = select_footage_for_scenes(
                    [scene],
                    [self.candidate(
                        license=license_name,
                        rights_confirmed=True,
                        rights_documentation="permission email recorded in rights folder",
                    )],
                    self.manifest,
                )
                self.assertNotIn("footage", scenes[0])
                self.assertEqual(report["scenes_chalk_fallback"], 1)
                self.assertIn("not suitable", report["candidates_rejected"][0]["reason"])

    def test_rejects_ambiguous_creative_commons_label(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "OpenAI GPT-5 product launch",
                "labels": ["OpenAI", "GPT-5"],
                "narration_excerpt": "OpenAI GPT-5 product launch demonstration.",
            }],
            [self.candidate(license="Creative Commons")],
            self.manifest,
        )
        self.assertNotIn("footage", scenes[0])
        self.assertIn("unsupported or unclear license", report["candidates_rejected"][0]["reason"])

    def test_one_shared_topic_term_in_title_is_not_enough(self):
        scenes, report = select_footage_for_scenes(
            [{
                "id": "beat-001",
                "title": "OpenAI GPT-5 product launch",
                "labels": ["OpenAI", "GPT-5", "product demonstration"],
                "narration_excerpt": "OpenAI reveals the GPT-5 product in a live launch demonstration.",
            }],
            [self.candidate(
                title="OpenAI talks about the future",
                description="GPT-5 product launch demonstration and details",
                tags=["GPT-5", "product", "launch", "demonstration"],
            )],
            self.manifest,
        )
        self.assertNotIn("footage", scenes[0])
        self.assertEqual(report["scenes_chalk_fallback"], 1)
        self.assertIn("two key topic terms", report["decisions"][0]["candidate_rejections"][0]["reason"])

    def test_manifest_is_ignored_for_different_script(self):
        script = "The exact narration for this video."
        self.manifest.write_text(json.dumps({
            "schema": MANIFEST_SCHEMA,
            "script_sha256": script_fingerprint("a different video"),
            "candidates": [self.candidate()],
        }), encoding="utf-8")
        plan = {"scenes": [{"id": "beat-001", "title": "OpenAI GPT-5 launch"}]}
        applied = apply_footage_manifest(plan, script, self.manifest)
        self.assertEqual(applied["footage_report"]["status"], "stale_manifest")
        self.assertNotIn("footage", applied["scenes"][0])

    def test_matching_manifest_assigns_a_licensed_clip(self):
        script = "The exact narration for this video."
        self.manifest.write_text(json.dumps({
            "schema": MANIFEST_SCHEMA,
            "script_sha256": script_fingerprint(script),
            "candidates": [self.candidate(
                title="OpenAI GPT-5 product launch footage",
                description="OpenAI GPT-5 product launch and demonstration",
            )],
        }), encoding="utf-8")
        plan = {"scenes": [{
            "id": "beat-001",
            "title": "GPT-5 launch",
            "labels": ["OpenAI product demo"],
            "narration_excerpt": "OpenAI GPT-5 product launch and demonstration.",
        }]}
        applied = apply_footage_manifest(plan, script, self.manifest)
        self.assertEqual(applied["footage_report"]["status"], "applied")
        self.assertEqual(applied["footage_report"]["scenes_with_footage"], 1)

    def test_does_not_reuse_same_clip_for_multiple_chunks(self):
        scenes, report = select_footage_for_scenes(
            [
                {"id": "beat-001", "title": "OpenAI GPT-5 launch", "labels": ["OpenAI product"], "narration_excerpt": "OpenAI GPT-5 product launch."},
                {"id": "beat-002", "title": "OpenAI GPT-5 demo", "labels": ["OpenAI product"], "narration_excerpt": "OpenAI GPT-5 product demonstration."},
            ],
            [self.candidate()],
            self.manifest,
        )
        self.assertEqual(report["scenes_with_footage"], 1)
        self.assertEqual(report["scenes_chalk_fallback"], 1)
        self.assertTrue(any("footage" not in scene for scene in scenes))


if __name__ == "__main__":
    unittest.main()
