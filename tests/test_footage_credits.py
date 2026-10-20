import json
import tempfile
import unittest
from pathlib import Path

from moduli.footage_credits import append_footage_credits, MAX_DESCRIPTION_CHARS


class FootageCreditsTests(unittest.TestCase):
    def _render_sidecar(self, root, scenes):
        video = root / "output_finale.mp4"
        plan = video.with_suffix(".scene_plan.json")
        plan.write_text(json.dumps({"scenes": scenes}), encoding="utf-8")
        return video

    def test_appends_creator_source_and_license_to_description(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self._render_sidecar(root, [{
                "id": "beat-001",
                "footage": {
                    "title": "OpenAI GPT-5 launch",
                    "attribution": "OpenAI",
                    "source_url": "https://archive.org/details/openai-gpt5",
                    "license": "CC BY 4.0",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                },
            }])
            metadata = append_footage_credits(
                {"title": "A title", "description": "Original description"},
                video,
            )
            description = metadata["description"]
            self.assertIn("Original description", description)
            self.assertIn("FOOTAGE CREDITS AND LICENSES", description)
            self.assertIn("OpenAI", description)
            self.assertIn("https://archive.org/details/openai-gpt5", description)
            self.assertIn("https://creativecommons.org/licenses/by/4.0/", description)
            self.assertLessEqual(len(description), MAX_DESCRIPTION_CHARS)

    def test_deduplicates_repeated_source_urls(self):
        footage = {
            "title": "OpenAI launch",
            "attribution": "OpenAI",
            "source_url": "https://archive.org/details/openai-gpt5",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
        }
        with tempfile.TemporaryDirectory() as tmp:
            video = self._render_sidecar(Path(tmp), [
                {"id": "beat-001", "footage": footage},
                {"id": "beat-002", "footage": {**footage, "title": "OpenAI launch repeat"}},
            ])
            metadata = append_footage_credits({"description": "Original"}, video)
            self.assertEqual(metadata["description"].count("https://archive.org/details/openai-gpt5"), 1)

    def test_incomplete_cc_by_attribution_blocks_upload_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = self._render_sidecar(Path(tmp), [{
                "footage": {
                    "title": "Incomplete source",
                    "attribution": "",
                    "source_url": "https://archive.org/details/example",
                    "license": "CC BY 4.0",
                }
            }])
            with self.assertRaisesRegex(ValueError, "CC BY attribution"):
                append_footage_credits({"description": "Original"}, video)

    def test_unreadable_scene_plan_blocks_upload_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "output_finale.mp4"
            video.with_suffix(".scene_plan.json").write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene-plan sidecar is unreadable"):
                append_footage_credits({"description": "Original"}, video)

    def test_credits_too_long_raise_instead_of_truncating_rights_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenes = []
            for index in range(15):
                scenes.append({
                    "footage": {
                        "title": "T" * 450,
                        "attribution": "Creator",
                        "source_url": "https://archive.org/details/" + str(index) + "x" * 300,
                        "license": "CC BY 4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    }
                })
            video = self._render_sidecar(Path(tmp), scenes)
            with self.assertRaisesRegex(ValueError, "exceed YouTube"):
                append_footage_credits({"description": "Original"}, video)


if __name__ == "__main__":
    unittest.main()
