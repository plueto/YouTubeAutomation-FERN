import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from moduli.archive_footage import (
    _media_files,
    _search_query,
    discover_archive_footage,
    license_from_url,
)


class ArchiveFootageTests(unittest.TestCase):
    def test_license_whitelist_accepts_cc0_pd_mark_and_cc_by(self):
        self.assertEqual(
            license_from_url("https://creativecommons.org/publicdomain/zero/1.0/"),
            "CC0",
        )
        self.assertEqual(
            license_from_url("https://creativecommons.org/publicdomain/mark/1.0/"),
            "Public Domain Mark",
        )
        self.assertEqual(
            license_from_url("https://creativecommons.org/licenses/by/4.0/"),
            "CC BY 4.0",
        )
        self.assertEqual(
            license_from_url("https://creativecommons.org/licenses/by/3.0/"),
            "CC BY 3.0",
        )

    def test_license_whitelist_rejects_nc_nd_sa_and_standard_license(self):
        for url in (
            "https://creativecommons.org/licenses/by-nc/4.0/",
            "https://creativecommons.org/licenses/by-nd/4.0/",
            "https://creativecommons.org/licenses/by-sa/4.0/",
            "https://example.com/license",
            "",
        ):
            with self.subTest(url=url):
                self.assertIsNone(license_from_url(url))

    def test_archive_query_uses_only_sanitized_scene_terms(self):
        query = _search_query(["openai", 'gpt" OR mediatype:audio', "launch"])
        self.assertIn("mediatype:movies", query)
        self.assertIn('title:"openai"', query)
        self.assertNotIn("mediatype:audio", query)
        self.assertNotIn('" OR', query)

    def test_file_selection_skips_non_mp4_private_and_oversized_files(self):
        files = [
            {"name": "poster.jpg", "size": "2000", "format": "JPEG"},
            {"name": "private.mp4", "size": "2000", "format": "MPEG4", "private": "true"},
            {"name": "huge.mp4", "size": str(500 * 1024 * 1024), "format": "MPEG4"},
            {"name": "actual.mp4", "size": "10000", "format": "MPEG4"},
        ]
        selected = _media_files(files, 200 * 1024 * 1024)
        self.assertEqual([f["name"] for f in selected], ["actual.mp4"])

    @patch("moduli.archive_footage._probe_duration", return_value=15.0)
    @patch("moduli.archive_footage.requests.get")
    def test_discovery_writes_script_bound_manifest_for_a_license_cleared_relevant_mp4(
        self, get_mock, _probe_mock
    ):
        scene = {
            "id": "beat-001",
            "title": "OpenAI GPT-5 product launch",
            "labels": ["OpenAI", "GPT-5", "product demo"],
            "narration_excerpt": "OpenAI reveals the GPT-5 product in a live launch demonstration.",
            "visual_intent": "show the actual OpenAI product launch",
        }
        search_response = MagicMock()
        search_response.raise_for_status.return_value = None
        search_response.json.return_value = {
            "response": {
                "docs": [{
                    "identifier": "openai-gpt5-license-demo",
                    "title": "OpenAI GPT-5 product launch demonstration",
                    "description": "Recording of the OpenAI GPT-5 launch and product demo.",
                    "creator": "OpenAI",
                    "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
                    "subject": ["OpenAI", "GPT-5", "launch"],
                    "mediatype": "movies",
                }]
            }
        }
        metadata_response = MagicMock()
        metadata_response.raise_for_status.return_value = None
        metadata_response.json.return_value = {
            "metadata": {
                "title": "OpenAI GPT-5 product launch demonstration",
                "description": "Recording of the OpenAI GPT-5 launch and product demo.",
                "creator": "OpenAI",
                "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
                "subject": ["OpenAI", "GPT-5", "launch"],
            },
            "files": [{
                "name": "openai-gpt5-demo.mp4",
                "size": "4096",
                "format": "MPEG4",
                "filetype": "video/mp4",
            }],
        }
        download_response = MagicMock()
        download_response.__enter__.return_value = download_response
        download_response.__exit__.return_value = False
        download_response.url = "https://archive.org/download/openai-gpt5-license-demo/openai-gpt5-demo.mp4"
        download_response.headers = {"content-type": "video/mp4", "content-length": "4096"}
        download_response.iter_content.return_value = [b"x" * 4096]

        def route(url, *args, **kwargs):
            if "advancedsearch.php" in url:
                return search_response
            if "/metadata/" in url:
                return metadata_response
            if "/download/" in url:
                return download_response
            self.fail(f"Unexpected URL: {url}")

        get_mock.side_effect = route
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ",
            {
                "FERN_ARCHIVE_MAX_SEARCHES": "2",
                "FERN_ARCHIVE_MAX_DOWNLOADS": "1",
                "FERN_ARCHIVE_MAX_DOWNLOAD_MB": "20",
            },
        ):
            manifest_path = Path(tmp) / "output" / "footage_candidates_auto.json"
            report = discover_archive_footage([scene], scene["narration_excerpt"], manifest_path)

            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["candidates_downloaded"], 1)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["candidates"][0]["source_type"], "licensed_archive_video")
            self.assertEqual(manifest["candidates"][0]["license"], "CC BY 4.0")
            self.assertEqual(manifest["candidates"][0]["review_status"], "approved")
            self.assertTrue(Path(manifest["candidates"][0]["path"]).is_file())
            self.assertEqual(
                manifest["script_sha256"],
                __import__("hashlib").sha256(
                    " ".join(scene["narration_excerpt"].split()).encode("utf-8")
                ).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
