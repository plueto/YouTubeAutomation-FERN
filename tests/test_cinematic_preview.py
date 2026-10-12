import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import preview_video


class LivePlanTests(unittest.TestCase):
    SCRIPT = (
        "You are being graded right now. Not by a teacher or a boss. "
        "By the language model you opened to draft an email. "
        "Every prompt you type feeds a silent calculation. "
        "It decides how smart and dangerous you might be. "
        "And how much effort the model spends on you."
    )

    def _shots(self):
        anchors = [
            "You are being graded right now",
            "Not by a teacher or a boss",
            "By the language model you opened to draft an email",
            "Every prompt you type feeds a silent calculation",
            "It decides how smart and dangerous you might be",
            "And how much effort the model spends on you",
        ]
        return [{
            "id": f"shot-{index + 1:02d}",
            "narration_anchor": anchor,
            "dramatic_function": "hook" if index == 0 else "escalation",
            "visual_intent": f"Show a concrete illustrative image for {anchor}",
            "primary_query": f"documentary visual subject {index + 1}",
            "shot_type": "close-up" if index else "establishing wide",
            "composition": "Subject in clear focus with deliberate framing",
            "motion": "slow_push" if index % 2 == 0 else "static_hold",
            "duration_hint": 5.0,
            "transition": "cut",
            "overlay_text": "",
            "sound_cue": "",
            "music_intensity": 0.3,
            "hold_for_reveal": False,
        } for index, anchor in enumerate(anchors)]

    def test_missing_plan_uses_one_request_then_reuses_only_for_exact_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = os.path.join(tmp, "live-plan.json")
            request = Mock(return_value=json.dumps({"shots": self._shots()}))
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "mocked-test-key"}):
                shots, requests_made, reused = preview_video.build_live_shot_plan(
                    self.SCRIPT, self.SCRIPT, plan_path, request_fn=request
                )
                self.assertEqual((requests_made, reused), (1, False))
                self.assertEqual(len(shots), 6)
                cached, requests_made, reused = preview_video.build_live_shot_plan(
                    self.SCRIPT, self.SCRIPT, plan_path, request_fn=Mock()
                )
            self.assertEqual((requests_made, reused), (1, True))
            self.assertEqual(cached, shots)
            request.assert_called_once()

    def test_invalid_single_plan_response_stops_without_generic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"OPENROUTER_API_KEY": "mocked-test-key"}):
            request = Mock(return_value=json.dumps({"shots": self._shots()[:-1]}))
            with self.assertRaisesRegex(ValueError, "6-8"):
                preview_video.build_live_shot_plan(
                    self.SCRIPT, self.SCRIPT, os.path.join(tmp, "plan.json"),
                    request_fn=request,
                )
        request.assert_called_once()

    def test_subtitle_window_uses_exact_matching_script_prefix(self):
        cues = [
            {"start": 0.0, "end": 2.0, "text": "You are being graded right now."},
            {"start": 2.0, "end": 5.0, "text": "Not by a teacher or a boss."},
            {"start": 5.0, "end": 44.0, "text": "By the language model you opened to draft an email."},
            {"start": 45.0, "end": 48.0, "text": "Outside the preview window."},
        ]
        window, selected, source = preview_video.first_script_window(
            self.SCRIPT + " Outside the preview window.", cues, 480.0
        )
        self.assertEqual(source, "srt")
        self.assertTrue(window.endswith("draft an email."))
        self.assertEqual(len(selected), 3)

    def test_subtitle_cue_crossing_45_seconds_is_excluded(self):
        script = "Opening statement is here. A later sentence starts at the boundary."
        cues = [
            {"start": 0.0, "end": 44.0, "text": "Opening statement is here."},
            {"start": 44.0, "end": 47.0, "text": "A later sentence starts at the boundary."},
        ]
        window, selected, source = preview_video.first_script_window(script, cues, 300.0)
        self.assertEqual(source, "srt")
        self.assertEqual(len(selected), 1)
        self.assertEqual(window, "Opening statement is here.")


class LiveOpenverseTests(unittest.TestCase):
    def test_exact_query_cache_can_be_reused_by_adjacent_beats_without_search(self):
        from moduli import asset, montaggio

        query = "person typing email laptop"
        with tempfile.TemporaryDirectory() as tmp:
            media_paths = []
            for index in range(2):
                media = Path(tmp) / f"ov_cached_{index}.mp4"
                media.write_bytes(b"local cached media")
                Path(str(media) + ".json").write_text(json.dumps({
                    "query": query,
                    "title": f"Person typing {index}",
                    "creator": "Creator",
                    "asset_url": f"https://assets.example/{index}.jpg",
                    "source_page": f"https://source.example/{index}",
                    "license": "by",
                    "license_url": "https://creativecommons.org/licenses/by/4.0/",
                    "visual_type": "illustration",
                }), encoding="utf-8")
                media_paths.append(str(media.resolve()))

            shots = [{"id": "beat-a", "primary_query": query, "duration_hint": 5.0},
                     {"id": "beat-b", "primary_query": query, "duration_hint": 5.0}]
            with patch.object(asset, "_ov_search", side_effect=AssertionError("unexpected search")):
                assets, sources, searches = preview_video.retrieve_openverse_beat_assets(
                    shots, cache_dir=tmp
                )

        self.assertEqual(searches, 0)
        self.assertEqual(set(assets.values()), set(media_paths))
        events = [{"beat": {"id": beat_id, "primary_query": query,
                            "visual_intent": "person typing"}}
                  for beat_id in ("beat-a", "beat-b")]
        selected = montaggio._select_timeline_clips(events, assets, media_paths)
        self.assertEqual(len(selected), 2)
        self.assertTrue(any("beat-a" in key for key in assets))
        self.assertTrue(any("beat-b" in key for key in assets))
        self.assertEqual({source["beat_id"] for source in sources}, {"beat-a", "beat-b"})

    def test_retrieval_caps_eight_queries_and_two_candidates_each(self):
        from moduli import asset

        shots = [{
            "id": f"shot-{index:02d}",
            "primary_query": f"specific subject {index}",
            "duration_hint": 5.0,
        } for index in range(8)]

        def search(query, count, candidate_limit=None):
            return [{
                "url": f"https://images.example/{query.replace(' ', '-')}-{candidate}.jpg",
                "title": f"{query} candidate {candidate}",
                "creator": "Creator",
                "license": "by",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "foreign_landing_url": f"https://source.example/{query.replace(' ', '-')}-{candidate}",
                "_query": query,
            } for candidate in range(candidate_limit)]

        downloads = []

        def render(item, duration):
            downloads.append(item["url"])
            return f"cache/openverse/{len(downloads)}.mp4"

        with patch.object(asset, "_ov_search", side_effect=search) as search_mock, \
             patch.object(asset, "_ov_download_and_render", side_effect=render):
            assets, sources, searches = preview_video.retrieve_openverse_beat_assets(shots)

        self.assertEqual(searches, 8)
        self.assertEqual(search_mock.call_count, 8)
        self.assertTrue(all(call.kwargs["candidate_limit"] == 2 for call in search_mock.call_args_list))
        self.assertEqual(len(sources), 16)
        self.assertEqual(len(assets), 16)
        self.assertTrue(all("beat_id" in source and source["license"] == "by" for source in sources))


class LocalAssetTests(unittest.TestCase):
    def test_pixabay_manifest_videos_keep_shot_associations_for_renderer(self):
        from moduli.montaggio import _select_timeline_clips

        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            pixabay_dir = cache / "pixabay_preview"
            pixabay_dir.mkdir()
            laptop_path = pixabay_dir / "pixabay_101.mp4"
            classroom_path = pixabay_dir / "pixabay_202.mp4"
            invalid_path = pixabay_dir / "pixabay_invalid.jpg"
            for path in (laptop_path, classroom_path, invalid_path):
                path.write_bytes(b"x" * 2048)

            manifest = {
                "clips": [
                    {
                        "pixabay_video_id": 101,
                        "download_path": str(laptop_path),
                        "queries": ["person typing email laptop"],
                        "shot_ids": ["shot-02"],
                        "title": "Hands typing on a keyboard",
                        "uploader": "Example uploader",
                        "source_page_url": "https://pixabay.com/videos/id-101/",
                        "duration_seconds": 12,
                        "resolution": {"width": 1920, "height": 1080},
                        "license": "Pixabay Content License",
                        "license_url": "https://pixabay.com/service/license-summary/",
                        "provenance": "Pixabay official video API",
                    },
                    {
                        "pixabay_video_id": 101,
                        "download_path": str(laptop_path),
                        "queries": ["person typing email laptop"],
                        "shot_ids": ["shot-03"],
                        "title": "Hands typing on a keyboard",
                        "uploader": "Example uploader",
                        "source_page_url": "https://pixabay.com/videos/id-101/",
                        "duration_seconds": 12,
                        "resolution": {"width": 1920, "height": 1080},
                        "license": "Pixabay Content License",
                        "license_url": "https://pixabay.com/service/license-summary/",
                        "provenance": "Pixabay official video API",
                    },
                    {
                        "pixabay_video_id": 303,
                        "download_path": str(pixabay_dir / "missing.mp4"),
                        "queries": ["missing clip"],
                        "shot_ids": ["shot-03"],
                    },
                    {
                        "pixabay_video_id": 404,
                        "download_path": str(invalid_path),
                        "queries": ["not an mp4"],
                        "shot_ids": ["shot-04"],
                    },
                    {
                        "pixabay_video_id": 202,
                        "download_path": str(classroom_path),
                        "queries": ["empty classroom teacher desk"],
                        "shot_ids": ["shot-01"],
                        "title": "Empty classroom",
                        "uploader": "Another uploader",
                        "source_page_url": "https://pixabay.com/videos/id-202/",
                        "duration_seconds": 10,
                        "resolution": {"width": 1280, "height": 720},
                        "license": "Pixabay Content License",
                        "license_url": "https://pixabay.com/service/license-summary/",
                        "provenance": "Pixabay official video API",
                    },
                ]
            }
            manifest_path = pixabay_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            visual_plan = [
                {"id": "shot-01", "primary_query": "empty classroom teacher desk",
                 "visual_intent": "vacant classroom", "visual_type": "video"},
                {"id": "shot-02", "primary_query": "person typing email laptop",
                 "visual_intent": "person typing", "visual_type": "video"},
                {"id": "shot-03", "primary_query": "person typing email laptop",
                 "visual_intent": "hands typing", "visual_type": "video"},
            ]

            assets = preview_video.collect_local_assets(visual_plan, cache_root=tmp)

        pixabay_assets = {key: path for key, path in assets.items() if key.startswith("video|pixabay|")}
        self.assertEqual(set(pixabay_assets.values()), {str(laptop_path.resolve()), str(classroom_path.resolve())})
        laptop_key = next(key for key, path in pixabay_assets.items() if path == str(laptop_path.resolve()))
        self.assertIn("shot-02", laptop_key)
        self.assertIn("shot-03", laptop_key)
        self.assertIn("person typing email laptop", laptop_key)
        self.assertIn("https://pixabay.com/videos/id-101/", laptop_key)
        self.assertIn("Pixabay Content License", laptop_key)

        events = [
            {"beat": {"id": "shot-02", "primary_query": "person typing email laptop",
                      "visual_intent": "person typing", "visual_type": "video"}},
            {"beat": {"id": "shot-03", "primary_query": "person typing email laptop",
                      "visual_intent": "hands typing", "visual_type": "video"}},
            {"beat": {"id": "shot-01", "primary_query": "empty classroom teacher desk",
                      "visual_intent": "vacant classroom", "visual_type": "video"}},
        ]
        selected = _select_timeline_clips(events, pixabay_assets, list(pixabay_assets.values()))
        self.assertEqual(selected, [str(laptop_path.resolve()), str(laptop_path.resolve()),
                                    str(classroom_path.resolve())])

    def test_cached_metadata_ranks_relevant_files_and_marks_openverse_stills(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pexels = root / "pexels"
            openverse = root / "openverse"
            pexels.mkdir()
            openverse.mkdir()
            (pexels / "pex_1.mp4").write_bytes(b"video")
            (pexels / "index.json").write_text(json.dumps({
                "pex_1": {"file": "pex_1.mp4", "tags": ["company archive records"]},
            }), encoding="utf-8")
            still = openverse / "ov_abc.mp4"
            still.write_bytes(b"still")
            Path(str(still) + ".json").write_text(json.dumps({
                "query": "company archive records", "title": "Historic company ledger",
                "visual_type": "illustration",
            }), encoding="utf-8")

            assets = preview_video.collect_local_assets([
                {"primary_query": "company archive records", "visual_intent": "old ledger"},
            ], cache_root=tmp)

        keys = list(assets)
        self.assertTrue(keys[0].startswith("video|"))
        self.assertTrue(any(key.startswith("still|") for key in keys))
        self.assertTrue(any("company archive" in key for key in keys))


class PreviewIsolationTests(unittest.TestCase):
    def test_upload_gate_precedes_youtube_client_construction(self):
        source = Path("moduli/pubblica.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "pubblica_video")
        gate_line = next(node.lineno for node in ast.walk(function)
                         if isinstance(node, ast.Compare)
                         and "YOUTUBE_UPLOAD_ENABLED" in ast.unparse(node))
        client_line = next(node.lineno for node in ast.walk(function)
                           if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name)
                           and node.func.id == "_get_youtube")
        self.assertLess(gate_line, client_line)

    def test_missing_local_artifact_stops_before_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_script = os.path.join(tmp, "missing-script.txt")
            missing_audio = os.path.join(tmp, "missing-audio.mp3")
            with self.assertRaisesRegex(FileNotFoundError, "Missing local preview input"):
                preview_video.render_preview(
                    missing_script, missing_audio, output_path=os.path.join(tmp, "preview.mp4")
                )

    def test_preview_is_capped_and_calls_only_the_renderer(self):
        from moduli import montaggio

        with tempfile.TemporaryDirectory() as tmp:
            script_path = os.path.join(tmp, "script.txt")
            audio_path = os.path.join(tmp, "narration.mp3")
            output_path = os.path.join(tmp, "preview.mp4")
            Path(script_path).write_text("A company hid the losses. Workers found the records.", encoding="utf-8")
            Path(audio_path).write_bytes(b"local audio")
            with patch.object(preview_video, "collect_local_assets",
                              return_value={"still|illustrative still": "cache/openverse/ov_local.mp4"}), \
                 patch.object(montaggio, "_media_duration", return_value=75.0), \
                 patch.object(montaggio, "monta_video") as render:
                result = preview_video.render_preview(
                    script_path, audio_path, os.path.join(tmp, "missing.srt"),
                    os.path.join(tmp, "missing-plan.json"), output_path, 30.0,
                )
            self.assertEqual(result, output_path)
            self.assertEqual(render.call_args.kwargs["max_duration"], 30.0)
            self.assertIsNotNone(render.call_args.kwargs["visual_plan"])
            self.assertEqual(render.call_args.args[0], audio_path)

    def test_preview_module_has_no_upload_or_provider_entrypoint(self):
        source = Path("preview_video.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.append(node.func.id)
        self.assertFalse(any("pubblica" in name or "video_ai" in name for name in imports))
        self.assertNotIn("pubblica_video", calls)
        self.assertIn("monta_video", source)


if __name__ == "__main__":
    unittest.main()