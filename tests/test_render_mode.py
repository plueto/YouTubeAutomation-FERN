import os
import unittest
from unittest.mock import patch

from moduli.render_mode import get_renderer_mode


class RenderModeTests(unittest.TestCase):
    def test_chunk_engine_is_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_renderer_mode(), "chalk")

    def test_chunk_engine_aliases(self):
        for value in ("chalk", "chunk", "chunk_engine", "chalkboard"):
            with self.subTest(value=value), patch.dict(os.environ, {"FERN_RENDERER": value}, clear=True):
                self.assertEqual(get_renderer_mode(), "chalk")

    def test_legacy_clip_montage_remains_available(self):
        for value in ("clips", "stock", "montage", "legacy"):
            with self.subTest(value=value), patch.dict(os.environ, {"FERN_RENDERER": value}, clear=True):
                self.assertEqual(get_renderer_mode(), "clips")

    def test_invalid_renderer_fails_closed(self):
        with patch.dict(os.environ, {"FERN_RENDERER": "random-youtube-clips"}, clear=True):
            with self.assertRaisesRegex(ValueError, "FERN_RENDERER"):
                get_renderer_mode()


if __name__ == "__main__":
    unittest.main()
