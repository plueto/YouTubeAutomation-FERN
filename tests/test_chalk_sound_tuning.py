"""Regression tests for conservative chalk-sound levels; visuals and narration are outside this tuning patch."""
import os
import unittest
from unittest.mock import patch

import numpy as np

from moduli import chalk_sound as sound


class ChalkSoundTuningTests(unittest.TestCase):
    def test_default_levels_reduce_effects_and_soften_sharp_ticks(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FERN_CHALK_SFX_DB", None)
            gains = sound.default_gains()

        self.assertEqual(gains.chalk, -6.0)
        self.assertEqual(gains.body, -6.0)
        self.assertEqual(gains.erase, -6.0)
        self.assertEqual(gains.thump, -6.0)
        self.assertEqual(gains.whoosh, -6.0)
        self.assertEqual(gains.tick, -7.0)
        self.assertEqual(gains.tap, -7.0)
        self.assertEqual(gains.music, 0.0)

    def test_environment_override_can_lower_but_never_boost(self):
        for raw, expected in (("-9", -9.0), ("0", 0.0), ("+4", 0.0)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"FERN_CHALK_SFX_DB": raw}):
                gains = sound.default_gains()
                self.assertTrue(all(gains.as_dict()[name] == expected for name in sound.STEM_ORDER))

        with patch.dict(os.environ, {"FERN_CHALK_SFX_DB": "invalid"}):
            gains = sound.default_gains()
        default_profile = {
            "chalk": -6.0, "body": -6.0, "tick": -7.0, "erase": -6.0,
            "thump": -6.0, "tap": -7.0, "whoosh": -6.0, "music": 0.0,
        }
        self.assertEqual(gains.as_dict(), default_profile)

    def test_per_stem_gain_is_applied_to_the_rendered_effect(self):
        stems = {name: np.zeros(64, dtype=np.float32) for name in sound.STEM_ORDER}
        stems["chalk"][:] = 1.0

        reference = sound.sum_stems(stems, sound.SfxGains())
        quieter = sound.sum_stems(stems, sound.SfxGains(chalk=-6.0))
        np.testing.assert_allclose(quieter, reference * (10.0 ** (-6.0 / 20.0)), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
