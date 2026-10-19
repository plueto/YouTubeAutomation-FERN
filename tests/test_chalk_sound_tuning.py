"""Mixer-path regression tests for conservative chalk SFX levels."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from moduli import chalk_sound as sound


class ChalkSoundTuningTests(unittest.TestCase):
    def test_default_levels_reduce_effects_and_soften_sharp_ticks(self):
        with patch.dict(os.environ, {"FERN_CHALK_SFX_DB": ""}):
            gains = sound.default_gains()

        self.assertEqual(gains.chalk, -15.0)
        self.assertEqual(gains.body, -15.0)
        self.assertEqual(gains.erase, -15.0)
        self.assertEqual(gains.thump, -15.0)
        self.assertEqual(gains.whoosh, -15.0)
        self.assertEqual(gains.tick, -18.0)
        self.assertEqual(gains.tap, -18.0)
        self.assertEqual(gains.music, 0.0)

    def test_environment_override_can_lower_but_never_boost(self):
        for raw, expected in (("-18", -18.0), ("0", 0.0), ("+4", 0.0)):
            with self.subTest(raw=raw), patch.dict(os.environ, {"FERN_CHALK_SFX_DB": raw}):
                gains = sound.default_gains()
                self.assertTrue(all(gains.as_dict()[name] == expected for name in sound.STEM_ORDER))

        with patch.dict(os.environ, {"FERN_CHALK_SFX_DB": "invalid"}):
            gains = sound.default_gains()
        default_profile = {
            "chalk": -15.0, "body": -15.0, "tick": -18.0, "erase": -15.0,
            "thump": -15.0, "tap": -18.0, "whoosh": -15.0, "music": 0.0,
        }
        self.assertEqual(gains.as_dict(), default_profile)

    def test_per_stem_gain_is_applied_to_the_rendered_effect(self):
        stems = {name: np.zeros(64, dtype=np.float32) for name in sound.STEM_ORDER}
        stems["chalk"][:] = 1.0

        reference = sound.sum_stems(stems, sound.SfxGains())
        quieter = sound.sum_stems(stems, sound.SfxGains(chalk=-6.0))
        np.testing.assert_allclose(quieter, reference * (10.0 ** (-6.0 / 20.0)), rtol=1e-6, atol=1e-6)

    def test_default_attenuation_survives_the_complete_soundtrack_mixer(self):
        """Exercise build_soundtrack, not just the gain helper: defaults must reduce its actual SFX bus."""
        duration = 4.0
        t = np.arange(int(duration * sound.SR), dtype=np.float32) / sound.SR
        voice = (0.18 * (0.45 + 0.55 * np.sin(2 * np.pi * 2 * t) ** 2)
                 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        program = SimpleNamespace(
            strokes=[SimpleNamespace(t0=0.15, dur=3.4, length=1600.0, width=5.8)],
            cues=[],
        )

        with tempfile.TemporaryDirectory(prefix="fern_sfx_gain_test_") as temp_dir:
            root = Path(temp_dir)
            narration = root / "narration.wav"
            reference_mix = root / "reference.wav"
            tuned_mix = root / "tuned.wav"
            sound.write_wav(narration, voice)

            reference_components = {}
            sound.build_soundtrack(
                program, narration, None, duration, reference_mix,
                gains=sound.SfxGains(), seed=31, components=reference_components,
            )

            tuned_components = {}
            with patch.dict(os.environ, {"FERN_CHALK_SFX_DB": ""}):
                report = sound.build_soundtrack(
                    program, narration, None, duration, tuned_mix,
                    gains=None, seed=31, components=tuned_components,
                )

            reference_rms = sound._rms(reference_components["sfx"])
            tuned_rms = sound._rms(tuned_components["sfx"])
            self.assertGreater(reference_rms, 0.0)
            self.assertLess(
                tuned_rms / reference_rms, 0.30,
                f"default SFX bus was not at least ~10.5 dB down: ratio={tuned_rms / reference_rms:.3f}",
            )
            self.assertEqual(report["gains_db"]["chalk"], -18.0)
            self.assertEqual(report["gains_db"]["tick"], -18.0)
            self.assertTrue(tuned_mix.is_file())


if __name__ == "__main__":
    unittest.main()
