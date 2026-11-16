"""Synthetic-only invariants for native-resolution research mask refinement."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import mask_refinement as refinement
import pipeline as pl


class NativeMaskRefinementTests(unittest.TestCase):
    def test_every_configuration_is_a_subset_of_upscaled_legacy_mask(self):
        rng = np.random.default_rng(20261115)
        image = rng.integers(0, 256, (61, 87, 3), dtype=np.uint8)
        coarse = rng.random((13, 19)) > .45
        baseline = pl.to_orig(coarse, image.shape[:2])
        for config in refinement.CONFIGS:
            with self.subTest(config=config["name"]):
                result = refinement.refine(image, coarse, config)
                self.assertFalse(np.any(result & ~baseline))
                self.assertLessEqual(np.count_nonzero(result), np.count_nonzero(baseline))

    def test_empty_coarse_mask_cannot_create_a_new_image_alarm(self):
        image = np.full((43, 71, 3), 190, np.uint8)
        image[:, 33:36] = 15  # Strong dark structure must not invent baseline detections.
        coarse = np.zeros((9, 15), bool)
        for config in refinement.CONFIGS:
            with self.subTest(config=config["name"]):
                self.assertFalse(refinement.refine(image, coarse, config).any())

    def test_guard_keeps_eroded_core_when_intensity_support_is_empty(self):
        image = np.full((101, 151, 3), 128, np.uint8)
        coarse = np.zeros((20, 30), bool)
        coarse[4:15, 6:22] = True
        config = dict(window=15, offset=0, guard=False)
        without_guard = refinement.refine(image, coarse, config)
        self.assertFalse(without_guard.any())
        guarded = refinement.refine(image, coarse, {**config, "guard": True})
        eroded = cv2.erode(coarse.astype(np.uint8), pl.ellipse(3)) > 0
        expected_core = pl.to_orig(eroded, image.shape[:2])
        self.assertTrue(expected_core.any())
        np.testing.assert_array_equal(guarded, expected_core)

    def test_native_dark_line_is_kept_while_bright_mask_boundary_is_removed(self):
        image = np.full((65, 81, 3), 180, np.uint8)
        image[:, 39:42] = 35
        coarse = np.zeros(image.shape[:2], bool)
        coarse[:, 35:46] = True
        result = refinement.refine(image, coarse, dict(window=31, offset=0, guard=False))
        self.assertTrue(result[12:-12, 39:42].all())
        self.assertFalse(result[:, 35:38].any())
        self.assertFalse(result[:, 43:46].any())

    def test_larger_offset_cannot_remove_pixels_for_fixed_window_and_guard(self):
        rng = np.random.default_rng(2026)
        gray = rng.integers(80, 170, (99, 137), dtype=np.uint8)
        image = np.repeat(gray[..., None], 3, axis=2)
        coarse = np.zeros((17, 23), bool)
        coarse[2:-2, 2:-2] = True
        grew = False
        for window in (15, 31):
            for guard in (False, True):
                results = [refinement.refine(image, coarse,
                           dict(window=window, offset=offset, guard=guard)) for offset in (0, 4, 8)]
                for lower, higher in zip(results, results[1:]):
                    self.assertFalse(np.any(lower & ~higher))
                    grew |= np.any(higher & ~lower)
        self.assertTrue(grew)

    def test_original_odd_tiny_and_extreme_shapes_are_boolean(self):
        for shape in ((1, 1), (37, 59), (1, 2000), (2000, 1)):
            image = np.full(shape + (3,), 129, np.uint8)
            coarse_shape = (max(1, shape[0] // 4), max(1, shape[1] // 4))
            coarse = np.ones(coarse_shape, bool)
            for config in refinement.CONFIGS:
                with self.subTest(shape=shape, config=config["name"]):
                    result = refinement.refine(image, coarse, config)
                    self.assertEqual(result.shape, shape)
                    self.assertEqual(result.dtype, np.bool_)

    def test_image_and_coarse_mask_are_not_mutated(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (55, 81, 3), dtype=np.uint8)
        coarse = rng.random((11, 17)) > .5
        image_before, coarse_before = image.copy(), coarse.copy()
        for config in refinement.CONFIGS:
            refinement.refine(image, coarse, config)
        np.testing.assert_array_equal(image, image_before)
        np.testing.assert_array_equal(coarse, coarse_before)


if __name__ == "__main__":
    unittest.main()
