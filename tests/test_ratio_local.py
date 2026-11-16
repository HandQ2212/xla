"""Synthetic-only tests of ratio statistics, protected ranking and refinement."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline as pl
import ratio_local as method

CONFIG = dict(window=31, k=1.5, protect_core=True)


class RatioStatisticsTests(unittest.TestCase):
    def test_flat_ratio_has_no_local_response_and_finite_confidence(self):
        for value in (0, 1, 128, 129, 253, 255):
            support, confidence, stats = method.local_support(
                {"ratio": np.full((63, 97), value, np.uint8)}, CONFIG)
            self.assertFalse(stats["local_raw"].any())
            self.assertFalse(support.any())
            np.testing.assert_allclose(stats["local_mean"], value, atol=1e-5)
            np.testing.assert_allclose(stats["local_deviation"], 0, atol=1e-5)
            self.assertTrue(np.isfinite(confidence).all())

    def test_ratio_mean_and_deviation_match_independent_window_statistics(self):
        rng = np.random.default_rng(2026)
        ratio = rng.integers(50, 200, (65, 97), dtype=np.uint8)
        _, confidence, stats = method.local_support({"ratio": ratio}, CONFIG)
        patch = ratio[17:48, 33:64].astype(np.float64)
        self.assertAlmostEqual(float(stats["local_mean"][32, 48]), float(patch.mean()), delta=1e-4)
        self.assertAlmostEqual(float(stats["local_deviation"][32, 48]), float(patch.std()), delta=1e-3)
        self.assertTrue(np.all(stats["local_deviation"] >= 0))
        self.assertTrue(np.isfinite(confidence).all())

    def test_cached_ratio_drives_statistics_independently_of_raw_shadow_channels(self):
        ratio = np.full((80, 160), 128, np.uint8)
        ratio[:, 60:65] = 40
        clear = {"ratio": ratio, "gray": np.full(ratio.shape, 180, np.uint8)}
        shadow = {"ratio": ratio, "gray": np.full(ratio.shape, 35, np.uint8)}
        shadow["gray"][:, :80] = 180
        first = method.local_support(clear, CONFIG)
        second = method.local_support(shadow, CONFIG)
        self.assertTrue(first[0].any())
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


class ProtectedBudgetTests(unittest.TestCase):
    def test_forced_pixels_win_before_stable_raster_ties(self):
        candidate = np.ones((2, 3), bool)
        forced = np.zeros_like(candidate)
        forced[1, 2] = True
        confidence = np.ones(candidate.shape, np.float32)
        confidence[1, 2] = -100
        result = method.budget_mask(candidate, forced, 3, confidence)
        expected = np.array([[True, True, False], [False, False, True]])
        np.testing.assert_array_equal(result, expected)

    def test_exact_cap_subset_zero_budget_and_already_fitting_candidate(self):
        candidate = np.zeros((5, 7), bool)
        candidate[1:4, 2:5] = True
        confidence = np.arange(35, dtype=np.float32).reshape(candidate.shape)
        empty = np.zeros_like(candidate)
        for budget in (0, 1, 4, 9, 100):
            result = method.budget_mask(candidate, empty, budget, confidence)
            self.assertEqual(int(result.sum()), min(9, budget))
            self.assertFalse(np.any(result & ~candidate))
            self.assertEqual(result.dtype, np.bool_)
        np.testing.assert_array_equal(method.budget_mask(candidate, empty, 20, confidence), candidate)

    def test_forced_core_exceeding_budget_raises_instead_of_dropping_core(self):
        candidate = np.ones((3, 4), bool)
        forced = np.zeros_like(candidate)
        forced[1, :] = True
        with self.assertRaisesRegex(ValueError, "core exceeds"):
            method.budget_mask(candidate, forced, 3, np.ones(candidate.shape, np.float32))

    def test_budget_inputs_are_not_mutated(self):
        candidate = np.ones((4, 7), bool)
        forced = np.zeros_like(candidate)
        forced[2, 2] = True
        confidence = np.arange(28, dtype=np.float32).reshape(candidate.shape)
        originals = [array.copy() for array in (candidate, forced, confidence)]
        method.budget_mask(candidate, forced, 6, confidence)
        for actual, original in zip((candidate, forced, confidence), originals):
            np.testing.assert_array_equal(actual, original)


class RatioRefinementTests(unittest.TestCase):
    @staticmethod
    def stages(with_line):
        coarse = np.zeros((80, 160), bool)
        coarse[15:30, 60:65] = True
        ratio = np.full(coarse.shape, 128, np.uint8)
        if with_line:
            ratio[:, 60:65] = 40
        return dict(Z=coarse, ratio=ratio)

    def test_guarded_core_survives_when_budget_fits_and_when_ranking_is_needed(self):
        shape = (321, 641)
        for with_line in (False, True):
            stages = self.stages(with_line)
            baseline = pl.to_orig(stages["Z"], shape)
            core = cv2.erode(stages["Z"].astype(np.uint8), pl.ellipse(3)) > 0
            forced = pl.to_orig(core, shape)
            result = method.refine(stages, shape, CONFIG)
            self.assertTrue(forced.any())
            self.assertFalse(np.any(forced & ~result))
            self.assertLessEqual(int(result.sum()), int(baseline.sum()))
            if with_line:
                self.assertEqual(int(result.sum()), int(baseline.sum()))
            else:
                np.testing.assert_array_equal(result, forced)
                self.assertLess(int(result.sum()), int(baseline.sum()))

    def test_small_images_use_exact_baseline_without_accessing_ratio(self):
        coarse = np.zeros((13, 19), bool)
        coarse[2:10, 5:8] = True
        for shape in ((1, 1), (99, 137), (320, 640)):
            expected = pl.to_orig(coarse, shape)
            np.testing.assert_array_equal(method.refine({"Z": coarse}, shape, CONFIG), expected)

    def test_actual_prediction_shape_decision_and_area_for_odd_tiny_and_extreme_inputs(self):
        nonempty_large = False
        for shape in ((1, 1), (99, 137), (129, 641), (1, 1001), (1001, 1)):
            image = np.full(shape + (3,), 180, np.uint8)
            if min(shape) > 10:
                x = shape[1] // 2
                image[8:-8, x:x + 3] = 30
            original = image.copy()
            baseline = pl.pipeline(image, pl.P)
            for protect in (False, True):
                result = method.predict(image, {**CONFIG, "protect_core": protect})
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.bool_)
                self.assertEqual(bool(result.any()), bool(baseline.any()))
                self.assertLessEqual(int(result.sum()), int(baseline.sum()))
                if max(shape) <= 640:
                    np.testing.assert_array_equal(result, baseline)
            nonempty_large |= max(shape) > 640 and baseline.any()
            np.testing.assert_array_equal(image, original)
        self.assertTrue(nonempty_large)

    def test_stage_arrays_remain_unchanged(self):
        stages = self.stages(True)
        originals = {key: array.copy() for key, array in stages.items()}
        method.refine(stages, (321, 641), CONFIG)
        for key in stages:
            np.testing.assert_array_equal(stages[key], originals[key])


if __name__ == "__main__":
    unittest.main()
