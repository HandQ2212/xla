"""Synthetic-only checks for local statistics, seeded fusion and area budgeting."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import local_contrast as local
import local_fusion as fusion
import pipeline as pl

CONFIG = dict(median=3, window=71, k=1.5, blend="supported", area_guard="ranked")


class LocalStatisticsTests(unittest.TestCase):
    def test_constant_images_have_correct_mean_zero_deviation_and_no_dark_region(self):
        for value in (0, 1, 129, 255):
            image = np.full((17, 640, 3), value, np.uint8)
            mask, stages = local.raw_mask(image, CONFIG)
            np.testing.assert_allclose(stages["mean"], value, atol=1e-5)
            np.testing.assert_allclose(stages["deviation"], 0, atol=1e-5)
            self.assertTrue(np.isfinite(stages["threshold"]).all())
            self.assertFalse(mask.any())

    def test_window_mean_and_deviation_match_direct_float64_statistics(self):
        rng = np.random.default_rng(42)
        gray = rng.integers(30, 225, (97, 640), dtype=np.uint8)
        image = np.repeat(gray[..., None], 3, axis=2)
        _, stages = local.raw_mask(image, CONFIG)
        y, x, radius = 46, 200, CONFIG["window"] // 2
        patch = stages["den"][y - radius:y + radius + 1, x - radius:x + radius + 1].astype(np.float64)
        self.assertAlmostEqual(float(stages["mean"][y, x]), float(patch.mean()), delta=1e-4)
        self.assertAlmostEqual(float(stages["deviation"][y, x]), float(patch.std()), delta=1e-3)
        self.assertTrue(np.all(stages["deviation"] >= 0))
        self.assertTrue(np.isfinite(stages["deviation"]).all())

    def test_straight_step_is_rejected_but_narrow_dark_valley_is_detected(self):
        step = np.full((97, 640, 3), 40, np.uint8)
        step[:, 320:] = 200
        self.assertFalse(local.raw_mask(step, CONFIG)[0].any())
        valley = np.full_like(step, 180)
        valley[:, 319:322] = 30
        mask, _ = local.raw_mask(valley, CONFIG)
        self.assertTrue(mask[:, 320].all())
        self.assertFalse(mask[:, 100:200].any())


class SeededFusionTests(unittest.TestCase):
    def test_connected_fusion_recovers_diagonal_component_only_if_seeded(self):
        support = np.zeros((12, 18), bool)
        support[2, 2] = support[3, 3] = support[4, 4] = True
        support[4:8, 4:7] = True
        support[2:6, 13:16] = True
        coarse = np.zeros_like(support)
        coarse[2, 2] = True
        expected = support.copy()
        expected[:, 13:] = False
        np.testing.assert_array_equal(fusion.fuse(coarse, support, "connected"), expected)

    def test_supported_blend_preserves_coarse_only_within_local_cross_neighborhood(self):
        support = np.zeros((11, 13), bool)
        support[5, 5] = True
        coarse = np.zeros_like(support)
        coarse[5, 6] = True  # In the 3x3 ellipse's cross neighborhood.
        coarse[6, 6] = True  # Diagonal is outside that ellipse.
        coarse[2, 2] = True
        coarse_before, support_before = coarse.copy(), support.copy()
        expected = np.zeros_like(coarse)
        expected[5, 6] = True
        np.testing.assert_array_equal(fusion.fuse(coarse, support, "supported"), expected)
        self.assertFalse(fusion.fuse(coarse, support, "connected").any())
        np.testing.assert_array_equal(coarse, coarse_before)
        np.testing.assert_array_equal(support, support_before)

    def test_no_global_baseline_seed_cannot_create_image_alarm(self):
        empty = np.zeros((11, 17), bool)
        for blend in ("connected", "supported", "imagegate"):
            self.assertFalse(fusion.fuse(empty, ~empty, blend).any())


class RankedAreaTests(unittest.TestCase):
    def test_budget_selects_exact_count_from_candidate_and_never_outside_it(self):
        mask = np.zeros((5, 7), bool)
        mask[1:4, 2:5] = True
        confidence = np.arange(mask.size, dtype=np.float32).reshape(mask.shape)
        confidence[~mask] = 1000  # High scores outside candidate must be ignored.
        for budget in (0, 1, 4, 9, 100):
            result = fusion.area_budget(mask, budget, confidence)
            self.assertEqual(result.dtype, np.bool_)
            self.assertEqual(result.shape, mask.shape)
            self.assertEqual(int(result.sum()), min(int(mask.sum()), budget))
            self.assertFalse(np.any(result & ~mask))
        best = fusion.area_budget(mask, 1, confidence)
        self.assertTrue(best[3, 4])

    def test_equal_confidence_ties_are_stable_in_candidate_raster_order(self):
        mask = np.array([[False, True, True], [True, False, True]], bool)
        confidence = np.ones(mask.shape, np.float32)
        expected = np.array([[False, True, True], [False, False, False]], bool)
        np.testing.assert_array_equal(fusion.area_budget(mask, 2, confidence), expected)
        np.testing.assert_array_equal(fusion.area_budget(mask, 20, confidence), mask)

    def test_zero_budget_and_empty_mask_are_empty_and_inputs_are_unchanged(self):
        mask = np.array([[True, False], [True, True]], bool)
        confidence = np.array([[.8, .1], [.3, .5]], np.float32)
        original_mask, original_confidence = mask.copy(), confidence.copy()
        self.assertFalse(fusion.area_budget(mask, 0, confidence).any())
        self.assertFalse(fusion.area_budget(np.zeros_like(mask), 3, confidence).any())
        fusion.area_budget(mask, 2, confidence)
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(confidence, original_confidence)

    def test_full_ranked_prediction_is_native_bool_and_within_legacy_area_budget(self):
        any_baseline = False
        for shape in ((1, 1), (37, 59), (99, 137), (1, 2000), (2000, 1)):
            image = np.full(shape + (3,), 180, np.uint8)
            if min(shape) > 10:
                x = shape[1] // 2
                image[5:-5, x:x + 2] = 30
            original_image = image.copy()
            baseline = pl.pipeline(image, pl.P)
            prediction = fusion.predict(image, CONFIG)
            self.assertEqual(prediction.shape, shape)
            self.assertEqual(prediction.dtype, np.bool_)
            self.assertLessEqual(int(prediction.sum()), int(baseline.sum()))
            if not baseline.any():
                self.assertFalse(prediction.any())
            any_baseline |= baseline.any()
            np.testing.assert_array_equal(image, original_image)
        self.assertTrue(any_baseline)  # Do not make the area check entirely vacuous.


class ConservativePredictionTests(unittest.TestCase):
    def test_preservation_keeps_image_decision_and_never_exceeds_legacy_area(self):
        for shape in ((1, 1), (37, 59), (99, 137), (1, 2000), (2000, 1)):
            image = np.full(shape + (3,), 180, np.uint8)
            if min(shape) > 10:
                image[5:-5, shape[1] // 2:shape[1] // 2 + 2] = 30
            baseline = pl.pipeline(image, pl.P)
            for fraction in (0, .25, .5):
                result = fusion.predict(image, {**CONFIG, "preserve_detection": True,
                                               "min_retained": fraction})
                self.assertEqual(bool(result.any()), bool(baseline.any()))
                self.assertLessEqual(int(result.sum()), int(baseline.sum()))
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.dtype, np.bool_)

    def test_empty_local_support_falls_back_exactly_only_when_preservation_is_enabled(self):
        image = np.full((99, 137, 3), 180, np.uint8)
        image[5:-5, 67:69] = 30
        config = {**CONFIG, "window": 1}
        baseline = pl.pipeline(image, pl.P)
        self.assertTrue(baseline.any())
        self.assertFalse(local.raw_mask(image, config)[0].any())
        default = fusion.predict(image, config)
        disabled = fusion.predict(image, {**config, "preserve_detection": False})
        self.assertFalse(default.any())
        np.testing.assert_array_equal(default, disabled)
        preserved = fusion.predict(image, {**config, "preserve_detection": True,
                                          "min_retained": 0})
        np.testing.assert_array_equal(preserved, baseline)

    def test_retained_fraction_controls_fallback_on_nonempty_partial_prediction(self):
        image = np.full((128, 640, 3), 180, np.uint8)
        image[10:110, 100:103] = 30
        image[10:118, 400:425] = 30
        baseline = pl.pipeline(image, pl.P)
        research = fusion.predict(image, CONFIG)
        ratio = float(research.sum()) / float(baseline.sum())
        self.assertGreater(ratio, .25)
        self.assertLess(ratio, .5)
        for fraction in (0, .25):
            preserved = fusion.predict(image, {**CONFIG, "preserve_detection": True,
                                              "min_retained": fraction})
            np.testing.assert_array_equal(preserved, research)
        guarded = fusion.predict(image, {**CONFIG, "preserve_detection": True, "min_retained": .5})
        np.testing.assert_array_equal(guarded, baseline)


if __name__ == "__main__":
    unittest.main()
