"""Mathematical checks on synthetic arrays only; no source images or GT are read."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import line_response as line


class SymmetricResponseTests(unittest.TestCase):
    def test_uniform_dark_gray_and_bright_images_have_zero_response(self):
        for value in (0, 1, 127, 128, 129, 253, 255):
            image = np.full((67, 91, 3), value, np.uint8)
            for width in (2, 3):
                with self.subTest(value=value, width=width):
                    strength = line.response(image, width)
                    self.assertEqual(strength.shape, image.shape[:2])
                    self.assertEqual(strength.dtype, np.uint8)
                    self.assertFalse(strength.any())

    def test_monotonic_step_is_not_a_two_sided_dark_line(self):
        image = np.full((97, 101, 3), 40, np.uint8)
        image[:, 50:] = 200
        for oriented in (image, np.rot90(image).copy()):
            for width in (2, 3):
                with self.subTest(shape=oriented.shape, width=width):
                    strength = line.response(oriented, width)
                    self.assertFalse(strength[12:-12, 12:-12].any())

    def test_dark_line_has_contrast_at_its_center_and_not_on_flat_flanks(self):
        image = np.full((97, 101, 3), 180, np.uint8)
        image[:, 49:52] = 35
        for width in (2, 3):
            strength = line.response(image, width)
            self.assertGreater(int(strength[20:-20, 50].min()), 50)
            self.assertFalse(strength[20:-20, 15:25].any())

    def test_ninety_degree_rotation_and_reflection_preserve_response(self):
        image = np.full((97, 101, 3), 180, np.uint8)
        image[12:85, 49:52] = 35
        image[55:58, 30:80] = 60
        expected = line.response(image, 3)
        for turns in (1, 2, 3):
            actual = line.response(np.rot90(image, turns).copy(), 3)
            error = np.abs(actual.astype(np.int16) - np.rot90(expected, turns).astype(np.int16))
            self.assertLessEqual(int(error.max()), 1)
        actual = line.response(image[:, ::-1].copy(), 3)
        error = np.abs(actual.astype(np.int16) - expected[:, ::-1].astype(np.int16))
        self.assertLessEqual(int(error.max()), 1)

    def test_diagonal_line_is_supported_without_claiming_exact_isotropy(self):
        image = np.full((129, 129, 3), 180, np.uint8)
        yy, xx = np.indices(image.shape[:2])
        image[np.abs(xx - yy) <= 1] = 35
        strength = line.response(image, 3)
        diagonal = np.arange(20, 109)
        self.assertGreater(int(strength[diagonal, diagonal].min()), 50)

    def test_shift_reflects_same_border_instead_of_wrapping_opposite_edge(self):
        array = np.arange(35, dtype=np.float32).reshape(5, 7)
        right = line.shifted(array, 1, 0)
        np.testing.assert_array_equal(right[:, :-1], array[:, 1:])
        np.testing.assert_array_equal(right[:, -1], array[:, -2])
        self.assertFalse(np.array_equal(right[:, -1], array[:, 0]))
        up = line.shifted(array, 0, -1)
        np.testing.assert_array_equal(up[1:], array[:-1])
        np.testing.assert_array_equal(up[0], array[1])
        np.testing.assert_array_equal(line.shifted(array, 0, 0), array)

    def test_longer_tangent_persistence_only_suppresses_response(self):
        rng = np.random.default_rng(42)
        image = rng.integers(30, 220, (71, 93, 3), dtype=np.uint8)
        image[8:62, 44:47] = 20
        for width in (2, 3):
            short = line.response(image, width, persistence=3)
            np.testing.assert_array_equal(line.response(image, width), short)
            longer = line.response(image, width, persistence=7)
            self.assertTrue(np.all(longer <= short))
            self.assertTrue(np.any(longer < short))


class RegionEvidenceTests(unittest.TestCase):
    def test_line_mask_keeps_long_thin_region_and_rejects_round_small_and_unseeded(self):
        strength = np.zeros((70, 90), np.uint8)
        strength[10:14, 5:70] = 30  # Long narrow high-contrast region.
        strength[40:50, 45:55] = 30  # Compact square, despite sufficient area.
        strength[60, 70] = 30  # Tiny isolated response.
        strength[25:28, 5:65] = 2  # Weak component disconnected from strong markers.
        mask, threshold = line.line_mask(strength)
        self.assertGreaterEqual(threshold, 3)
        self.assertTrue(mask[11, 30])
        self.assertFalse(mask[45, 50])
        self.assertFalse(mask[60, 70])
        self.assertFalse(mask[26, 30])

    def test_gate_retains_whole_component_only_when_evidence_fraction_passes(self):
        mask = np.zeros((18, 30), bool)
        mask[3:5, 2:12] = True
        mask[10:12, 15:25] = True
        evidence = np.zeros_like(mask)
        evidence[3, 2:4] = True  # 2/20 pixels: exactly 10%.
        evidence[10, 15] = True  # 1/20 pixels: below 10%.
        evidence[0, :] = True  # Background evidence must not become foreground.
        expected = np.zeros_like(mask)
        expected[3:5, 2:12] = True
        np.testing.assert_array_equal(line.gate_components(mask, evidence, .1), expected)
        self.assertFalse(line.gate_components(mask, evidence, .1001).any())

    def test_gate_with_no_evidence_or_no_components_returns_empty(self):
        empty = np.zeros((13, 19), bool)
        self.assertFalse(line.gate_components(~empty, empty, .1).any())
        self.assertFalse(line.gate_components(empty, ~empty, .1).any())

    def test_predictions_are_original_size_bool_for_odd_tiny_and_extreme_inputs(self):
        configs = [next(c for c in line.CONFIGS if c["method"] == method)
                   for method in ("line", "gate", "augment")]
        for shape in ((1, 1), (53, 79), (1, 2000), (2000, 1)):
            image = np.full(shape + (3,), 129, np.uint8)
            for config in configs:
                with self.subTest(shape=shape, method=config["method"]):
                    prediction = line.predict(image, config)
                    self.assertEqual(prediction.shape, shape)
                    self.assertEqual(prediction.dtype, np.bool_)
                    self.assertFalse(prediction.any())


if __name__ == "__main__":
    unittest.main()
