"""Synthetic numerical equality checks, not image-quality evaluation cases."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from fast_median import tiled_median


class TiledMedianTests(unittest.TestCase):
    def test_matches_opencv_on_tiny_odd_and_regular_images(self):
        rng = np.random.default_rng(42)
        shapes = [(1, 1), (1, 97), (97, 1), (9, 11), (127, 193), (143, 257), (319, 641)]
        for shape in shapes:
            image = rng.integers(0, 256, shape, dtype=np.uint8)
            original = image.copy()
            for k in (1, 3, 5, 31, 71):
                reference = cv2.medianBlur(image, k)
                for workers in (1, 2, 4):
                    for axis in (0, 1):
                        with self.subTest(shape=shape, k=k, workers=workers, axis=axis):
                            result = tiled_median(image, k, workers, axis=axis)
                            np.testing.assert_array_equal(result, reference)
                            self.assertEqual(result.dtype, np.uint8)
                            self.assertEqual(result.shape, shape)
            np.testing.assert_array_equal(image, original)

    def test_flat_ramp_and_boundary_impulses(self):
        shape = (321, 641)
        flat = np.full(shape, 128, np.uint8)
        ramp = np.tile(np.linspace(0, 255, shape[1], dtype=np.uint8), (shape[0], 1))
        impulse = flat.copy()
        for row in (0, 34, 35, 79, 80, 81, 159, 160, 161, 285, 286, 320):
            impulse[row] = 255 if row % 2 else 0
        for image in (flat, ramp, impulse):
            for k in (3, 5, 31, 71):
                reference = cv2.medianBlur(image, k)
                for workers in (2, 4):
                    for axis in (0, 1):
                        np.testing.assert_array_equal(tiled_median(image, k, workers, axis), reference)

    def test_noncontiguous_arrays(self):
        rng = np.random.default_rng(123)
        full = rng.integers(0, 256, (645, 487), dtype=np.uint8)
        for image in (full[::2, ::2], full.T):
            for k in (31, 71):
                for workers in (2, 4):
                    for axis in (0, 1):
                        np.testing.assert_array_equal(tiled_median(image, k, workers, axis), cv2.medianBlur(image, k))

    def test_invalid_inputs_fail_before_filtering(self):
        image = np.zeros((20, 30), np.uint8)
        for k in (0, 2, -3, 3.5):
            with self.assertRaises(ValueError):
                tiled_median(image, k)
        for workers in (0, -1, 2.5):
            with self.assertRaises(ValueError):
                tiled_median(image, 3, workers)
        with self.assertRaises(ValueError):
            tiled_median(np.zeros((0, 3), np.uint8), 3)
        with self.assertRaises(ValueError):
            tiled_median(image, 3, axis=2)
        for invalid in (image.astype(np.float32), image[..., None], [[0, 1]]):
            with self.assertRaises(TypeError):
                tiled_median(invalid, 3)


if __name__ == "__main__":
    unittest.main()
