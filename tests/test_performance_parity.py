"""Exact parity with immutable d155 P, using synthetic and bundled images only."""
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline as optimized

SNAPSHOT = ROOT / "docs/benchmarks/performance/baseline_pipeline.py"
SNAPSHOT_SHA256 = "2ac8116ce3b5411e7722efef3b8d9ee3e4d2a4901e553f3e68d9ee55826224f1"
spec = importlib.util.spec_from_file_location("immutable_d155_pipeline", SNAPSHOT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class PerformanceParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)  # Exercise the optimized main-process tiled-median route.

    @classmethod
    def tearDownClass(cls):
        cv2.setNumThreads(cls.previous_threads)

    def assert_stages_equal(self, old, new):
        self.assertEqual(set(old), set(new))
        for key in old:
            with self.subTest(stage=key):
                if isinstance(old[key], np.ndarray):
                    np.testing.assert_array_equal(new[key], old[key])
                else:
                    self.assertEqual(new[key], old[key])

    def test_baseline_snapshot_is_immutable(self):
        content = SNAPSHOT.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(content).hexdigest(), SNAPSHOT_SHA256)
        self.assertEqual(optimized.P, baseline.P)

    def test_illumination_division_and_subtraction_match_without_input_mutation(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (63, 97), dtype=np.uint8)
        image[:9, :9] = 0
        image[-9:, -9:] = 255
        original = image.copy()
        for mode in ("median", "close"):
            for op in ("div", "sub"):
                for k in (3, 31, 71):
                    with self.subTest(mode=mode, op=op, k=k):
                        old = baseline.illumination_ratio(image, k, mode, op)
                        new = optimized.illumination_ratio(image, k, mode, op)
                        for actual, expected in zip(new, old):
                            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, original)

    def test_tiled_illumination_route_is_exact_in_both_axis_orientations(self):
        rng = np.random.default_rng(2026)
        image = rng.integers(0, 256, (401, 501), dtype=np.uint8)
        for oriented in (image, image.T):
            for k in (31, 71):
                for op in ("div", "sub"):
                    with self.subTest(shape=oriented.shape, k=k, op=op):
                        old = baseline.illumination_ratio(oriented, k, op=op)
                        new = optimized.illumination_ratio(oriented, k, op=op)
                        for actual, expected in zip(new, old):
                            np.testing.assert_array_equal(actual, expected)

    def test_histogram_fast_path_and_small_fallback_paths_are_exact(self):
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, (71, 103), dtype=np.uint8)
        arrays = (image, image.T, image[:, ::2], image.ravel(), image.astype(np.uint16),
                  np.repeat(image[..., None], 3, axis=2), np.empty((0, 17), np.uint8))
        for array in arrays:
            for window in (1, 3, 5):
                with self.subTest(shape=array.shape, dtype=array.dtype, window=window):
                    np.testing.assert_array_equal(optimized.histogram(array, window),
                                                  baseline.histogram(array, window))

    def test_large_histogram_fallback_preserves_count_above_float32_exact_limit(self):
        image = np.full((1, 2**24 + 1), 128, np.uint8)
        histogram = optimized.histogram(image)
        self.assertEqual(histogram[128], 2**24 + 1)
        np.testing.assert_array_equal(histogram, baseline.histogram(image))

    def test_inplace_sobel_and_edge_region_match_exactly(self):
        rng = np.random.default_rng(2026)
        images = [rng.integers(0, 256, shape, dtype=np.uint8)
                  for shape in ((1, 1), (1, 91), (91, 1), (67, 95))]
        images += [images[-1][:, ::2], np.zeros((31, 49), np.uint8)]
        for image in images:
            original = image.copy()
            np.testing.assert_array_equal(optimized.sobel_l1(image), baseline.sobel_l1(image))
            for actual, expected in zip(optimized.edge_region(image, 3), baseline.edge_region(image, 3)):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(image, original)

    def test_contour_rectangle_matches_pixel_rectangle_on_degenerate_and_complex_regions(self):
        shapes = []
        for angle in (0, 15, 30, 45, 60, 75, 90, 121, 179):
            mask = np.zeros((81, 137), np.uint8)
            box = cv2.boxPoints(((68., 40.), (91., 7.), angle)).astype(np.int32)
            cv2.fillPoly(mask, [box], 1)
            shapes.append(mask)
        for points in ([(0, 0)], [(4, 5), (5, 6)], [(i, i) for i in range(50)]):
            mask = np.zeros((81, 137), np.uint8)
            for y, x in points:
                mask[y, x] = 1
            shapes.append(mask)
        ring = np.zeros((81, 137), np.uint8)
        cv2.rectangle(ring, (3, 5), (100, 65), 1, 3)
        ring[70:78, 120:130] = 1
        shapes.append(ring)
        rng = np.random.default_rng(42)
        shapes += [(rng.random((81, 137)) < density).astype(np.uint8) for density in (.005, .1, .7)]
        for index, mask in enumerate(shapes):
            for rect in ("axis", "rotated"):
                with self.subTest(shape=index, rect=rect):
                    self.assertEqual(optimized.elongation(mask, rect), baseline.elongation(mask, rect))
                    for area in (0, 30):
                        for minimum in (0, 2, 3, 5):
                            old = baseline.elongation_filter(mask, minimum, rect, area, True)
                            new = optimized.elongation_filter(mask, minimum, rect, area, True)
                            for actual, expected in zip(new, old):
                                np.testing.assert_array_equal(actual, expected)

    def test_full_default_pipeline_and_every_intermediate_match_all_twenty_bundled_images(self):
        count = 0
        for group in ("samples", "samples_pcs"):
            for path in sorted((ROOT / group).glob("*.jpg")):
                if path.stem.endswith("_pred"):
                    continue
                with self.subTest(image=path.name):
                    image = cv2.imread(str(path))
                    self.assertIsNotNone(image)
                    self.assert_stages_equal(baseline.pipeline(image, baseline.P, steps=True),
                                             optimized.pipeline(image, optimized.P, steps=True))
                    np.testing.assert_array_equal(optimized.pipeline(image, optimized.P),
                                                  baseline.pipeline(image, baseline.P))
                count += 1
        self.assertEqual(count, 20)

    def test_full_pipeline_parity_for_supported_ablation_settings(self):
        rng = np.random.default_rng(2026)
        image = rng.integers(70, 180, (129, 197, 3), dtype=np.uint8)
        image[10:115, 90:94] = 20
        variants = [baseline.P, baseline.P0] + [{**baseline.P, **change} for change in
                    (dict(op="sub"), dict(bg_mode="close", bg=31), dict(denoise="gauss"),
                     dict(thresh="otsu"), dict(rect="axis"), dict(post=False),
                     dict(use_edge=False), dict(bg=0), dict(hist_w=1))]
        for params in variants:
            with self.subTest(params=params):
                self.assert_stages_equal(baseline.pipeline(image, params, steps=True),
                                         optimized.pipeline(image, params, steps=True))
                np.testing.assert_array_equal(optimized.pipeline(image, params), baseline.pipeline(image, params))


if __name__ == "__main__":
    unittest.main()
