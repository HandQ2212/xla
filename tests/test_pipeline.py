"""Behavioral regression tests for classical segmentation and region reconstruction."""
import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import pipeline as pl


class ReconstructionTests(unittest.TestCase):
    def test_diagonal_growth_excludes_disconnected_dark_region(self):
        support = np.zeros((9, 12), bool)
        support[1, 1] = support[2, 2] = support[3, 3] = True
        support[3:6, 3:6] = True
        support[1:4, 9:11] = True
        seed = np.zeros_like(support)
        seed[1, 1] = True
        seed[8, 11] = True  # A marker outside support must not leak into it.
        expected = support.copy()
        expected[:, 9:] = False
        np.testing.assert_array_equal(pl.reconstruct(seed, support), expected)

    def test_matches_iterated_constrained_dilation(self):
        rng = np.random.default_rng(2026)
        for _ in range(12):
            support = rng.random((23, 31)) < 0.42
            seed = rng.random(support.shape) < 0.025
            expected = seed & support
            while True:
                grown = cv2.dilate(expected.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                grown &= support
                if np.array_equal(grown, expected):
                    break
                expected = grown
            np.testing.assert_array_equal(pl.reconstruct(seed, support), expected)

    def test_no_marker_and_empty_support(self):
        empty = np.zeros((11, 17), bool)
        self.assertFalse(pl.reconstruct(empty, ~empty).any())
        self.assertFalse(pl.reconstruct(~empty, empty).any())


class PipelineRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(str(ROOT / "samples_pcs" / "B_00038.jpg"))
        if cls.image is None:
            raise RuntimeError("Missing bundled regression image B_00038.jpg")

    def test_single_pass_region_filter_preserves_two_stage_result(self):
        mask = np.zeros((100, 130), bool)
        mask[4:8, 3:65] = True
        mask[14:31, 4:20] = True
        mask[46, 12] = True
        for offset in range(25):
            mask[48 + offset, 35 + offset] = True
        mask[70:90, 100:103] = True
        for rect in ("axis", "rotated"):
            for area in (0, 2, 30, 500):
                for minimum in (0, 2, 3):
                    with self.subTest(rect=rect, area=area, minimum=minimum):
                        expected_area = pl.area_filter(mask, area)
                        expected_final = pl.elongation_filter(expected_area, minimum, rect)
                        area_mask, final = pl.elongation_filter(
                            mask, minimum, rect, a_min=area, return_area=True)
                        np.testing.assert_array_equal(area_mask, expected_area)
                        np.testing.assert_array_equal(final, expected_final)

    def test_post_bypass_is_identical_in_direct_and_step_modes(self):
        params = {**pl.P, "post": False}
        raw = pl.raw_mask(self.image, params)[0]
        direct = pl.pipeline(self.image, params)
        steps = pl.pipeline(self.image, params, steps=True)
        np.testing.assert_array_equal(direct, pl.to_orig(raw, self.image.shape))
        for key in ("M", "C", "A", "F", "Z"):
            np.testing.assert_array_equal(steps[key], raw)
        self.assertFalse(np.array_equal(direct, pl.pipeline(self.image, pl.P)))

    def test_direct_and_step_modes_agree_for_each_preset(self):
        for params in (pl.P, pl.P_RECON, pl.P_THIN):
            with self.subTest(params=params):
                steps = pl.pipeline(self.image, params, steps=True)
                prediction = pl.pipeline(self.image, params)
                np.testing.assert_array_equal(prediction, pl.to_orig(steps["Z"], self.image.shape))
                self.assertEqual(prediction.shape, self.image.shape[:2])
                self.assertEqual(prediction.dtype, np.bool_)
                self.assertTrue(np.all(steps["seed"] <= steps["M"]))
                self.assertTrue(np.all(steps["M"] <= steps["D"]))

    def test_uniform_images_do_not_create_cracks_or_change_output_shape(self):
        for shape in ((1, 1), (53, 79), (1, 2000), (2000, 1)):
            for value in (0, 128, 255):
                for params in (pl.P, pl.P_RECON):
                    with self.subTest(shape=shape, value=value, params=params):
                        image = np.full(shape + (3,), value, np.uint8)
                        result = pl.pipeline(image, params)
                        self.assertEqual(result.shape, shape)
                        self.assertEqual(result.dtype, np.bool_)
                        self.assertFalse(result.any())

    def test_display_resize_preserves_nonzero_short_dimension(self):
        image = np.zeros((1, 2000), np.uint8)
        self.assertEqual(pl.small(image, size=640).shape, (1, 640))
        self.assertEqual(pl.small(image.T, size=640, nearest=True).shape, (640, 1))

    def test_notebook_code_cells_compile_after_shared_import_refactor(self):
        path = ROOT / "code" / "crack_segmentation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                with self.subTest(cell=index):
                    compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")


if __name__ == "__main__":
    unittest.main()
