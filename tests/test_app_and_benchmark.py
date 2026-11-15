"""App integration and dataset-validation tests; no browser or network needed."""
import logging
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import benchmark
import pipeline as pl


class AppTests(unittest.TestCase):
    def test_original_size_metrics_post_ablation_and_no_candidate_selector(self):
        # Run by absolute path from repository root, rather than relying on cd code.
        app = AppTest.from_file(str(ROOT / "code" / "app.py"), default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.button), 10)
        self.assertEqual(len(app.selectbox), 0)
        app.button(key="sample_B_00038.jpg").click().run()
        self.assertEqual(len(app.exception), 0)
        image = cv2.imread(str(ROOT / "samples_pcs" / "B_00038.jpg"))
        gt = cv2.imread(str(ROOT / "samples_pcs" / "B_00038_gt.png"), 0) > 127
        self.assertIn("1702×866", app.get("imgs")[0].proto.imgs[0].caption)
        expected = pl.scores(pl.pipeline(image, pl.P), gt)
        self.assertEqual(app.metric[0].value, f"{expected['dice']:.3f}")
        self.assertEqual(app.metric[4].value, f"{gt.sum()} px nứt")

        app.checkbox[3].uncheck().run()
        self.assertEqual(len(app.exception), 0)
        bypass = pl.scores(pl.pipeline(image, {**pl.P, "post": False}), gt)
        self.assertEqual(app.metric[0].value, f"{bypass['dice']:.3f}")
        self.assertNotEqual(app.metric[0].value, f"{expected['dice']:.3f}")

        app.checkbox[3].check().run()
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(app.metric[0].value, f"{expected['dice']:.3f}")

    def test_upload_named_like_sample_does_not_get_sample_ground_truth(self):
        app = AppTest.from_file(str(ROOT / "code" / "app.py"), default_timeout=15).run()
        app.session_state["img_data"] = (ROOT / "samples_pcs" / "B_00038.jpg").read_bytes()
        app.session_state["img_name"] = "B_00038.jpg"
        app.session_state["is_sample"] = False
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 0)


class DatasetTests(unittest.TestCase):
    def test_missing_val_never_falls_back_to_test(self):
        with tempfile.TemporaryDirectory() as folder:
            images = Path(folder) / "test" / "images"
            images.mkdir(parents=True)
            cv2.imwrite(str(images / "example.jpg"), np.zeros((5, 7, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "fallback"):
                list(benchmark.pairs(folder, "val"))

    def test_missing_gt_and_wrong_mask_shape_fail_instead_of_resizing(self):
        with tempfile.TemporaryDirectory() as folder:
            split = Path(folder) / "val"
            (split / "images").mkdir(parents=True)
            (split / "masks").mkdir()
            cv2.imwrite(str(split / "images" / "example.jpg"), np.zeros((5, 7, 3), np.uint8))
            with self.assertRaises(ValueError):
                list(benchmark.pairs(folder, "val"))
            cv2.imwrite(str(split / "masks" / "example.png"), np.zeros((7, 5), np.uint8))
            with self.assertRaisesRegex(ValueError, "khác nhau"):
                list(benchmark.pairs(folder, "val"))


if __name__ == "__main__":
    logging.disable(logging.WARNING)
    unittest.main()
