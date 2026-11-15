"""Integrity, label-preservation and decision-gate checks; no network required."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import external_data as data
import stress_eval
from stress_metrics import gate, summary


def metric_row(sid, positive, variant="legacy", dice=.5, area=0.):
    """Controlled numerical fixture, never part of the external evaluation dataset."""
    row = dict(id=sid, group="block", tags="crack" if positive else "sealed",
               transform="original", variant=variant, gt_pos=positive,
               pred_pos=positive or area > 0, dice=dice, iou=dice / (2 - dice),
               precision=.7, recall=.7, tp=5 if positive else 0,
               fp=5 if positive else int(area > 0), fn=5 if positive else 0,
               area_pct=area, elapsed_ms=1.)
    for tag in ("sealed", "patch", "cover", "joint"):
        row[tag + "_pixels"] = 0
        row[tag + "_fp"] = 0
    return row


def gate_fixture():
    return [metric_row(sid, positive, variant, dice=.5 if variant == "legacy" else .65)
            for variant in ("legacy", "candidate")
            for sid, positive in (("p1", True), ("p2", True), ("n1", False), ("n2", False))]


class TransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = cv2.imread(str(ROOT / "samples_pcs" / "B_00038.jpg"))
        cls.label = cv2.imread(str(ROOT / "samples_pcs" / "B_00038_gt.png"), 0)

    def test_photometric_changes_preserve_exact_source_ground_truth(self):
        original_image, original_label = self.image.copy(), self.label.copy()
        for name in ("original", "gamma075", "gamma125", "jpeg60", "shadow"):
            with self.subTest(name=name):
                image, label = stress_eval.transform(self.image, self.label, name)
                self.assertEqual(image.shape, self.image.shape)
                self.assertEqual(image.dtype, np.uint8)
                np.testing.assert_array_equal(label, original_label)
                if name != "original":
                    self.assertFalse(np.array_equal(image, original_image))
        np.testing.assert_array_equal(self.image, original_image)
        np.testing.assert_array_equal(self.label, original_label)

    def test_horizontal_flip_moves_image_and_label_together(self):
        image, label = stress_eval.transform(self.image, self.label, "flip")
        np.testing.assert_array_equal(image, self.image[:, ::-1])
        np.testing.assert_array_equal(label, self.label[:, ::-1])
        self.assertEqual(np.count_nonzero(label), np.count_nonzero(self.label))


class IntegrityTests(unittest.TestCase):
    def test_source_bytes_must_match_hash_before_processing(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "img_preview").mkdir()
            (base / "label_preview").mkdir()
            ip = base / "img_preview" / "fixture.jpg"
            lp = base / "label_preview" / "fixture_mask.png"
            ip.write_bytes((ROOT / "samples_pcs" / "B_00038.jpg").read_bytes())
            lp.write_bytes((ROOT / "samples_pcs" / "B_00038_gt.png").read_bytes())
            case = dict(id="fixture", image_sha256=hashlib.sha256(ip.read_bytes()).hexdigest(),
                        label_sha256=hashlib.sha256(lp.read_bytes()).hexdigest())
            image, label = data.read_case(case, data=base)
            self.assertEqual(image.shape[:2], label.shape)
            np.testing.assert_array_equal(label, cv2.imread(str(lp), 0))
            lp.write_bytes(lp.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Source hash mismatch"):
                data.read_case(case, data=base)

    def test_frozen_manifest_is_pinned_and_groups_do_not_cross_splits(self):
        manifest = json.loads(data.MANIFEST.read_text(encoding="utf-8"))
        cases = manifest["cases"]
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({case["id"] for case in cases}), 100)
        self.assertEqual(manifest["positive_labels"], [30, 150])
        discovery = {c["group"] for c in cases if c["split"] == "discovery"}
        confirmation = {c["group"] for c in cases if c["split"] == "confirmation"}
        self.assertFalse(discovery & confirmation)
        for case in cases:
            self.assertIn(manifest["commit"], case["image_url"])
            self.assertIn(manifest["commit"], case["label_url"])


class DecisionGateTests(unittest.TestCase):
    def test_negative_dice_never_inflates_positive_summary(self):
        rows = [metric_row("p", True, dice=.2), metric_row("n", False, dice=1.)]
        result = summary(rows)[0]
        self.assertEqual(result["positive_dice"], .2)
        self.assertEqual(result["negative_false_alarms"], 0)
        self.assertEqual(result["positives"], 1)
        self.assertEqual(result["negatives"], 1)

    def test_new_negative_false_alarm_rejects_large_positive_gain(self):
        rows = gate_fixture()
        self.assertTrue(gate(rows)[0]["numerical_pass"])
        candidate = next(r for r in rows if r["id"] == "n1" and r["variant"] == "candidate")
        candidate.update(pred_pos=True, area_pct=.01)
        result = gate(rows)[0]
        self.assertGreater(result["positive_dice_gain"], .1)
        self.assertFalse(result["numerical_pass"])
        self.assertIn("new_negative_false_alarm", result["failures"])

    def test_mean_gain_does_not_hide_severe_individual_regression(self):
        rows = gate_fixture()
        for row in rows:
            if row["variant"] == "candidate" and row["gt_pos"]:
                row["dice"] = .44 if row["id"] == "p1" else .85
                row["iou"] = row["dice"] / (2 - row["dice"])
        result = gate(rows)[0]
        self.assertGreater(result["positive_dice_gain"], .1)
        self.assertIn("individual_dice_loss_over_0.05", result["failures"])
        self.assertFalse(result["numerical_pass"])

    def test_expanding_existing_negative_mask_is_rejected(self):
        rows = gate_fixture()
        for row in rows:
            if row["id"] == "n1":
                row.update(pred_pos=True, area_pct=1. if row["variant"] == "legacy" else 1.1)
        result = gate(rows)[0]
        self.assertNotIn("new_negative_false_alarm", result["failures"])
        self.assertIn("negative_mean_area_pct_increased", result["failures"])
        self.assertIn("negative_max_area_pct_increased", result["failures"])
        self.assertFalse(result["numerical_pass"])


class CommandLineTests(unittest.TestCase):
    def test_candidates_require_baseline_before_reading_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "missing-manifest.json"
            for candidate in ("thin", "reconstruct"):
                with self.subTest(candidate=candidate):
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "code" / "stress_eval.py"),
                         "--manifest", str(missing), "--variants", candidate],
                        capture_output=True, text=True, encoding="utf-8", timeout=15)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("require the legacy baseline", result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_variants_fail_before_reading_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            result = subprocess.run(
                [sys.executable, str(ROOT / "code" / "stress_eval.py"), "--manifest",
                 str(Path(folder) / "missing-manifest.json"), "--variants", "legacy", "legacy"],
                capture_output=True, text=True, encoding="utf-8", timeout=15)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Variants must be unique", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
