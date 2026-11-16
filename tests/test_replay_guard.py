"""Historical replay integrity checks; never open source photographs or GT."""
import copy
import json
import shutil
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import confirm_local
import replay_guard as guard

ART = ROOT / "docs/benchmarks/new-method"
PROFILES = (("large", "reserve", "reserve-manifest.json"),
            ("optimized", "reserve", "reserve-manifest.json"),
            ("ratio", "ratio-reserve", "ratio-reserve-manifest.json"))


def frozen(name):
    return json.loads((ART / (name + "-freeze.json")).read_text(encoding="utf-8"))


def copy_sources(destination, profile, newline=None):
    destination.mkdir()
    for source in (ART / ("frozen-" + profile)).glob("*.py"):
        payload = source.read_bytes()
        if newline is not None:
            payload = payload.replace(b"\r\n", b"\n").replace(b"\n", newline)
        (destination / source.name).write_bytes(payload)


class HistoricalReplayGuardTests(unittest.TestCase):
    def test_actual_frozen_profiles_and_their_manifests_pass(self):
        for profile, stage, _ in PROFILES:
            record = frozen(profile)
            with self.subTest(profile=profile):
                guard.verify_sources(ART / ("frozen-" + profile), record)
                self.assertTrue(guard.verify_manifest(ART, stage, record)["ids"])
                if profile != "ratio":
                    self.assertEqual(len(guard.verify_manifest(ART, "deepcrack", record)["cases"]), 40)

    def test_lf_and_crlf_source_and_manifest_encodings_both_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            for profile, stage, filename in PROFILES:
                for label, newline in (("lf", b"\n"), ("crlf", b"\r\n")):
                    source_dir = temp / (profile + label)
                    copy_sources(source_dir, profile, newline)
                    guard.verify_sources(source_dir, frozen(profile))
                    manifest_dir = temp / (profile + label + "-manifest")
                    manifest_dir.mkdir()
                    manifests = [(stage, filename)]
                    if profile != "ratio":
                        manifests.append(("deepcrack", "deepcrack-confirmation.json"))
                    for checked_stage, checked_name in manifests:
                        payload = (ART / checked_name).read_bytes().replace(b"\r\n", b"\n")
                        (manifest_dir / checked_name).write_bytes(payload.replace(b"\n", newline))
                        verified = guard.verify_manifest(manifest_dir, checked_stage, frozen(profile))
                        self.assertTrue(verified.get("ids", verified.get("cases")))

    def test_ids_order_or_recorded_crc_mutation_fails_main_before_any_decoder(self):
        for profile, stage, filename in (PROFILES[0], PROFILES[2]):
            original = json.loads((ART / filename).read_text(encoding="utf-8"))
            for mutation in ("ids", "order", "crc"):
                with self.subTest(profile=profile, mutation=mutation), tempfile.TemporaryDirectory() as folder:
                    temp = Path(folder)
                    for name in (profile + "-freeze.json", profile + "-config.json"):
                        shutil.copyfile(ART / name, temp / name)
                    changed = copy.deepcopy(original)
                    if mutation == "ids":
                        changed["ids"][0] += "_changed"
                    elif mutation == "order":
                        changed["ids"].reverse()
                    else:
                        changed["members"][0]["zip_crc32"] ^= 1
                    (temp / filename).write_text(json.dumps(changed, indent=2), encoding="utf-8")
                    argv = ["confirm_local", "--stage", stage, "--freeze", profile + "-freeze.json",
                            "--config", profile + "-config.json", "--source-dir",
                            str(ART / ("frozen-" + profile)), "--output", str(temp / "output")]
                    with patch.object(confirm_local, "ART", temp), patch.object(sys, "argv", argv), \
                            patch.object(confirm_local.cv2, "imread", side_effect=AssertionError("Decoded")) as decoder:
                        with self.assertRaisesRegex(ValueError, "case-selection manifest changed"):
                            confirm_local.main()
                        decoder.assert_not_called()
                        self.assertFalse((temp / "output").exists())

    def test_implementation_mutation_does_not_pass_source_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            source_dir = Path(folder) / "source"
            copy_sources(source_dir, "optimized")
            path = source_dir / "local_fusion.py"
            path.write_bytes(path.read_bytes() + b"\n# changed implementation\n")
            with self.assertRaisesRegex(ValueError, "Frozen implementation changed: local_fusion.py"):
                guard.verify_sources(source_dir, frozen("optimized"))

    def test_early_freeze_omitted_helper_is_pinned_before_any_decoder(self):
        record = frozen("large")
        self.assertNotIn("experiments.py", record["lf_source_sha256"])
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            source_dir = temp / "source"
            copy_sources(source_dir, "large")
            helper = source_dir / "experiments.py"
            helper.write_bytes(helper.read_bytes() + b"\n# changed imported helper\n")
            argv = ["confirm_local", "--stage", "reserve", "--freeze", "large-freeze.json",
                    "--config", "large-config.json", "--source-dir", str(source_dir),
                    "--output", str(temp / "output")]
            with patch.object(sys, "argv", argv), \
                    patch.object(confirm_local.cv2, "imread", side_effect=AssertionError("Decoded")) as decoder:
                with self.assertRaisesRegex(ValueError, "Frozen reconstruction helper changed"):
                    confirm_local.main()
                decoder.assert_not_called()
                self.assertFalse((temp / "output").exists())

    def test_freeze_for_another_manifest_stage_is_rejected(self):
        for stage, profile in (("ratio-reserve", "large"), ("reserve", "ratio"), ("deepcrack", "ratio")):
            with self.subTest(stage=stage, profile=profile), self.assertRaises(ValueError):
                guard.verify_manifest(ART, stage, frozen(profile))

    def test_supplied_verified_manifest_checks_crc_before_decode_without_rereading(self):
        with tempfile.TemporaryDirectory() as folder:
            temp = Path(folder)
            members = []
            for subfolder, suffix, payload in (("images", ".jpg", b"0123456789"),
                                                ("masks", ".png", b"abcdefgh")):
                directory = temp / "train" / subfolder
                directory.mkdir(parents=True)
                (directory / ("fixture" + suffix)).write_bytes(payload)
                members.append(dict(id="fixture", folder=subfolder, uncompressed_bytes=len(payload),
                                    zip_crc32=zlib.crc32(payload)))
            manifest = dict(ids=["fixture"], members=members)
            (temp / "train/images/fixture.jpg").write_bytes(b"9123456789")  # Same size, changed CRC.
            with patch.object(confirm_local, "DATA", temp), patch.object(confirm_local, "ART", temp / "absent"), \
                    patch.object(confirm_local.cv2, "imread", side_effect=AssertionError("Decoded")) as decoder:
                with self.assertRaisesRegex(ValueError, "Source CRC/size mismatch"):
                    next(confirm_local.pairs("reserve", manifest))
                decoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
