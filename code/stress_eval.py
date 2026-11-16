"""Fixed discovery comparisons and baseline-only confirmation on PaveSeg preview."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

import external_data as data
import pipeline as pl
import experiments as exp
from benchmark import write_csv
from stress_metrics import gate, summary

VARIANTS = {"legacy": pl.P, "reconstruct": exp.P_RECON, "thin": exp.P_THIN}
TRANSFORMS = ("original", "flip", "gamma075", "gamma125", "jpeg60", "shadow")


def transform(image, label, name):
    if name == "original":
        return image, label
    if name == "flip":
        return cv2.flip(image, 1), cv2.flip(label, 1)
    if name.startswith("gamma"):
        gamma = {"gamma075": .75, "gamma125": 1.25}[name]
        lut = np.clip(255 * (np.arange(256) / 255) ** gamma, 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut), label
    if name == "jpeg60":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ok:
            raise ValueError("JPEG encoding failed")
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR), label
    if name == "shadow":
        h, w = label.shape
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        diagonal = xx / max(w - 1, 1) + .4 * yy / max(h - 1, 1)
        band = (np.tanh((diagonal - .4) / .025) - np.tanh((diagonal - .8) / .025)) / 2
        illumination = 1 - .65 * band
        return np.clip(image * illumination[..., None], 0, 255).astype(np.uint8), label
    raise ValueError(f"Unknown transformation: {name}")


def evaluate(manifest, split, variants, output):
    selected = [c for c in manifest["cases"] if c["split"] == split]
    if not selected:
        raise ValueError(f"Empty split: {split}")
    rows = []
    for index, case in enumerate(selected):
        image, label = data.read_case(case)
        for name in TRANSFORMS:
            img, lab = transform(image, label, name)
            gt = np.isin(lab, manifest["positive_labels"])
            for variant in variants:
                params = VARIANTS[variant]
                exp.pipeline(img, params)  # Untimed warmup, identical for every method.
                times = []
                for _ in range(3):
                    started = time.perf_counter()
                    prediction = exp.pipeline(img, params)
                    times.append((time.perf_counter() - started) * 1000)
                if prediction.dtype != np.bool_ or prediction.shape != gt.shape:
                    raise ValueError(f"Invalid output: {case['id']}/{variant}/{name}")
                metrics = pl.scores(prediction, gt)
                row = dict(id=case["id"], group=case["group"], split=split,
                           tags="|".join(case["tags"]), transform=name, variant=variant,
                           **metrics, area_pct=float(prediction.mean() * 100),
                           elapsed_ms=float(np.median(times)))
                for value, tag in data.LABELS.items():
                    if tag in {"sealed", "patch", "cover", "joint"}:
                        region = lab == value
                        row[tag + "_pixels"] = int(region.sum())
                        row[tag + "_fp"] = int((prediction & region).sum())
                rows.append(row)
                folder = output / "predictions" / name / variant
                folder.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(folder / (case["id"] + ".png")), pl.u8(prediction)):
                    raise OSError("Could not save prediction")
                if name == "original":
                    if not cv2.imwrite(str(folder / (case["id"] + "_overlay.jpg")),
                                       pl.overlay_red(img, prediction)):
                        raise OSError("Could not save overlay")
        if (index + 1) % 10 == 0:
            print(f"{split}: {index + 1}/{len(selected)} originals processed", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data.MANIFEST)
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--output", type=Path, help="Default: results/external-<split>")
    args = parser.parse_args()
    if len(args.variants) != len(set(args.variants)):
        parser.error("Variants must be unique")
    if any(v != "legacy" for v in args.variants) and "legacy" not in args.variants:
        parser.error("Candidate comparisons require the legacy baseline")
    if args.split == "confirmation" and args.variants != ["legacy"]:
        parser.error("This iteration reserves confirmation for baseline diagnostics after candidate rejection")
    if args.output is None:
        args.output = data.ROOT / "results" / ("external-" + args.split)
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = evaluate(manifest, args.split, args.variants, args.output)
    write_csv(args.output / "per_image.csv", rows)
    totals = summary(rows)
    write_csv(args.output / "summary.csv", totals)
    (args.output / "gate.json").write_text(json.dumps(gate(rows), indent=2) + "\n", encoding="utf-8")
    metadata = dict(split=args.split, variants={k: VARIANTS[k] for k in args.variants},
                    transforms=TRANSFORMS, opencv=cv2.__version__, numpy=np.__version__, threads=1,
                    python=platform.python_version(), platform=platform.system(),
                    manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    manifest_lf_sha256=hashlib.sha256(args.manifest.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                    pipeline_sha256=hashlib.sha256(Path(pl.__file__).read_bytes()).hexdigest(),
                    pipeline_lf_sha256=hashlib.sha256(Path(pl.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                    experiments_sha256=hashlib.sha256(Path(exp.__file__).read_bytes()).hexdigest(),
                    timings="One warmup and median of three pipeline-only runs per image/method/transform",
                    limitation=manifest["limitation"])
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for row in totals:
        print(f"{row['transform']:10} {row['variant']:12} positive Dice={row['positive_dice']:.4f} "
              f"FP negatives={row['negative_false_alarms']}/{row['negatives']}", flush=True)


if __name__ == "__main__":
    main()
