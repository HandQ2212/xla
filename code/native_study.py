"""Prospective train/validation/test evaluation of native-resolution refinement."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import mask_refinement as method
import pipeline as pl
from benchmark import write_csv
from line_study import development_pairs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Pavement_Crack_Segmentation_Dataset"


def pcs_pairs(split, ids):
    for sid in ids:
        image = cv2.imread(str(DATA / split / "images" / (sid + ".jpg")))
        label = cv2.imread(str(DATA / split / "masks" / (sid + ".png")), 0)
        if image is None or label is None or image.shape[:2] != label.shape:
            raise ValueError(f"Invalid/missing pair: {split}/{sid}")
        yield "pcs_" + split, sid, image, label > 127


def pairs(stage):
    if stage == "train":
        yield from development_pairs()
        manifest = json.loads((ROOT / "docs/benchmarks/new-method/train-manifest.json").read_text(encoding="utf-8"))
        yield from pcs_pairs("train", manifest["ids"])
    else:
        all_ids = sorted(p.stem for p in (DATA / stage / "images").glob("*.jpg"))
        expected = {"val": 498, "test": 231}[stage]
        if len(all_ids) != expected:
            raise ValueError(f"Expected {expected} complete {stage} images, found {len(all_ids)}")
        excluded = {p.stem for p in (ROOT / "samples_pcs").glob("*.jpg") if not p.stem.endswith("_pred")}
        yield from pcs_pairs(stage, [sid for sid in all_ids if stage != "test" or sid not in excluded])


def summarize(rows):
    output = []
    for source in sorted({r["source"] for r in rows}):
        for name in sorted({r["method"] for r in rows}):
            subset = [r for r in rows if r["source"] == source and r["method"] == name]
            pos = [r for r in subset if r["gt_pos"]]
            neg = [r for r in subset if not r["gt_pos"]]
            output.append(dict(source=source, method=name, images=len(subset), positives=len(pos),
                               dice=np.mean([r["dice"] for r in subset]),
                               positive_dice=np.mean([r["dice"] for r in pos]),
                               positive_iou=np.mean([r["iou"] for r in pos]),
                               precision=np.mean([r["precision"] for r in pos]),
                               recall=np.mean([r["recall"] for r in pos]),
                               negative_false_alarms=sum(r["pred_pos"] for r in neg), negatives=len(neg),
                               negative_mean_area=np.mean([r["area"] for r in neg]) if neg else 0))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "val", "test"), required=True)
    parser.add_argument("--config", type=Path, help="Frozen single configuration, required for val/test")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage != "train" and args.config is None:
        parser.error("Validation/test requires a frozen single configuration")
    configs = [json.loads(args.config.read_text(encoding="utf-8"))] if args.config else method.CONFIGS
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing run")
    args.output.mkdir(parents=True)
    metadata = dict(stage=args.stage, configs=configs, legacy=pl.P,
                    implementation_sha256=hashlib.sha256(Path(method.__file__).read_bytes()).hexdigest(),
                    opencv=cv2.__version__, numpy=np.__version__, threads=1)
    (args.output / "configs.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    cv2.setNumThreads(1)
    for index, (source, sid, image, gt) in enumerate(pairs(args.stage)):
        coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
        baseline = pl.to_orig(coarse, image.shape[:2])
        rows.append(dict(source=source, id=sid, method="legacy", area=float(baseline.mean()),
                         **pl.scores(baseline, gt)))
        for config in configs:
            prediction = method.refine(image, coarse, config)
            if np.any(prediction & ~baseline):
                raise ValueError("Refinement violated the no-new-foreground invariant")
            rows.append(dict(source=source, id=sid, method=config["name"], area=float(prediction.mean()),
                             **pl.scores(prediction, gt)))
        if (index + 1) % 25 == 0:
            print(f"{args.stage}: {index + 1} images", flush=True)
    write_csv(args.output / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(args.output / "summary.csv", totals)
    for name in ["legacy"] + [c["name"] for c in configs]:
        subset = [r for r in totals if r["method"] == name]
        print(name, [(r["source"], round(float(r["positive_dice"]), 4),
                       round(float(r["precision"]), 4), round(float(r["recall"]), 4)) for r in subset], flush=True)


if __name__ == "__main__":
    main()
