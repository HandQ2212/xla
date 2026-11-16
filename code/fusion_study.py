"""Development-only coarse/fine binary reconstruction experiment."""
import json

import cv2
import numpy as np

import experiments as exp
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pairs, summarize

CONFIGS = [dict(name=f"fusion_c{c}_e{e}_{b}", close=c, elongation=e, blend=b)
           for c in (3, 7) for e in (2, 3) for b in ("fine", "union", "supported")]
OUT = ROOT / "results" / "fusion-train"


def fuse(coarse, fine, blend):
    reconstructed = exp.reconstruct(coarse, fine)
    if blend == "union":
        return coarse | reconstructed
    if blend == "supported":
        neighborhood = cv2.dilate(fine.astype(np.uint8), pl.ellipse(3)) > 0
        return reconstructed | (coarse & neighborhood)
    return reconstructed


def main():
    if OUT.exists():
        raise ValueError("Existing run must not be overwritten")
    OUT.mkdir(parents=True)
    (OUT / "configs.json").write_text(json.dumps(CONFIGS, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, image, gt) in enumerate(pairs("train")):
        coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
        baseline = pl.to_orig(coarse, image.shape[:2])
        rows.append(dict(source=source, id=sid, method="legacy", area=float(baseline.mean()),
                         **pl.scores(baseline, gt)))
        raw, _ = pl.raw_mask(image, {**pl.P, "median": 3})
        fine_masks = {(c, e): pl.post(raw, {**pl.P, "close": c, "e_min": e, "tmin": 0})[0]
                      for c in (3, 7) for e in (2, 3)}
        for config in CONFIGS:
            mask = fuse(coarse, fine_masks[config["close"], config["elongation"]], config["blend"])
            prediction = pl.to_orig(mask, image.shape[:2])
            if not baseline.any() and prediction.any():
                raise ValueError("Fusion created a new positive image")
            rows.append(dict(source=source, id=sid, method=config["name"], area=float(prediction.mean()),
                             **pl.scores(prediction, gt)))
        if (index + 1) % 25 == 0:
            print(f"{index + 1} development images processed", flush=True)
    write_csv(OUT / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(OUT / "summary.csv", totals)
    for name in ["legacy"] + [c["name"] for c in CONFIGS]:
        subset = [r for r in totals if r["method"] == name]
        print(name, [(r["source"], round(float(r["positive_dice"]), 4),
                       r["negative_false_alarms"], round(float(r["negative_mean_area"]), 4))
                      for r in subset], flush=True)


if __name__ == "__main__":
    main()
