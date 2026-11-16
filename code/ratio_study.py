"""Frozen12-config development sweep on150train originals plus fixedshadow."""
import json

import cv2

import pipeline as pl
import ratio_local as method
from benchmark import write_csv
from native_study import ROOT, pcs_pairs, summarize
from stress_eval import transform

OUT = ROOT / "results" / "ratio-local-train"


def main():
    if OUT.exists():
        raise ValueError("Existing run cannot be overwritten")
    OUT.mkdir(parents=True)
    (OUT / "configs.json").write_text(json.dumps(method.CONFIGS, indent=2), encoding="utf-8")
    manifest = json.loads((ROOT / "docs/benchmarks/new-method/train-manifest.json").read_text(encoding="utf-8"))
    cv2.setNumThreads(1)
    rows = []
    for index, (_, sid, image, gt) in enumerate(pcs_pairs("train", manifest["ids"])):
        for condition in ("original", "shadow"):
            img, label = transform(image, gt.astype("uint8") * 255, condition)
            target = label > 127
            stages = pl.pipeline(img, pl.P, steps=True)
            baseline = pl.to_orig(stages["Z"], img.shape[:2])
            rows.append(dict(source="train_" + condition, id=sid, method="legacy", area=float(baseline.mean()),
                             **pl.scores(baseline, target)))
            for config in method.CONFIGS:
                prediction = method.refine(stages, img.shape[:2], config)
                if prediction.any() != baseline.any() or prediction.sum() > baseline.sum():
                    raise ValueError("Detection or area invariant broken")
                rows.append(dict(source="train_" + condition, id=sid, method=config["name"],
                                 area=float(prediction.mean()), **pl.scores(prediction, target)))
        if (index + 1) % 25 == 0:
            print(f"{index + 1} train originals and shadows", flush=True)
    write_csv(OUT / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(OUT / "summary.csv", totals)
    for name in ["legacy"] + [c["name"] for c in method.CONFIGS]:
        subset = [r for r in totals if r["method"] == name]
        print(name, [(r["source"], round(float(r["positive_dice"]), 4),
                       round(float(r["precision"]), 4), round(float(r["recall"]), 4)) for r in subset], flush=True)


if __name__ == "__main__":
    main()
