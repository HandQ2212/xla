"""Frozen12-config multiscale development sweep on150train plus fixedshadow."""
import json

import cv2

import multiscale as method
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pcs_pairs, summarize
from stress_eval import transform

OUT = ROOT / "results" / "multiscale-train"


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
            coarse = pl.pipeline(img, pl.P, steps=True)["Z"]
            baseline = pl.to_orig(coarse, img.shape[:2])
            rows.append(dict(source="train_" + condition, id=sid, method="legacy", area=float(baseline.mean()),
                             **pl.scores(baseline, target)))
            cache = {}
            for config in method.CONFIGS:
                key = config["fine_size"], config["geometry"]
                if key not in cache:
                    params = method.fine_params(img, config)
                    cache[key] = params, pl.pipeline(img, params, steps=True)["Z"]
                params, fine = cache[key]
                prediction = method.refine(img, coarse, fine, config, params)
                if prediction.any() != baseline.any():
                    raise ValueError("Image decision changed")
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
                       round(float(r["precision"]), 4), round(float(r["recall"]), 4),
                       round(float(r["negative_mean_area"]), 4)) for r in subset], flush=True)


if __name__ == "__main__":
    main()
