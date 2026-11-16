"""Development-only twelve-configuration seeded local-contrast sweep."""
import json

import cv2

import local_contrast as local
import local_fusion as method
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pairs, summarize

OUT = ROOT / "results" / "local-fusion-train"


def main():
    if OUT.exists():
        raise ValueError("Existing run cannot be overwritten")
    OUT.mkdir(parents=True)
    (OUT / "configs.json").write_text(json.dumps(method.CONFIGS, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, image, gt) in enumerate(pairs("train")):
        coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
        baseline = pl.to_orig(coarse, image.shape[:2])
        rows.append(dict(source=source, id=sid, method="legacy", area=float(baseline.mean()),
                         **pl.scores(baseline, gt)))
        supports = {}
        for median in (3, 5):
            for window in (31, 71):
                raw, _ = local.raw_mask(image, dict(median=median, window=window, k=1.5))
                supports[median, window] = pl.post(raw, pl.P)[0]
        for config in method.CONFIGS:
            mask = method.fuse(coarse, supports[config["median"], config["window"]], config["blend"])
            prediction = pl.to_orig(mask, image.shape[:2])
            rows.append(dict(source=source, id=sid, method=config["name"], area=float(prediction.mean()),
                             **pl.scores(prediction, gt)))
        if (index + 1) % 25 == 0:
            print(f"{index + 1} development images processed", flush=True)
    write_csv(OUT / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(OUT / "summary.csv", totals)
    for name in ["legacy"] + [c["name"] for c in method.CONFIGS]:
        subset = [r for r in totals if r["method"] == name]
        print(name, [(r["source"], round(float(r["positive_dice"]), 4),
                       r["negative_false_alarms"], round(float(r["negative_mean_area"]), 4))
                      for r in subset], flush=True)


if __name__ == "__main__":
    main()
