"""Frozen twelve-configuration local-contrast development sweep."""
import json

import cv2

import local_contrast as method
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pairs, summarize

OUT = ROOT / "results" / "local-train"


def main():
    if OUT.exists():
        raise ValueError("Existing development run must not be overwritten")
    OUT.mkdir(parents=True)
    (OUT / "configs.json").write_text(json.dumps(method.CONFIGS, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, image, gt) in enumerate(pairs("train")):
        baseline = pl.pipeline(image, pl.P)
        rows.append(dict(source=source, id=sid, method="legacy", area=float(baseline.mean()),
                         **pl.scores(baseline, gt)))
        for config in method.CONFIGS:
            prediction = method.predict(image, config)
            rows.append(dict(source=source, id=sid, method=config["name"], area=float(prediction.mean()),
                             **pl.scores(prediction, gt)))
        if (index + 1) % 25 == 0:
            print(f"{index + 1} development images processed", flush=True)
    write_csv(OUT / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(OUT / "summary.csv", totals)
    for name in ["legacy"] + [c["name"] for c in method.CONFIGS]:
        selected = [r for r in totals if r["method"] == name]
        print(name, [(r["source"], round(float(r["positive_dice"]), 4),
                       r["negative_false_alarms"], round(float(r["negative_mean_area"]), 4))
                      for r in selected], flush=True)


if __name__ == "__main__":
    main()
