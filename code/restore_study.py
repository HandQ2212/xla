"""Prospective restoration-only train sweep; detection masks never change."""
import json

import cv2

import mask_restore as method
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pcs_pairs, summarize
from stress_eval import transform

OUT = ROOT / "results" / "restore-train"


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
            for name, mask in [("legacy", pl.to_orig(coarse, img.shape[:2]))] + [
                    (c["name"], method.restore(coarse, img.shape[:2], c)) for c in method.CONFIGS]:
                rows.append(dict(source="train_" + condition, id=sid, method=name, area=float(mask.mean()),
                                 **pl.scores(mask, target)))
        if (index + 1) % 25 == 0:
            print(f"{index + 1} originals and shadows", flush=True)
    write_csv(OUT / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(OUT / "summary.csv", totals)
    for row in totals:
        print(row, flush=True)


if __name__ == "__main__":
    main()
