"""One fixed ranked-local candidate; validation/test must use the frozen config."""
import argparse
import hashlib
import json

import cv2
import numpy as np

import local_fusion as method
import pipeline as pl
from benchmark import write_csv
from native_study import ROOT, pairs, summarize

CONFIG = dict(name="ranked_local", median=3, window=71, k=1.5, blend="supported", area_guard="ranked")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("train", "val", "test"), required=True)
    parser.add_argument("--config", type=str)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    config = CONFIG
    if args.stage != "train":
        if not args.config:
            parser.error("Validation/test requires the previously frozen single config")
        config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = ROOT / args.output
    if out.exists():
        raise ValueError("Existing results must not be overwritten")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    files = [ROOT / "code" / name for name in ("pipeline.py", "local_contrast.py", "local_fusion.py")]
    (out / "code_hashes.json").write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                     for p in files}, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, image, gt) in enumerate(pairs(args.stage)):
        baseline = pl.pipeline(image, pl.P)
        prediction = method.predict(image, config)
        if prediction.sum() > baseline.sum() or (not baseline.any() and prediction.any()):
            raise ValueError("Area or image-alarm invariant violated")
        for name, mask in [("legacy", baseline), (config["name"], prediction)]:
            rows.append(dict(source=source, id=sid, method=name, area=float(mask.mean()),
                             mask_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                             height=mask.shape[0], width=mask.shape[1],
                             **pl.scores(mask, gt)))
        if (index + 1) % 25 == 0:
            print(f"{args.stage}: {index + 1} images", flush=True)
    write_csv(out / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(out / "summary.csv", totals)
    for row in totals:
        print(row, flush=True)


if __name__ == "__main__":
    main()
