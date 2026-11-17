"""Development-only sweep of the prospectively enumerated symmetric-line candidates."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

import external_data as external
import line_response as line
import pipeline as pl
from benchmark import write_csv

ROOT = external.ROOT
OUT = ROOT / "results" / "line-study"


def development_pairs():
    for source in ("samples", "samples_pcs"):
        for path in sorted((ROOT / source).glob("*.jpg")):
            if path.stem.endswith("_pred"):
                continue
            yield source, path.stem, cv2.imread(str(path)), cv2.imread(str(path.with_name(path.stem + "_gt.png")), 0) > 127
    manifest = json.loads(external.MANIFEST.read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        image, label = external.read_case(case)
        yield "paveseg_seen", case["id"], image, np.isin(label, [30, 150])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--wave", choices=("line", "gate"), default="line")
    args = parser.parse_args()
    output = args.output
    if args.wave == "gate":
        line.CONFIGS = [dict(name=f"gate_w{w}_p{p}_f{f}", method="gate", widths=w, persistence=p, fraction=f)
                        for w in (2, 3) for p in (3, 7) for f in (.2, .4, .6)]
    output.mkdir(parents=True, exist_ok=True)
    frozen = dict(configs=line.CONFIGS, core=pl.P,
                  prototype_sha256=hashlib.sha256(Path(line.__file__).read_bytes()).hexdigest(),
                  scope="All previously observed images are development only; no held-out claim")
    config_path = output / "configs.json"
    if config_path.exists():
        raise ValueError("Refusing to replace a study run; move the old directory before another iteration")
    config_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, image, gt) in enumerate(development_pairs()):
        stages = pl.pipeline(image, pl.P, steps=True)
        old = pl.to_orig(stages["Z"], image.shape[:2])
        keys = {(c["widths"], c.get("persistence", 3)) for c in line.CONFIGS}
        cache = (stages["Z"], {key: line.response(stages["small"], *key) for key in keys})
        rows.append(dict(source=source, id=sid, method="legacy", pixels=gt.size, **pl.scores(old, gt)))
        for config in line.CONFIGS:
            prediction = line.predict(image, config, cached=cache)
            rows.append(dict(source=source, id=sid, method=config["name"], pixels=gt.size, **pl.scores(prediction, gt)))
        if (index + 1) % 20 == 0:
            print(f"{index + 1} development images processed", flush=True)
    write_csv(output / "per_image.csv", rows)
    totals = []
    for source in sorted({r["source"] for r in rows}):
        for method in ["legacy"] + [c["name"] for c in line.CONFIGS]:
            subset = [r for r in rows if r["source"] == source and r["method"] == method]
            pos = [r for r in subset if r["gt_pos"]]
            neg = [r for r in subset if not r["gt_pos"]]
            totals.append(dict(source=source, method=method, positive_dice=np.mean([r["dice"] for r in pos]),
                               positive_iou=np.mean([r["iou"] for r in pos]),
                               precision=np.mean([r["precision"] for r in pos]),
                               recall=np.mean([r["recall"] for r in pos]),
                               negative_false_alarms=sum(r["pred_pos"] for r in neg), negatives=len(neg),
                               negative_mean_area=np.mean([(r["tp"] + r["fp"]) / r["pixels"] for r in neg]) if neg else 0))
    write_csv(output / "summary.csv", totals)
    for method in ["legacy"] + [c["name"] for c in line.CONFIGS]:
        subset = [r for r in totals if r["method"] == method]
        print(method, [(r["source"], round(float(r["positive_dice"]), 4),
                        r["negative_false_alarms"]) for r in subset], flush=True)


if __name__ == "__main__":
    started = time.perf_counter()
    main()
    print(f"Total {time.perf_counter() - started:.1f}s")
