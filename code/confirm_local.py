"""Replay frozen local candidates and fixed stresses on previously scored sources."""
import argparse
import hashlib
import importlib
import json
import sys
import time
import zlib

import cv2
import numpy as np

from benchmark import write_csv
from native_study import ROOT, DATA, summarize
from stress_eval import TRANSFORMS, transform
from replay_guard import verify_manifest, verify_sources

ART = ROOT / "docs/benchmarks/new-method"


def pairs(stage, manifest=None):
    filename = {"reserve": "reserve-manifest.json", "ratio-reserve": "ratio-reserve-manifest.json",
                "deepcrack": "deepcrack-confirmation.json"}[stage]
    if manifest is None:
        manifest = json.loads((ART / filename).read_text(encoding="utf-8"))
    if stage != "deepcrack":
        entries = {(m["id"], m["folder"]): m for m in manifest["members"]}
        for sid in manifest["ids"]:
            for folder, suffix in [("images", ".jpg"), ("masks", ".png")]:
                payload = (DATA / "train" / folder / (sid + suffix)).read_bytes()
                expected = entries[sid, folder]
                if len(payload) != expected["uncompressed_bytes"] or zlib.crc32(payload) != expected["zip_crc32"]:
                    raise ValueError(f"Source CRC/size mismatch: {sid}/{folder}")
            image = cv2.imread(str(DATA / "train/images" / (sid + ".jpg")))
            label = cv2.imread(str(DATA / "train/masks" / (sid + ".png")), 0)
            if image is None or label is None or image.shape[:2] != label.shape:
                raise ValueError(f"Invalid reserve pair: {sid}")
            yield sid, image, label
    else:
        for case in manifest["cases"]:
            paths = [(ROOT / "data/external/deepcrack" / case[k], case[k + "_sha256"])
                     for k in ("image", "mask")]
            for path, expected in paths:
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError("DeepCrack source checksum mismatch")
            image, label = cv2.imread(str(paths[0][0])), cv2.imread(str(paths[1][0]), 0)
            if image is None or label is None or image.shape[:2] != label.shape:
                raise ValueError(f"Invalid DeepCrack pair: {case['id']}")
            if not set(np.unique(label)).issubset({0, 255}):
                raise ValueError("Unexpected DeepCrack binary label values")
            yield case["id"], image, label


def paired(rows):
    results = []
    for source in sorted({r["source"] for r in rows}):
        baseline = {r["id"]: r for r in rows if r["source"] == source and r["method"] == "legacy"}
        candidate = [r for r in rows if r["source"] == source and r["method"] != "legacy"]
        positive = [r for r in candidate if r["gt_pos"]]
        delta = np.array([r["dice"] - baseline[r["id"]]["dice"] for r in positive])
        rng = np.random.default_rng(42)
        means = np.mean(delta[rng.integers(0, len(delta), (10000, len(delta)))], axis=1)
        losses = sorted([(r["id"], r["dice"] - baseline[r["id"]]["dice"]) for r in positive], key=lambda p: p[1])
        results.append(dict(source=source, positive_images=len(positive),
                            positive_dice_gain=float(delta.mean()),
                            image_bootstrap_95ci=np.quantile(means, [.025, .975]).tolist(),
                            wins=int((delta > 0).sum()), losses=int((delta < 0).sum()),
                            individual_losses_over_005=[dict(id=sid, delta=d) for sid, d in losses if d < -.05],
                            limitation="Image bootstrap only; capture-session independence unknown"))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("reserve", "ratio-reserve", "deepcrack"), required=True)
    parser.add_argument("--conditions", nargs="+", choices=TRANSFORMS, default=["original"])
    parser.add_argument("--repeats", type=int, choices=(1, 3), default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="conservative-config.json")
    parser.add_argument("--freeze", default="conservative-freeze.json")
    parser.add_argument("--source-dir", default="code", help="Frozen source snapshot directory, relative to repo")
    parser.add_argument("--engine", choices=("local_fusion", "ratio_local"), default="local_fusion")
    args = parser.parse_args()
    config = json.loads((ART / args.config).read_text(encoding="utf-8"))
    frozen = json.loads((ART / args.freeze).read_text(encoding="utf-8"))
    if config != frozen["config"]:
        raise ValueError("Parameters differ from the frozen configuration")
    source_dir = ROOT / args.source_dir
    manifest = verify_manifest(ART, args.stage, frozen)
    verify_sources(source_dir, frozen)
    if source_dir.resolve() != (ROOT / "code").resolve():
        for name in ("local_fusion", "ratio_local", "local_contrast", "experiments", "pipeline"):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(source_dir))
    pl = importlib.import_module("pipeline")
    method = importlib.import_module(args.engine)
    out = ROOT / args.output
    if out.exists():
        raise ValueError("Refusing to replace confirmation evidence")
    out.mkdir(parents=True)
    metadata = dict(stage=args.stage, config=config, conditions=args.conditions, repeats=args.repeats,
                    warmup=True, threads=1, opencv=cv2.__version__, numpy=np.__version__,
                    frozen=frozen, scope="Historical frozen replay; these cases have already been observed")
    metadata["source_dir"] = args.source_dir
    metadata["engine"] = args.engine
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cv2.setNumThreads(1)
    rows = []
    for index, (sid, image, label) in enumerate(pairs(args.stage, manifest)):
        for condition in args.conditions:
            img, lab = transform(image, label, condition)
            gt = lab > 127
            predictions = {}
            for name, predictor in [("legacy", lambda: pl.pipeline(img, pl.P)),
                                    ("conservative_local", lambda: method.predict(img, config))]:
                predictor()
                times = []
                for _ in range(args.repeats):
                    started = time.perf_counter()
                    mask = predictor()
                    times.append((time.perf_counter() - started) * 1000)
                if mask.shape != gt.shape or mask.dtype != np.bool_:
                    raise ValueError("Invalid original-resolution bool output")
                predictions[name] = mask
                rows.append(dict(source=args.stage + "_" + condition, id=sid, method=name,
                                 area=float(mask.mean()), elapsed_ms=float(np.median(times)),
                                 height=mask.shape[0], width=mask.shape[1],
                                 mask_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(),
                                 **pl.scores(mask, gt)))
            baseline, improved = predictions["legacy"], predictions["conservative_local"]
            if baseline.any() != improved.any() or improved.sum() > baseline.sum():
                raise ValueError("Area/detection-preservation invariant failed")
        if (index + 1) % 20 == 0:
            print(f"{args.stage}: {index + 1} originals", flush=True)
    write_csv(out / "per_image.csv", rows)
    totals = summarize(rows)
    write_csv(out / "summary.csv", totals)
    (out / "paired.json").write_text(json.dumps(paired(rows), indent=2), encoding="utf-8")
    for row in totals:
        print(row, flush=True)
    print(paired(rows), flush=True)


if __name__ == "__main__":
    main()
