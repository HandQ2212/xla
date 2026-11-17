"""Pixel parity on already observed sources; never opens the final 158 reserve."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

import pipeline as optimized
from benchmark import write_csv
from stress_eval import TRANSFORMS, transform

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs/benchmarks/new-method"


def known_images():
    ids = set()
    for name in ("train-manifest.json", "reserve-manifest.json", "ratio-reserve-manifest.json"):
        ids.update(json.loads((ART / name).read_text(encoding="utf-8"))["ids"])
    unopened = set(json.loads((ART / "multiscale-reserve-manifest.json").read_text(encoding="utf-8"))["ids"])
    if ids & unopened or len(ids) != 1050:
        raise ValueError("Invalid observed/unopened split")
    for sid in sorted(ids):
        yield "pcs_train_seen", sid, ROOT / "data/Pavement_Crack_Segmentation_Dataset/train/images" / (sid + ".jpg")
    for source in ("samples", "samples_pcs"):
        for path in sorted((ROOT / source).glob("*.jpg")):
            if not path.stem.endswith("_pred"):
                yield source, path.stem, path
    for i in range(1, 101):
        yield "paveseg_seen", f"{i:06d}", ROOT / f"data/external/paveseg/img_preview/{i:06d}.jpg"
    manifest = json.loads((ART / "deepcrack-confirmation.json").read_text(encoding="utf-8"))
    for case in manifest["cases"]:
        yield "deepcrack_seen", case["id"], ROOT / "data/external/deepcrack" / case["image"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite evidence")
    args.output.mkdir(parents=True)
    baseline_path = ROOT / "docs/benchmarks/performance/baseline_pipeline.py"
    spec = importlib.util.spec_from_file_location("baseline_pipeline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    cv2.setNumThreads(1)
    rows = []
    for index, (source, sid, path) in enumerate(known_images()):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing known image: {source}/{sid}")
        # PCS checks original only; small external/bundled sources also check stresses.
        conditions = ("original",) if source == "pcs_train_seen" else TRANSFORMS
        for condition in conditions:
            changed, _ = transform(image, np.zeros(image.shape[:2], np.uint8), condition)
            old = baseline.pipeline(changed, baseline.P, steps=True)
            new = optimized.pipeline(changed, optimized.P, steps=True)
            for key in old:
                if not np.array_equal(old[key], new[key]):
                    raise AssertionError(f"Intermediate changed: {source}/{sid}/{condition}/{key}")
            old_mask = baseline.to_orig(old["Z"], changed.shape)
            new_mask = optimized.to_orig(new["Z"], changed.shape)
            if not np.array_equal(old_mask, new_mask):
                raise AssertionError(f"Native mask changed: {source}/{sid}/{condition}")
            rows.append(dict(source=source, id=sid, condition=condition,
                             input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             mask_sha256=hashlib.sha256(np.packbits(new_mask).tobytes()).hexdigest()))
        if (index + 1) % 100 == 0:
            print(f"Parity: {index + 1} images / {len(rows)} conditions", flush=True)
    write_csv(args.output / "per_case.csv", rows)
    summary = dict(images=len({(r["source"], r["id"]) for r in rows}), conditions=len(rows),
                   all_intermediates_and_native_masks_equal=True, gt_loaded=False,
                   final_158_opened=False, cv_threads=cv2.getNumThreads(), opencv=cv2.__version__,
                   source_sha256={p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                                  for p in (baseline_path, Path(optimized.__file__), ROOT / "code/fast_median.py")})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
