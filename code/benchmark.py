"""So sánh cố định các ứng viên cổ điển; không chọn tham số từ tập test."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline as pl
import experiments as exp

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "legacy": pl.P,
    "reconstruct": exp.P_RECON,
    "thin": exp.P_THIN,
    "close3": {**pl.P, "close": 3},
    "reconstruct_close3": {**exp.P_RECON, "close": 3},
}


def pairs(data_root=None, split="val"):
    """Ảnh mẫu hoặc đúng một split của PCS; không tự chuyển sang test khi thiếu val."""
    if data_root:
        folder = Path(data_root) / split
        candidates = [(split, p, folder / "masks" / (p.stem + ".png"))
                      for p in sorted((folder / "images").glob("*.jpg"))]
    else:
        candidates = [(group, p, p.with_name(p.stem + "_gt.png"))
                      for group in ("samples", "samples_pcs")
                      for p in sorted((ROOT / group).glob("*.jpg")) if not p.stem.endswith("_pred")]
    if not candidates:
        raise ValueError("Không tìm thấy ảnh. Kiểm tra data-root/split; không có fallback sang tập khác.")
    for group, path, mask_path in candidates:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        gt_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or gt_raw is None:
            raise ValueError(f"Không đọc được cặp ảnh/GT: {group}/{path.name}")
        if image.shape[:2] != gt_raw.shape:
            raise ValueError(f"Kích thước ảnh và GT khác nhau: {group}/{path.name}")
        digest = hashlib.sha256(path.read_bytes() + mask_path.read_bytes()).hexdigest()
        yield group, path.stem, image, gt_raw > 127, digest


def summarize(rows):
    output = []
    for group in sorted({r["dataset"] for r in rows}):
        for variant in VARIANTS:
            subset = [r for r in rows if r["dataset"] == group and r["variant"] == variant]
            positive = [r for r in subset if r["gt_pos"]]
            negative = [r for r in subset if not r["gt_pos"]]
            tp, fp, fn = (sum(r[k] for r in subset) for k in ("tp", "fp", "fn"))
            output.append(dict(
                dataset=group, variant=variant, images=len(subset),
                dice=float(np.mean([r["dice"] for r in subset])),
                iou=float(np.mean([r["iou"] for r in subset])),
                positive_dice=float(np.mean([r["dice"] for r in positive])) if positive else None,
                precision=float(np.mean([r["precision"] for r in positive])) if positive else None,
                recall=float(np.mean([r["recall"] for r in positive])) if positive else None,
                pooled_dice=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
                pooled_iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                negative_images=len(negative), negative_correct=sum(not r["pred_pos"] for r in negative),
                positive_empty=sum(not r["pred_pos"] for r in positive),
                mean_ms=float(np.mean([r["elapsed_ms"] for r in subset])),
            ))
    return output


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, help="PCS chứa train/val/test, mỗi split có images và masks")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "benchmark")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rows, manifest = [], []
    for dataset, sid, image, gt, digest in pairs(args.data_root, args.split):
        manifest.append(dict(dataset=dataset, id=sid, sha256_pair=digest))
        for name, params in VARIANTS.items():
            started = time.perf_counter()
            prediction = exp.pipeline(image, params)
            elapsed = (time.perf_counter() - started) * 1000
            rows.append(dict(dataset=dataset, id=sid, variant=name,
                             **pl.scores(prediction, gt), elapsed_ms=elapsed))
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "per_image.csv", rows)
    write_csv(args.output / "summary.csv", summary)
    metadata = dict(
        scope=args.split if args.data_root else "bundled_test_examples_exploratory",
        warning="Bundled samples have already been inspected; they are not a held-out validation set.",
        python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
        platform=platform.system(), threads=1, variants=VARIANTS, inputs=manifest,
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for row in summary:
        print(f"{row['dataset']:12} {row['variant']:20} Dice={row['dice']:.4f} "
              f"IoU={row['iou']:.4f} {row['mean_ms']:.1f}ms")


if __name__ == "__main__":
    main()
