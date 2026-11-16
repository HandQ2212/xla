"""Compare unchanged P masks and actual full-pipeline latency against its snapshot."""
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

import pipeline as optimized

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/benchmarks/new-method/train-manifest.json")
    parser.add_argument("--baseline", type=Path, default=ROOT / "docs/benchmarks/performance/baseline_pipeline.py")
    parser.add_argument("--threads", type=int, help="Omit to retain OpenCV's installed default")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or (args.threads is not None and args.threads < 1):
        parser.error("repeats and threads must be positive")
    if args.output.exists():
        raise ValueError("Refusing to overwrite benchmark evidence")
    args.output.mkdir(parents=True)
    if args.threads is not None:
        cv2.setNumThreads(args.threads)
    spec = importlib.util.spec_from_file_location("baseline_pipeline", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    ids = json.loads(args.manifest.read_text(encoding="utf-8"))["ids"]
    rows, inputs = [], []
    for index, sid in enumerate(ids):
        path = ROOT / "data/Pavement_Crack_Segmentation_Dataset/train/images" / (sid + ".jpg")
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Missing image: {sid}")
        inputs.append(dict(id=sid, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        old = baseline.pipeline(image, baseline.P, steps=True)
        new = optimized.pipeline(image, optimized.P, steps=True)
        for key in old:
            equal = np.array_equal(old[key], new[key]) if isinstance(old[key], np.ndarray) else old[key] == new[key]
            if not equal:
                raise AssertionError(f"Intermediate changed: {sid}/{key}")
        old_mask = baseline.pipeline(image, baseline.P)
        new_mask = optimized.pipeline(image, optimized.P)
        if not np.array_equal(old_mask, new_mask):
            raise AssertionError(f"Native mask changed: {sid}")
        timings = {"baseline": [], "optimized": []}
        methods = [("baseline", baseline), ("optimized", optimized)]
        for repetition in range(args.repeats):
            for name, module in methods if (index + repetition) % 2 == 0 else methods[::-1]:
                started = time.perf_counter()
                module.pipeline(image, module.P)
                timings[name].append((time.perf_counter() - started) * 1000)
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        rows.append(dict(id=sid, **medians, speedup=medians["baseline"] / medians["optimized"],
                         mask_sha256=hashlib.sha256(np.packbits(new_mask).tobytes()).hexdigest()))
        if (index + 1) % 25 == 0:
            print(f"Parity/timing {index + 1}/{len(ids)}", flush=True)
    write_csv(args.output / "per_image.csv", rows)
    summary = dict(images=len(rows), intermediate_and_native_masks_equal=True,
                   baseline_median_ms=float(np.median([r["baseline"] for r in rows])),
                   optimized_median_ms=float(np.median([r["optimized"] for r in rows])),
                   median_paired_speedup=float(np.median([r["speedup"] for r in rows])),
                   mean_paired_speedup=float(np.mean([r["speedup"] for r in rows])),
                   slower_images=sum(r["speedup"] < 1 for r in rows),
                   worst_slowdown=max(r["optimized"] / r["baseline"] for r in rows))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sources = [args.baseline, Path(optimized.__file__), ROOT / "code/fast_median.py"]
    metadata = dict(python=platform.python_version(), platform=platform.system(), cpu=platform.processor(),
                    opencv=cv2.__version__, numpy=np.__version__, cv_threads=cv2.getNumThreads(),
                    repeats=args.repeats, warmups_per_method=2, order="alternating per image and repeat",
                    scope="Fixed150training images; no GT loaded; timing excludes image I/O",
                    baseline_ref="d155f98464dbc9e3d30280efce904d8f6b71bf1e:code/pipeline.py",
                    sources={p.name: hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                             for p in sources}, inputs=inputs)
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
