"""Download pinned PaveSeg public preview and freeze an image-grouped stress manifest."""
import argparse
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = "FuturePave/PaveSeg-Dataset"
COMMIT = "dc21495f61867c94336283636959c754d192e18a"
LANDING = f"https://github.com/{REPO}"
RAW = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
DATA = ROOT / "data" / "external" / "paveseg"
MANIFEST = ROOT / "docs" / "benchmarks" / "external" / "manifest.json"
LABELS = {0: "background", 30: "crack", 60: "pothole", 90: "sealed",
          120: "patch", 150: "alligator", 180: "cover", 210: "joint"}


def download(destination=DATA):
    url = f"https://codeload.github.com/{REPO}/zip/{COMMIT}"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        for member in bundle.infolist():
            parts = Path(member.filename).parts[1:]
            # Only these exact expected files; never extract arbitrary archive paths.
            if len(parts) == 2 and parts[0] in {"img_preview", "label_preview"}:
                if parts[1] not in {f"{i:06d}.jpg" for i in range(1, 101)} | {
                        f"{i:06d}_mask.png" for i in range(1, 101)}:
                    continue
            elif parts not in [("README.md",), ("LICENSE",)]:
                continue
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(member))


def read_case(case, data=DATA):
    image_path = data / "img_preview" / (case["id"] + ".jpg")
    label_path = data / "label_preview" / (case["id"] + "_mask.png")
    for path, key in [(image_path, "image_sha256"), (label_path, "label_sha256")]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != case[key]:
            raise ValueError(f"Source hash mismatch: {path.name}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if image is None or label is None or image.shape[:2] != label.shape:
        raise ValueError(f"Invalid image/label pair: {case['id']}")
    return image, label


def make_manifest(data=DATA):
    groups = np.arange(10)
    np.random.default_rng(42).shuffle(groups)
    discovery = set(groups[:5].tolist())
    rows, hashes, thumbnails = [], set(), []
    for i in range(1, 101):
        sid = f"{i:06d}"
        ip = data / "img_preview" / (sid + ".jpg")
        mp = data / "label_preview" / (sid + "_mask.png")
        ih = hashlib.sha256(ip.read_bytes()).hexdigest()
        if ih in hashes:
            raise ValueError(f"Duplicate original image: {sid}")
        hashes.add(ih)
        image, label = cv2.imread(str(ip)), cv2.imread(str(mp), 0)
        if image is None or label is None or image.shape[:2] != label.shape:
            raise ValueError(f"Invalid dimensions: {sid}")
        values = [int(x) for x in np.unique(label)]
        if not set(values) <= set(LABELS):
            raise ValueError(f"Unexpected class encoding: {sid}: {values}")
        block = (i - 1) // 10
        rows.append(dict(id=sid, source="paveseg-preview", group=f"id-block-{block:02d}",
                         split="discovery" if block in discovery else "confirmation",
                         image_url=f"{RAW}/img_preview/{sid}.jpg",
                         label_url=f"{RAW}/label_preview/{sid}_mask.png",
                         image_sha256=ih, label_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
                         height=image.shape[0], width=image.shape[1],
                         positive=bool(np.isin(label, [30, 150]).any()),
                         tags=[LABELS[x] for x in values if x]))
        thumbnails.append(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (64, 54)))
    # Conservative near-duplicate screen across the split; review any flagged pairs.
    near = []
    for i in range(100):
        for j in range(i):
            error = float(np.mean(np.abs(thumbnails[i].astype(float) - thumbnails[j])))
            if rows[i]["split"] != rows[j]["split"] and error < 5:
                near.append([rows[i]["id"], rows[j]["id"], error])
    return dict(source=LANDING, commit=COMMIT, license="Apache-2.0 (repository LICENSE)",
                positive_labels=[30, 150], negative_labels=[0, 60, 90, 120, 180, 210],
                split_seed=42, group_rule="Ten consecutive IDs per block, five blocks per split",
                limitation="One source; capture sessions unknown. Blocks do not establish independent scenes.",
                cross_split_near_duplicate_screen=near, cases=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=MANIFEST)
    args = parser.parse_args()
    if args.download:
        download()
    manifest = make_manifest()
    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Existing frozen manifest differs; refusing to overwrite it")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(manifest['cases'])} pairs; near-duplicate flags: "
          f"{len(manifest['cross_split_near_duplicate_screen'])}")


if __name__ == "__main__":
    main()
