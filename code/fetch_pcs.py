"""Retrieve selected PCS ZIP members with verified HTTP ranges and ZIP CRCs."""
import argparse
import concurrent.futures
import hashlib
import json
import random
import struct
import threading
import time
import zlib
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/api/records/20040434/files/Pavement_Crack_Segmentation_Dataset.zip/content"
SIZE = 1406283443
DEST = ROOT / "data" / "Pavement_Crack_Segmentation_Dataset"
SESSIONS = threading.local()


def byte_range(start, end):
    if not hasattr(SESSIONS, "client"):
        SESSIONS.client = requests.Session()
    for attempt in range(4):
        try:
            response = SESSIONS.client.get(URL + f"?range={start}-{end}",
                                    headers={"Range": f"bytes={start}-{end}"}, timeout=(20, 90))
            if response.status_code == 429:
                pause = max(float(response.headers.get("Retry-After", 1)),
                            float(response.headers.get("X-RateLimit-Reset", 0)) - time.time() + 1)
                time.sleep(min(max(pause, 2), 60))
            response.raise_for_status()
            expected = f"bytes {start}-{end}/{SIZE}"
            if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError(f"Unexpected range response: {response.status_code}, {response.headers.get('Content-Range')}")
            if len(response.content) != end - start + 1:
                raise ValueError("Truncated HTTP range")
            return response.content
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def directory():
    cache = ROOT / "data" / "external" / "pcs-directory.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    tail = byte_range(SIZE - 65536, SIZE - 1)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise ValueError("ZIP end record missing")
    record = struct.unpack_from("<4s4H2LH", tail, offset)
    if record[1] or record[2] or record[3] != record[4]:
        raise ValueError("Multi-disk ZIP is unsupported")
    content = byte_range(record[6], record[6] + record[5] - 1)
    items, pos = [], 0
    while pos < len(content):
        entry = struct.unpack_from("<4s6H3L5H2L", content, pos)
        if entry[0] != b"PK\x01\x02":
            raise ValueError("Invalid central-directory entry")
        name = content[pos + 46:pos + 46 + entry[10]].decode("utf-8" if entry[3] & 2048 else "cp437")
        items.append(dict(name=name, method=entry[4], crc=entry[7], compressed=entry[8],
                          size=entry[9], offset=entry[16]))
        pos += 46 + entry[10] + entry[11] + entry[12]
    if len(items) != record[4]:
        raise ValueError("ZIP entry count mismatch")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return items


def selection(items, split, limit, seed):
    folders = {"images": {}, "masks": {}}
    for entry in items:
        parts = Path(entry["name"]).parts
        if len(parts) >= 3 and parts[-3] == split and parts[-2] in folders and not entry["name"].endswith("/"):
            folders[parts[-2]][Path(parts[-1]).stem] = entry
    ids = sorted(folders["images"])
    if not ids or set(ids) != set(folders["masks"]):
        raise ValueError(f"Missing or unpaired {split} images/masks")
    if limit is not None:
        if not 1 <= limit <= len(ids):
            raise ValueError("limit is outside available image count")
        ids = sorted(random.Random(seed).sample(ids, limit))
    return [(sid, folder, folders[folder][sid]) for sid in ids for folder in folders]


def fetch(task, split, source=None):
    sid, folder, item = task
    target = DEST / split / folder / Path(item["name"]).name
    if DEST.resolve() not in target.resolve().parents:
        raise ValueError("Destination escapes dataset directory")
    payload = target.read_bytes() if target.exists() else b""
    if len(payload) != item["size"] or zlib.crc32(payload) != item["crc"]:
        # ZIP extra fields may be 65535 bytes; this one request includes a complete header.
        end = min(SIZE - 1, item["offset"] + 30 + len(item["name"].encode("utf-8")) + 65535 + item["compressed"] - 1)
        block = (source[1][item["offset"] - source[0]:] if source else byte_range(item["offset"], end))
        local = struct.unpack_from("<4s5H3L2H", block)
        if local[0] != b"PK\x03\x04" or local[3] != item["method"] or local[2] & 1:
            raise ValueError("Unsupported or inconsistent local ZIP header")
        begin = 30 + local[9] + local[10]
        compressed = block[begin:begin + item["compressed"]]
        if item["method"] == 8:
            payload = zlib.decompress(compressed, -15)
        elif item["method"] == 0:
            payload = compressed
        else:
            raise ValueError(f"Unsupported compression: {item['method']}")
        if len(payload) != item["size"] or zlib.crc32(payload) != item["crc"]:
            raise ValueError(f"CRC/size mismatch for {item['name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return dict(id=sid, folder=folder, source_member=item["name"], bytes=len(payload),
                crc32=item["crc"], sha256=hashlib.sha256(payload).hexdigest())


def batches(tasks):
    groups = []
    for task in sorted(tasks, key=lambda t: t[2]["offset"]):
        item = task[2]
        end = min(SIZE - 1, item["offset"] + 30 + len(item["name"].encode("utf-8")) + 65535 + item["compressed"] - 1)
        if not groups or item["offset"] > groups[-1][1] + 65536 or end - groups[-1][0] > 8 * 1024 * 1024:
            groups.append([item["offset"], end, [task]])
        else:
            groups[-1][1] = max(groups[-1][1], end)
            groups[-1][2].append(task)
    return groups


def fetch_batch(group, split):
    start, end, tasks = group
    missing = any(not (DEST / split / t[1] / Path(t[2]["name"]).name).exists() for t in tasks)
    source = (start, byte_range(start, end)) if missing else None
    return [fetch(task, split, source) for task in tasks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ids-manifest", type=Path, help="Frozen JSON containing explicit ids")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.ids_manifest and args.limit is not None:
        parser.error("Use either ids-manifest or limit")
    tasks = selection(directory(), args.split, args.limit, args.seed)
    if args.ids_manifest:
        ids = json.loads(args.ids_manifest.read_text(encoding="utf-8"))["ids"]
        available = {task[0] for task in tasks}
        if len(ids) != len(set(ids)) or not set(ids) <= available:
            parser.error("Manifest IDs must be unique and belong to the selected split")
        tasks = [task for task in tasks if task[0] in set(ids)]
    print(f"Retrieving {len(tasks) // 2} {args.split} pairs; no image/GT decoding or scoring", flush=True)
    rows = []
    groups = batches(tasks)
    print(f"Coalesced into {len(groups)} HTTP range batches", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_batch, group, args.split) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            previous = len(rows)
            rows.extend(future.result())
            if len(rows) // 20 > previous // 20 or len(rows) == len(tasks):
                print(f"Verified {len(rows)}/{len(tasks)} ZIP members", flush=True)
    manifest = dict(source="https://zenodo.org/records/20040434", url=URL, archive_size=SIZE,
                    published_archive_md5="4b530eba05708a195355a0064e520db4", split=args.split,
                    seed=args.seed, selection="sorted IDs; random.Random(seed).sample when limited",
                    limit=args.limit, files=sorted(rows, key=lambda r: (r["id"], r["folder"])))
    selection_tag = args.ids_manifest.stem if args.ids_manifest else args.limit or "all"
    manifest["ids_manifest"] = str(args.ids_manifest) if args.ids_manifest else None
    path = DEST / f"download-{args.split}-{selection_tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed: {path}", flush=True)


if __name__ == "__main__":
    main()
