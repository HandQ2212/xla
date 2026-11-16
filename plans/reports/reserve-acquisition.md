# Prospective PCS reserve acquisition

Client date: 2026-11-16. Data-only task: no reserve photograph or GT is decoded,
viewed, labeled by content, scored, or used to choose an algorithm.

## Selection frozen before retrieval

- Original source train: 1,480 paired IDs, from the cached official ZIP central directory.
- Historical TUNE300 reconstructed with the notebook's exact NumPy default_rng(42)
  sequence: choose 30 A, then 90 B, then 180 C without replacement from sorted train IDs.
- Excluded new train150 IDs and historical TUNE300. Their overlap is 28 IDs;
  the exclusion union is 422 IDs and the eligible pool has **1,058 IDs**.
- Chose exactly **400 IDs** using `random.Random(20261116).sample(sorted(pool), 400)`
  and sorted the result. Group counts: **A 21 / B 108 / C 271**.
- No selected image or mask file preexisted locally when the reserve was frozen.
- Frozen manifest: `docs/benchmarks/new-method/reserve-manifest.json`. Its original
  Windows-byte SHA-256 at creation was
  `81d002e05becefe6746904fdb247b2a7825b644bf1ef81997609d2fb56ead503`.

Runnable reconstruction of the historical exclusion (the manifest's A/B/C symbols
refer to these string literals):

```python
rng = np.random.default_rng(42)
TUNE_IDS = sorted(sum([
    list(rng.choice([s for s in TRAIN_IDS if s[0] == g], n, replace=False))
    for g, n in [('A', 30), ('B', 90), ('C', 180)]
], []))
```

`TRAIN_IDS` above contains all 1,480 original train IDs in sorted order, not the
partial set of files currently present on disk. The original train150 manifest
has not been changed. Existing studies must continue using that fixed list;
the train folder now also contains this reserved data.

## Retrieval and provenance

The ignored helper `results/reserve-acquisition/download_reserve.py` reuses
`fetch_pcs.py`'s verified bounded HTTP ranges, ZIP decompression and CRC checks.
It coalesces the selected members into 310 range batches and spaces requests to
remain below the source's observed rate limit. Only selected originals are stored.

Files go to the original `data/Pavement_Crack_Segmentation_Dataset/train/{images,masks}`
layout. Per-file SHA-256 and ZIP CRC provenance is written on completion to
`data/Pavement_Crack_Segmentation_Dataset/download-reserve-400.json`. That record
also includes raw and normalized-LF hashes of the already frozen reserve manifest.

Source is the original https://zenodo.org/records/20040434 archive, with published
archive MD5 `4b530eba05708a195355a0064e520db4`. Partial retrieval verifies each member;
it does not claim verification of the full archive's MD5.

This reserve is ID-disjoint from both excluded training selections. Unknown
capture/session relationships still limit scene independence. It may be unlocked
for evaluation only after the parent fixes the next candidate and acceptance rule.

## Completion

All **800 ZIP members / 400 image-mask pairs** are complete, with size/CRC checks
and per-file SHA-256 recorded. The frozen manifest's raw hash remains unchanged
from selection time. Its IDs are disjoint from the original train150 and all 300
historical tuning IDs. No reserve pixels or GT content were decoded, viewed or
scored during acquisition. The reserve is now ready for the parent's prospective
evaluation after candidate/gate freeze.
