# PCS source acquisition

Source: https://zenodo.org/records/20040434, original 2,209-image PCS dataset.
Published ZIP size: 1,406,283,443 bytes; published MD5:
`4b530eba05708a195355a0064e520db4`. The whole ZIP was not downloaded, so that
published MD5 is provenance rather than a checksum claimed to have been verified.

`code/fetch_pcs.py` reads the remote ZIP end record and central directory using
validated HTTP 206 byte ranges. It allows only paired image/mask IDs in the chosen
split, retrieves adjacent ZIP members in batches of at most about 8 MiB, checks
decompressed member size and ZIP CRC, and records SHA-256 of each original file.
Existing files are reused only if their size and CRC match. Temporary files are
replaced atomically after verification. Destinations are confined to the dataset
directory. Individual 429 responses honor Retry-After/rate-reset before retrying.

The first individual-member retrieval hit source rate limiting. It was stopped
and resumed from verified files after implementing coalesced ranges. This reduced
validation retrieval to 40 batches and test retrieval to 20 batches.

## Training selection

Frozen before viewing or decoding any image/GT: 150 IDs sampled by
`random.Random(42).sample` from sorted source train IDs; A/B/C counts are 7/48/95.
Selection and source member CRCs are in
`docs/benchmarks/new-method/train-manifest.json`.

All 150 train image/mask pairs are complete and verified. Original files are under
`data/Pavement_Crack_Segmentation_Dataset/train/{images,masks}`; recorded file hashes
are in `data/Pavement_Crack_Segmentation_Dataset/download-train-150.json`.

Validation/test downloads are retrieval-only. No image or mask has been decoded,
viewed, scored, selected by GT, or inspected for positive/negative content by the
data-acquisition task. The parent will control when they may be evaluated.

**All retrievals completed:** train 150 pairs (300 members), val 498 pairs (996
members), test 231 pairs (462 members). The full source ZIP central directory
confirms 1,480/498/231 paired IDs and zero ID overlap between every split. The
validation and test hash manifests are respectively
`data/Pavement_Crack_Segmentation_Dataset/download-val-all.json` and
`data/Pavement_Crack_Segmentation_Dataset/download-test-all.json`. The source
Readme was downloaded separately as `SOURCE-Readme.md`; it confirms split counts,
binary PNG masks, and manual image-by-image labeling. It adds no explicit license.

Commands:

```bash
python code/fetch_pcs.py --split train --limit 150 --seed 42 --workers 8
python code/fetch_pcs.py --split val --workers 6
python code/fetch_pcs.py --split test --workers 6
```

Compilation and Ruff E9/F checks passed. The downloader requires `requests`; data
and per-file download manifests stay in ignored `data/`. No raw dataset image is
added to Git. License text is not stated in the fetched record metadata; retain
the authors/source attribution and do not assume blanket redistribution rights.
