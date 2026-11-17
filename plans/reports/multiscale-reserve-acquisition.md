# Final unused PCS reserve acquisition

Client date: 2026-11-16. Data-only task: no selected image or GT was decoded,
viewed, scored, or used for algorithm selection. Positivity counts are unknown
to this task. Previous reserve400 and ratio-reserve500 are already observed
development evidence and are not reused as fresh confirmation.

## Selection before retrieval

Starting from all 1,480 original train IDs in source ZIP metadata, excluded the
historical TUNE300, original new train150, observed reserve400, and observed
ratio-reserve500. The exclusion union is **1,322 IDs**, leaving exactly **158**.
Selected **all remaining IDs in sorted order**, without sampling or GT selection.
Group counts are **A 5 / B 49 / C 104**. No selected source file preexisted locally;
a ZIP-CRC/size screen finds no selected source-image match against excluded images.
That screen does not establish visual/capture-session independence.

Frozen manifest: `docs/benchmarks/new-method/multiscale-reserve-manifest.json`.
Raw Windows-byte SHA-256 at selection and completion:
`2901995298598a074df2d5c5052880b743b286dfbac06172705287dbcc3a76df`.
Normalized-LF SHA-256:
`096e41416c27f71cec7d0dee3ee9a4ff1a40cf2d7457ce3387515736b9b072ed`.
Prior manifests and model code were not changed.

## Completed retrieval

All **316 original files / 158 image-mask pairs** completed bounded HTTP Range,
uncompressed-size and ZIP-CRC verification. Per-file SHA-256 and selection hashes
are recorded in `data/download-multiscale-reserve158.json`. Source remains
https://zenodo.org/records/20040434, published ZIP MD5
`4b530eba05708a195355a0064e520db4`; partial member retrieval does not independently
verify the entire archive's MD5.

The ignored reproducible helper is
`results/multiscale-reserve-acquisition/download_multiscale_reserve.py`, reusing
`code/fetch_pcs.py`. It used 163 coalesced range batches with request spacing.
No raw image or mask was added to Git.

This exhausts the remaining ID-virgin original PCS training pool under the stated
exclusions. Capture/session independence is still unknown. The physical train
folder contains 1,208 acquired paired IDs; the 272 missing IDs are historical
TUNE300-only cases, not an unused holdout. Studies must retain fixed manifests.
Evaluation can begin only after the parent freezes the multiscale candidate and
prospective gate. If it fails, these 158 cases also become observed evidence;
they cannot be retuned against and called fresh confirmation again.
