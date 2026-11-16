# Prospective ratio-normalized reserve acquisition

Client date: 2026-11-16. This is a data-only task. No newly selected photograph or
GT is decoded, viewed, scored, inspected for content, or used for algorithm choice.
The previous 400-image reserve is now observed development evidence after its
candidate failed the declared shadow stress; it is not reused as fresh confirmation.

## Frozen selection

Original source train has 1,480 paired IDs. Exclusions are historical TUNE300
(reconstructed with the exact notebook NumPy seed42 sequence), new train150, and
all previously used reserve400. Train150 and TUNE300 overlap in 28 IDs; reserve400
is disjoint from both. Therefore the exclusion union is **822 IDs** and the new
eligible pool contains **658 IDs**.

Selected exactly **500 IDs** with
`sorted(random.Random(20261117).sample(sorted(eligible_pool), 500))`.
The seed is 20261117; it is not a claim that acquisition occurred on that date.
Group counts: **A 21 / B 153 / C 326**. No selected image or mask file preexisted.
A metadata-only source-image ZIP-CRC/size comparison flags no match against the
excluded images; this does not establish visual or capture-session independence.

Frozen before retrieval: `docs/benchmarks/new-method/ratio-reserve-manifest.json`.
Original Windows-byte SHA-256:
`d0799fc27e434afa9b53ffc7b844490e83dcb5ef6b0bab791feb8e01e9b0c0d7`.
Normalized-LF SHA-256:
`6753fcb02add62239154eec1c3007b0832d057a8af9224266a93f52b0b9add29`.
Neither prior manifest was changed.

## Retrieval

The ignored helper `results/ratio-reserve-acquisition/download_ratio_reserve.py`
reuses `fetch_pcs.py`'s validated HTTP ranges, ZIP CRC and size checks, existing-file
verification and atomic writes. Source is https://zenodo.org/records/20040434,
published archive MD5 `4b530eba05708a195355a0064e520db4`. The whole ZIP MD5 is not
claimed independently verified by partial member retrieval.

The 1,000 image/mask members coalesce into 331 range batches. Starts are spaced to
respect the publisher rate limit. Original files go to
`data/Pavement_Crack_Segmentation_Dataset/train/{images,masks}`. Completion records
per-file SHA-256/CRC and the frozen selection hash at
`data/download-ratio-reserve500.json`.

The new reserve may be evaluated only after the parent freezes the ratio-normalized
candidate and prospective acceptance rule. Positivity/negative counts remain unknown
to this acquisition task. Unknown sessions and adjacent scenes remain limitations.

## Completion

All **1,000 members / 500 image-mask pairs** completed CRC/size verification and
per-file SHA-256 recording. The frozen manifest hashes remain unchanged. IDs are
disjoint from train150, historical TUNE300 and the observed reserve400. No selected
image or GT was decoded, viewed or scored in this task. Data are ready for the
already frozen ratio candidate's first inference.

The physical train folder now contains 1,050 selected paired IDs (150+400+500), not
the full 1,480 original train set. Studies must keep using their fixed manifests;
do not treat a directory glob as the original source's full train-ID list.
