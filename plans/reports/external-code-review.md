# External validation code review

## Result

No remaining blocking correctness, product-isolation, methodology or reproducibility findings after the two CLI fixes below. Full dataset/source independence remains unproven and is correctly reported as a limitation, not a passed gate.

## Findings resolved during review

- Candidate-only `--variants` previously allowed a long evaluation before crashing because paired gates require `legacy`. Argument parsing now rejects candidate comparisons without baseline before evaluating data. Duplicate variants are also rejected.
- Confirmation runs without `--output` previously inherited the discovery output directory. The default now resolves to `results/external-<split>`, preventing accidental discovery-table replacement for the normal split-only invocation.

## Independent checks

- Reviewed production cleanup, isolated research wrapper, downloader allowlist/path handling, manifest/source hash checks, transforms, positive/negative summaries, paired gates, gallery generation, tests, report and reproduction commands.
- Recomputed every saved discovery summary and all 12 gate verdicts from 900 per-image rows; recomputed confirmation summary and empty candidate gate from 300 rows. Saved values agree.
- Verified both saved metadata manifest hashes against the frozen manifest and exact denominators: 50 originals x 6 transforms x 3 methods for discovery; 50 x 6 x 1 for confirmation.
- Local pinned source README confirms expert pixel annotations and the exact grayscale class mapping used by evaluation.
- Checked production module exports and app source: rejected candidate names/reconstruction and experimental selector are absent. Research benchmarks import experiments.py explicitly.
- Inspected the saved gallery visually: label colors, failure examples and displayed metrics are consistent with the report.
- Ruff E9/F and compileall pass on final source during review.

## Interpretation limits

The report explicitly discloses one external source, unknown capture independence, coarse alligator-region GT, derived transforms not independent photographs, and baseline-only confirmation after candidate rejection. No reliable-improvement claim or bootstrap success is asserted. The gamma125 empty-positive case is accurately described as a changed image-level decision where previous Dice was already zero. Historical timing/core hashes remain attributed to the pre-cleanup implementation with separate pixel-equivalence verification.
