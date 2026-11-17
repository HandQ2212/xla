# External stress verification

Date: 2026-11-15. Python 3.12.9, OpenCV 5.0.0, NumPy 2.3.4,
Streamlit 1.56.0 on Windows. This audit made no network requests.

## Final checks

- **24/24 unittest tests passed**, final run 4.533 s (including post-review CLI checks).
- `python -m ruff check code tests --select E9,F`: passed.
- `python -m compileall -q code tests`: passed.
- Tests confirm production `pipeline.py` exposes no rejected P_RECON/P_THIN or
  reconstruction helper. Reconstruction regressions now exercise `experiments.py`.
- AppTest confirms no candidate selector, original-size B_00038 output/GT metrics,
  working postprocessing ablation and no assigned GT for sample-like upload names.

## Evidence equivalence after research separation

Manual audit used actual downloaded source image/label files validated by manifest
SHA-256, the same six declared transforms, and existing saved binary masks. It did
not rerun the timed benchmark or replace its timing measurements.

| Comparison | Result |
|---|---|
| Current production P vs local Git `b927189:code/pipeline.py`, all bundled samples | **20/20** final masks pixel-identical |
| Current research wrapper vs discovery saved evidence: 50 originals × 6 transforms × 3 variants | **900/900** masks pixel-identical |
| Current production P vs baseline-only confirmation saved evidence: 50 originals × 6 transforms | **300/300** masks pixel-identical |

The external equivalence audit completed in 56.96 s. No differences were found.
This verifies the research/production refactor preserves the evaluated predictions;
it does not establish candidate superiority.

## Additional regression coverage

- Photometric gamma, JPEG and shadow changes preserve exact source GT; the horizontal
  flip moves the source photograph and source GT together. Tests use existing bundled
  photographs/GT and do not create or infer external evaluation labels.
- Altered source bytes fail SHA-256 verification before image processing. Frozen
  manifest source URLs contain the pinned commit; IDs are unique and block groups
  do not cross discovery/confirmation splits.
- Positive Dice excludes negative-image Dice. Controlled numerical fixtures verify
  rejection for new negative false alarms even with a large positive gain, individual
  Dice regression hidden by a mean gain, and growth of an existing negative mask.
  These numerical fixtures are unit-test inputs, not evaluation cases or claimed results.
- Actual subprocess CLI tests reject both candidate-only variants and duplicate
  variants with exit code 2 and a clear parser error before reading even a deliberately
  missing manifest. CLI-only changes did not require repeating the pixel audit.

## Limits

The fixed external splits contain both positive and negative examples. The current
summary/gate functions assume both classes are present; arbitrary one-class manifests
are outside this verified scope. One source and unknown capture sessions still limit
independence, as documented in the frozen manifest. AppTest checks execution and
metrics without a manual browser layout review. Existing Streamlit deprecation logs
and the expected OpenCV log for a deliberately missing test mask are non-failing.

Unresolved regression failures: none.
