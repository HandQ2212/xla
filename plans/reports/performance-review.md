# Final performance and frozen-replay review

## Result

No unresolved blocking findings after the two frozen-replay integrity fixes below.
The accepted production change is exact-output engineering, not improved crack
segmentation quality. Failed quality variants remain research-only.

## Findings resolved during review

1. `confirm_local.py` originally checked frozen source/configuration but did not
   enforce the selected reserve/DeepCrack manifest hash. `replay_guard.py` now
   checks the historical raw/LF manifest identity before decoding any pair, and the
   verified manifest object is passed into `pairs` rather than reread.
2. The old local freezes omitted imported `experiments.py` from their source maps.
   Replay now pins that reconstruction helper to the immutable d155 implementation
   with a clearly labeled supplemental audit. Historical freeze files were not
   rewritten or presented as having originally covered that dependency.

Both issues concerned replay integrity; the current saved files already matched
historical identities. No discrepancy in reported predictions was discovered.

## Production correctness and compatibility

- Reviewed in-place illumination arithmetic and Sobel buffer reuse: operations
  retain the original float32 order and do not mutate the input.
- Histogram fast path uses exact float32 integer counts only at image sizes up to
  2^24; other dimensions/dtypes and larger counts retain the old bincount path.
- Tiled median includes a k//2 halo and copies only tile interiors, preserving the
  same neighborhood and replicated outer borders. A local executor is closed on
  each call; normal OpenCV and process-worker routes avoid nested worker pools.
- Contour points determine the same bounding rectangle hull as region pixels;
  reviewed exactness tests include degenerate/rotated/holed/disconnected regions.
- App keeps production P, actual intermediate rendering, original output geometry
  and bounded decode caching. New image-processing candidates are not exposed.
- Research methods use classical filtering, local statistics, thresholding,
  morphology, connectivity, geometry and interpolation; no trained model was found.

## Independent artifact and replay checks

- Baseline snapshot is byte-identical after LF normalization to the recorded d155
  Git source, not a locally optimized substitute.
- Current production/fast-median hashes match both saved timing runs.
- Both timing CSVs have exactly 150 unique manifest IDs. Recomputed paired speedup
  formulas, medians, slower-image counts and worst slowdown match summary JSON.
- Default result is 1.082501x paired median (2/150 slower); separate OpenCV-one-thread
  result is 1.177202x (4/150 slower). Documentation distinguishes these runs and
  discloses the two internal tile workers when OpenCV itself is set to one thread.
- Verified all recorded frozen-large, frozen-optimized and frozen-ratio source
  hashes, then exercised new source/manifest guards for their actual snapshots.
  DeepCrack selection guard also passes. No photograph/GT decoding was involved.
- Reviewed wide-parity selection and its saved 1210-image/2010-condition summary:
  only explicitly observed IDs are selected; final 158 IDs are excluded before
  reading images. The stored source hashes match current production code.

## Validation and scientific limits

The preceding 79-test final suite, low-level parity and AppTest report were reviewed.
New replay mutation/newline regressions and the post-fix full suite are being run
separately by the tester. This review intentionally did not start CPU-heavy tests
or timing while the isolated measurement was active, and did not open any of the
158 final-reserve images or their GT.

Reports correctly retain failed quality gates, shadow regressions, negative/positive
separation, observed-versus-new confirmation distinctions and image-bootstrap
independence limits. Pixel parity preserves existing metrics for the same inputs;
it does not establish better accuracy or guarantee exactness on all future OpenCV
versions. The measured speedup is hardware/input dependent, and some images are
slower. No accuracy or universal-runtime claim is approved by this review.
