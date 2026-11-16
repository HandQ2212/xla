# Final performance-change correctness checks

Date: 2026-11-16. Environment: Python 3.12.9, OpenCV 5.0.0, NumPy 2.3.4,
Streamlit 1.56.0, Windows. Only synthetic arrays and the 20 already bundled source
photographs were used for new parity checks. No photograph or GT from the 158-image
multiscale reserve was opened. No network request or implementation edit was made.

## Results

- **86/86 tests passed**, final full suite in 18.378 s, including Streamlit AppTest,
  tiled-median tests, all existing research invariants, nine new parity tests and
  seven post-review replay-guard tests.
- Focused new parity suite: **9/9 passed**, 5.565 s.
- `python -m ruff check code tests --select E9,F`: passed.
- `python -m compileall -q code tests`: passed.
- No functional regression or parity discrepancy was found.

## Immutable reference and full-pipeline parity

The reference is `docs/benchmarks/performance/baseline_pipeline.py`, verified manually
against local Git `d155f98464dbc9e3d30280efce904d8f6b71bf1e:code/pipeline.py`. Its
LF-normalized SHA-256 is pinned in the permanent test:

`2ac8116ce3b5411e7722efef3b8d9ee3e4d2a4901e553f3e68d9ee55826224f1`

All **20/20 original-resolution P masks and every returned intermediate** match the
reference pixel-for-pixel. P parameters match as well. A synthetic crack/texture
fixture matches for P, P0 and nine supported ablations: subtraction, closing-based
background, Gaussian denoising, Otsu, axis rectangle, no postprocessing, no edge
restriction, no illumination compensation and unsmoothed histogram.

## Low-level exactness

- Median and closing-based illumination backgrounds match for windows 3/31/71 under
  both division and subtraction, including zero and saturated regions. Main-process
  single-thread OpenCV settings exercise the parallel tiled path on both portrait
  and landscape arrays larger than 200000 pixels. Inputs remain unchanged.
- Histogram fast path matches the reference on contiguous and noncontiguous uint8
  arrays. Fallbacks match for 1-D, multi-channel, uint16 and empty arrays. The count
  **2^24+1** is preserved exactly, confirming large arrays avoid float32 count loss.
- In-place Sobel and the complete edge-region tuple match the reference exactly for
  random, flat, tiny, extreme-aspect and noncontiguous arrays without input mutation.
- Contour-based minAreaRect matches pixel-based minAreaRect exactly for single and
  paired pixels, diagonal lines, rotated thin rectangles, holes, disconnected regions
  and sparse/dense random masks. Area/elongation decisions also match across thresholds
  and both rectangle modes. An additional manual 600-shape sweep found zero numerical
  elongation differences; it is supplementary, not a general mathematical proof.
- Existing tiled-median tests cover both axes, workers 1/2/4, windows 1/3/5/31/71,
  border impulses, ramps, tiny/odd dimensions, noncontiguous inputs and invalid inputs.

AppTest still verifies original image dimensions and GT metrics, toggles postprocessing
off and on through cached routes, checks no rejected-method selector is present and
does not assign sample GT to a sample-like upload name. The extracted step renderer
executes successfully in that integration test.

## Interpretation

These tests verify unchanged predictions and successful integration on the checked
inputs. They do not measure speedup: use the separate alternating-order full-pipeline
benchmark for latency evidence. No segmentation-quality improvement is inferred from
an exact-output optimization. Results are for the installed OpenCV version; new versions
should rerun the parity suite. The expected OpenCV warning from an intentionally missing
unit-test mask is non-failing. No unresolved regression failures remain.

## Post-review historical replay guards

Seven focused guard tests passed; the final 86-test suite above includes them.
They read source snapshots and metadata only, never source photographs or GT:

- Actual frozen-large, frozen-optimized and frozen-ratio implementations pass their
  historical records. Associated reserve/ratio-reserve/DeepCrack selection manifests
  pass, including LF and CRLF copies of sources and all three manifest types.
- Temporary changes to selected IDs, ID ordering or recorded member CRC fail in the
  actual replay `main` before any image decoder or output-directory creation.
- Changed implementation source fails; changed `experiments.py` also fails for an
  early freeze that omitted it from its original listed source hashes.
- Incompatible freeze/manifest-stage combinations fail verification.
- A synthetic same-size byte change fails CRC verification before decoding. Supplying
  the verified manifest works without rereading a deliberately nonexistent manifest
  path. Decoder traps are used solely to assert early rejection, not to fabricate
  predictions or evaluation scores.

Production implementation was unchanged by these replay-only fixes and tests. The
new guard checks do not alter prior prediction parity or timing evidence, and they
do not relabel historical replay as fresh confirmation.
