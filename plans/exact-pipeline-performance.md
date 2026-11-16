# Exact-output pipeline performance

Quality variants in the local, multiscale and restoration protocols failed their
declared gates. Keep P parameters, thresholds and native mask restoration fixed.
Engineering scope is CPU/allocation reduction with exact stage and mask parity;
speed is not evidence of improved segmentation quality.

- [x] Snapshot d155f984 pipeline before optimization.
- [x] Reduce private float-array allocations for illumination ratio and Sobel.
- [x] Use exact-count uint8 histogram fast path with bounded float32 counts.
- [x] Use contour extrema for the same bounding rectangle; Python int labels.
- [x] Test halo-tiled median; enable only for large uint8 single-thread OpenCV in
  MainProcess, retaining OpenCV directly for other cases to avoid nested pools.
- [x] Cache a bounded number of decoded uploads in the demo.
- [x] Benchmark all 150 known training images, alternating baseline/candidate
  order; compare every intermediate and native mask before timing.
- [x] Verify observed PCS1050, PaveSeg100, DeepCrack40, bundled20; no GT needed.
- [x] Keep final158 unopened. Do not glob the physical PCS train folder.
- [x] Run regression tests, compile/lint, review and document evidence.

Wide check: all stages/native masks equal on1210images/2010conditions. Separate
default/single-OpenCV-thread150-image timings and hashes are saved under
docs/benchmarks/performance. Delivery target is the authorized branch hieugm.

Default OpenCV threading and one-thread mode are separate measurements. Preserve
per-image times, input/source hashes and baseline snapshot. Do not claim every
image becomes faster or extend measurements to unseen hardware.
