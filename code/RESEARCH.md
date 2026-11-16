# Research-only modules

`pipeline.py` is the production classical pipeline. The app imports it directly;
none of the methods below is selected in the app or enabled by default.

These are rejected hypotheses retained for reproducibility, **not improvements
approved for deployment**: `experiments.py`, `line_response.py`,
`mask_refinement.py`, `local_contrast.py`, `local_fusion.py`, `ratio_local.py`,
`multiscale.py`, `mask_restore.py`. The matching `*_study.py` scripts are
development sweeps. `confirm_local.py` replays the frozen local methods using
source snapshots and manifests in `docs/benchmarks/new-method`.

See [quality results and limitations](../docs/pipeline-improvement.md) and
[external false cracks](../docs/external-stress-report.md). Failure cases and
failed criteria are retained. Re-running a seen dataset does not create a new
held-out evaluation. The 158 IDs in `multiscale-reserve-manifest.json` remain
unopened because no candidate passed its training gate.

`fetch_pcs.py` downloads the selected original source bytes with HTTP Range and
ZIP CRC/size checks. Raw images and labels stay in ignored `data/`; no new PCS or
DeepCrack photographs are redistributed here.

`benchmark_performance.py` and `verify_performance.py` evaluate only exact-output
engineering changes against the immutable baseline snapshot. They do not measure
an accuracy gain. See [performance report](../docs/performance-report.md).
