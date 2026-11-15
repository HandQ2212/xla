# Verification: classical pipeline changes

Date: 2026-11-15. Environment: Python 3.12.9, OpenCV 5.0.0, NumPy 2.3.4,
Streamlit 1.56.0, Windows. No network calls or dataset downloads.

- **13 unittest tests passed**, final run in 3.922 s.
- `python -m compileall -q tests`: passed.
- `python -m ruff check tests --select E9,F63,F7,F82`: passed.
- Manual baseline comparison loaded `origin/mailinh:code/pipeline.py` from local Git,
  without fetching. All **20/20 original-size P masks** match pixel-for-pixel;
  intermediate M, C, A, F and Z also match for all 20 images.

## Coverage

- 8-neighbor reconstruction equals repeated constrained dilation on 12 deterministic
  masks; diagonal paths grow, disconnected components and outside-support markers do
  not grow; absent markers and empty support return empty masks.
- Combined area/elongation filtering matches separate filtering across 24 combinations
  of axis/rotated geometry and area/elongation thresholds.
- Direct and step-mode results agree for P, P_RECON and P_THIN. `post=False` truly
  bypasses every postprocessing stage and changes the real image result.
- Constant black, gray and white inputs at 1×1, odd, and extreme aspect-ratio sizes
  remain empty with boolean original-size output. Display resizing keeps a nonzero
  short dimension. Every notebook code cell compiles after shared imports.
- Streamlit AppTest starts from repository root and selects bundled B_00038.jpg.
  Original caption remains **1702×866** and GT area **12022**; Dice equals pipeline P
  (**0.806**). Disabling post changes Dice to **0.595**; both candidate selectors agree
  with direct pipeline scores. An uploaded image with a sample-like name receives no GT.
- Benchmark refuses missing validation data, missing GT and mismatched image/GT sizes;
  it never silently falls back to test or resizes GT to hide a mismatch.

## Limits

Full train/validation/test dataset is absent; these checks establish correctness and
legacy preservation, not held-out quality gains. AppTest validates UI execution and
metrics without manual browser layout review. Existing Streamlit width deprecation
warnings and the expected OpenCV warning for the deliberately missing GT do not cause
failures. Permanent tests require no Git checkout contents beyond committed fixtures,
no network and no external dataset.

Unresolved questions: none for these regression checks.
