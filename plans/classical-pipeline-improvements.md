# Plan: improve classical crack segmentation on hieugm

Base: origin/mailinh (contains origin/develop). Keep legacy parameters and historical benchmark reproducible. Only classical image processing from chapters 2–5, 7–8; no learned models.

## Findings

- `D & E` cuts dark crack interiors away from their edge support. A seed-based connected-component reconstruction can keep the full connected dark region when it has edge evidence.
- An ellipse closing of 7 pixels may thicken masks / join background texture. Compare smaller closing or reconstruction before changing defaults.
- Rotated-box elongation can reject curved or branching cracks; chapter 8 image moments or geometric compactness are eligible alternatives, but require measured validation.
- Global 0.3% image-area rejection loses short cracks (README already reports 45/186 positive test images returned empty). Do not simply remove it: it also rejects false positives on negative images.
- `pipeline(..., steps=True)` currently ignores `post=False`, making the app ablation misleading.
- App downsizes original input to 1280 before segmentation and metrics, and locates samples relative to cwd; preserve original mask dimensions and resolve paths relative to app file.
- Notebook still duplicates module definitions despite module claiming a shared source. Avoid leaving notebook and app using different new algorithms.

## Local course evidence

- `../../tmp/slide-text/Chương 7-notebook.txt`, section 3.3: Xk=(Xk-1 ⊕ B) ∩ A, connected-component extraction; section 4: marker growth by constrained dilation.
- Same file section 5: black-hat = close(image) - image for dark thin structures; optional comparison only, since PIPELINE.md records earlier weak results for closing-based backgrounds.
- `../../tmp/slide-text/5_XLA5_Phat hien bien.txt`: dual threshold / hysteresis.
- `../../tmp/slide-text/8_XLA8_NhanDangAnh.txt`, PDF page 16: area, perimeter, aspect ratio, compactness and moments.

## TODO

- [x] Record existing P performance on all bundled CFD / PCS image-GT pairs at original resolution.
- [x] Compare a small, explicitly exploratory set of curriculum-supported variants: edge-seeded reconstruction, shape filter alternative, smaller closing. Record per-image Dice/IoU, precision/recall, positive and negative cases, runtime.
- [x] Implement the useful variant as an optional experimental preset and preserve legacy P; do not promote parameters from these test samples as validated defaults.
- [x] Fix steps/direct API parity and app ablation, preserve original output dimensions, use cwd-independent paths; update notebook imports / experimental section as necessary.
- [x] Add focused regression tests for seed reconstruction (connected weak pixels recovered, disconnected components excluded), post bypass parity, empty/tiny/constant input and original-size boolean output.
- [x] Provide a deterministic benchmark command producing per-image CSV and summary; explain missing dataset and distinction between exploratory examples and held-out evaluation.
- [x] Complete independent review, compile/lint, tests, notebook static validation and documentation; ready for focused conventional commit and authorized push to origin hieugm.

## Evaluation limits

Only 10 CFD and 10 PCS bundled test samples are currently available. They are not a validation split. Report any gains as exploratory on bundled samples; they do not replace README's historical full-test results. Preserve failed cases and avoid claiming an improvement on the full 231-image test set. If the full train/val dataset becomes available, select parameters on train/val first, freeze them, then run test once.

## Outcome

- P remains default and is pixel-identical to mailinh for all 20 images, including intermediate masks.
- P_THIN increases PCS sample Dice 0.323799 → 0.335081; P_RECON gives 0.325740. Both regress on CFD samples. No held-out quality claim.
- Five fixed comparisons, input hashes and per-image metrics are saved under docs/benchmarks/samples/.
- 13 regression and AppTest checks pass. Full dataset evaluation is still pending availability of train/val.
