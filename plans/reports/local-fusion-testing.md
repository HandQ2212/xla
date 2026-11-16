# Local-statistics fusion and ranked-area mathematical tests

Date: 2026-11-15. Scope: `local_contrast.py` and `local_fusion.py` research modules.
Only synthetic numerical arrays were used. No source train/validation/test images,
GT files or evaluation outputs were read. No implementation file was edited.

- **10/10 focused unittest tests passed**, final run 0.374 s.
- Ruff `--select E9,F` and compileall passed for both modules and the focused tests.

## Verified properties

- Constant black/gray/bright images have the expected normalized mean, zero deviation,
  finite threshold and empty dark-region mask. On nonconstant texture, a complete
  interior window's mean/std matches an independent float64 mean/std calculation.
  All deviations are finite and nonnegative.
- A straight monotonic dark/bright step is not selected with the fixed 1.5-deviation
  rule; a narrow dark valley is selected while its flat flanks are not.
- Connected fusion grows only the 8-connected local component touched by the coarse
  mask. Supported fusion additionally retains coarse pixels within the exact 3×3
  ellipse neighborhood; diagonal-only and distant pixels are excluded. Supported
  blending can retain a nearby coarse pixel without an intersecting reconstruction
  seed, as its declared union formula permits. An entirely empty coarse mask returns
  empty for all blend modes.
- Ranked budgeting returns exactly min(candidate count, budget) pixels and never
  introduces a pixel outside the candidate. High scores outside the candidate are
  ignored; equal-score ties follow candidate raster order. Zero budgets and empty
  candidates return empty masks, and a candidate already within budget is unchanged.
- Input photographs, coarse/support masks and ranking confidence arrays remain
  unchanged. Full synthetic predictions have original boolean shape and area no
  greater than legacy P on odd/tiny/extreme inputs, including a nonempty baseline
  fixture so the area property is not checked only on empty cases.

No mathematical implementation failure was found for the declared configurations.
These invariants do not demonstrate higher Dice or semantic rejection of repaired
seams. A ranked mask may relocate pixels within its candidate and fragment regions;
area≤legacy does not mean subset-of-legacy. No quality-gate conclusion is made.

## Conservative detection-preserving variant: 2026-11-16

The earlier ranked candidate failed its official-test quality gate and is not admitted
under that protocol. The following checks cover the new conservative fallback only;
they do not reinterpret the earlier failure as a pass.

- **13/13 local-fusion tests passed**, 0.832 s. All **31/31 synthetic research tests**
  (line response, native refinement and local fusion) passed together in 1.219 s.
  Focused Ruff `--select E9,F` and compileall passed.
- Actual predictions with preservation enabled retain exactly the legacy empty/nonempty
  image decision and never exceed its foreground count, across odd/tiny/extreme inputs
  and retained fractions 0, .25 and .5.
- A genuine window=1 local-statistics calculation gives mean=self and empty local
  support on a synthetic dark line that legacy P detects. The default and explicit
  preservation-disabled routes stay empty; enabling preservation returns P pixel-for-pixel.
  No mocks or fabricated evaluation scores were used.
- A synthetic two-width line image yields a nonempty partial candidate with retained
  fraction between .25 and .5. Fractions 0/.25 keep that candidate unchanged; .5 returns
  P exactly. This exercises actual local masks, ranking and fallback together.

No fresh reserve/DeepCrack photographs, GT, result files or other confirmation data
were read. No implementation files were modified and no mathematical failure was found
for the declared inputs. Production integration and quality admission remain separate.
