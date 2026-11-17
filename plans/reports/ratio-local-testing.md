# Illumination-ratio local refinement: synthetic tests

Date: 2026-11-16. Research-code correctness checks only. All inputs were synthetic
arrays; no fresh 500-reserve photographs, GT, manifests/results or other confirmation
pixels were read. No implementation files were modified.

- **11/11 focused unittest tests passed**, 0.377 s.
- `python -m ruff check code/ratio_local.py tests/test_ratio_local.py --select E9,F`: passed.
- `python -m compileall -q code/ratio_local.py tests/test_ratio_local.py`: passed.

## Verified properties

- Flat ratio values across black/gray/bright produce no local response, correct mean,
  zero deviation and finite confidence. A complete random-texture window's mean/std
  matches an independent float64 calculation; deviation is nonnegative and finite.
- Equal cached R gives identical local support/confidence despite different synthetic
  raw-gray shadow bands. This confirms the function consumes the declared cached R;
  it does not prove real-world shadow invariance of the entire pipeline.
- Protected pixels outrank higher-confidence unforced pixels. Remaining equal-score
  ties follow stable raster order. Output has exactly min(candidate count, budget)
  pixels, stays within the candidate, respects zero budgets and retains fitting masks.
  An oversized forced core raises instead of silently dropping protected pixels.
- Real `refine` calls retain the enlarged eroded coarse core with both a fitting
  core-only candidate and an expanded supported component requiring ranked reduction.
- Small inputs up to and including a long side of 640 return exact enlarged P without
  even requiring ratio stages. Large odd/nonempty synthetic inputs exercise the
  enhanced path; tiny and extreme shapes also return original-size boolean masks.
  Image empty/nonempty decisions match P and foreground count never exceeds P.
- Input image, stage arrays, budget candidate/core and confidence remain unchanged.

No mathematical implementation failure was found for the declared inputs. The prior
raw-gray candidate remains rejected after its shadow stress failure. These tests do
not establish improved segmentation, source generalization or product admission for
the new ratio candidate. A pixel-count budget can relocate pixels and is not a
subset-of-P guarantee; semantic seam discrimination is outside this correctness check.
