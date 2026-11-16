# Native-mask refinement invariant tests

Date: 2026-11-15. This is a mathematical research-code audit, not a quality gate.
Only synthetic numerical images/masks were used. No train, validation or test source
photographs or GT files were read, and no implementation file was modified.

- **7/7 native-refinement tests passed**, 0.235 s.
- **11/11 symmetric-line tests passed**, 0.715 s, including the new persistence test.
- Ruff `--select E9,F` and compileall passed for the two research modules and their
  focused test files.

## Native-refinement properties

- Every one of the 12 declared configurations returned a subset of the nearest-neighbor
  enlarged baseline mask. An empty baseline remained empty despite a strong synthetic
  dark line, so this operation cannot introduce a new negative-image alarm.
- With a flat image and zero offset, intensity support was empty; enabling the guard
  retained exactly the enlarged eroded core. No pixel outside baseline support appeared.
- At zero offset, a dark native-resolution three-pixel line remained while the bright
  boundary band of an oversized coarse mask was removed.
- Increasing offset 0→4→8 could only add retained pixels for a fixed window/guard.
- Odd, tiny and extreme-aspect inputs produced boolean masks at original size across
  all configurations. Both input image and coarse mask remained unchanged.

## Line-response persistence update

Default persistence equals explicit persistence 3. For the same width and image,
persistence 7 yielded a pixelwise response no greater than persistence 3, with strict
suppression somewhere in the test texture. Existing constant, monotonic-step,
dark-line, rotation/reflection, border and component-evidence tests still passed.

No mathematical implementation failure was found under these declared inputs.
The subset property does not prove improved Dice, recall, or semantic rejection of
sealed seams. Positive offsets deliberately permit some locally flat pixels within
the baseline mask. No validation/test performance claim follows from these tests.
The full suite was not rerun here because earlier integration tests read bundled
photographs and GT; this task was restricted to synthetic inputs.
