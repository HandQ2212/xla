# Ranked area budget: prospective exact-constraint variant

Whole-image fallback on area growth reduced the selected local candidate's training
gain to .01846, below its .02 gate; reject that variant. No official validation/test
images or outputs have been read. Keep the same median3/window71/k1.5/supported blend.

Test one alternative area constraint before scores: if candidate foreground exceeds
the legacy native foreground count, retain only that many candidate pixels ranked
by the already-computed local dark contrast (mean-den)/max(deviation,1), bilinearly
interpolated to native resolution. Stable descending sort resolves ties in raster
order. This is thresholding/order statistics (C3/C5), not a trained confidence score.
If the candidate fits the budget, keep it unchanged. Do not select based on GT.

This keeps useful parts of an expanded candidate rather than reverting its entire
image. It guarantees output area<=legacy and no new image alarm, but may fragment
regions or lose true pixels; validate accuracy and topology visually. Original-size
bool masks required. All principal prior training/val/test quality gates unchanged:
PCS positiveDice+0.02, IoU+, precision/recall>=P-.02, no negativecount/area growth;
CFD>=P-.01 and PaveSeg>=P-.005. Freeze before official498val, then untouched221test
and40DeepCrack. Paired testCI>0 and runtime<=2xP. If failed keep research-only.
