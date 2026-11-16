# Mask restoration alignment: prospective small geometry correction

Multiscale training failed before opening the final158 reserve. User asked to
continue; the158remain untouched. Test one simpler C2/C5 issue: INTER_AREA downsamples
image values over pixel areas, while legacy INTER_NEAREST restores masks using a
different discrete sampling convention than pixel-center nearest-exact. A native
boundary can move by a pixel; this matters for thin manually annotated cracks.
Do not assume this geometric observation guarantees better GT agreement.

Freeze five restorers before scores: nearest_exact, linear thresholds .35/.45/.5,
cubic threshold.5. Use ONLY legacyP's existing640workingZ; no new detector, raw
threshold or component selection. Native longside<=640keepslegacyrestoration.
If reconstructed maskempty butlegacyisnonempty, retainlegacy. This preserves
image decisions; negativearea can change slightly with resampling, not guaranteed
to be a pixel subset. Original-resolution bool masks remain mandatory.

Develop on fixed150train originals+predeclaredshadow. Choose highest mean positive
Dice among restorers with originalgain>=.002, shadowgain>=.001, positiveIoUgain,
precision/recall each>=P-.005, negative meanarea<=1.02xP, maxarea<=1.05xP. This is
a prospectively SMALL boundary-alignment improvement, not a new high-gain detector.

Freeze one before final158 firstdecoding. Require positiveDicegain>=.002 ANDpaired
image-bootstrap95%CI>0 (10kseed42), IoUgain, P/R>=P-.005, sameimage decisions,
same negativearea bounds. Each fixed stress must have positiveDicegain; report all
regressions and no source-independent claim. Runtime<=1.1xP before any independent
exact-output engineering optimization. If no configqualifies, leave158unopened.
Failure means no admission. No claims of solving seams/shadows or of large gains.
