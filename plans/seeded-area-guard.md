# Exact area guard for selected seeded local candidate: prospective test

Training-only seeded_m3_w71_supported improves PCS positiveDice .2879->.3217,
precision .3336->.3749, recall .2981->.3200 and negative alarms18->14; CFD Dice
does not regress. But PaveSeg maximum negative area rises .03972->.04501, so the
unguarded method fails its declared negative-area condition and is not admitted.

Test ONE fixed guarded configuration, before any validation/test inference:
median3, window71, k1.5, supported blend, existing P postprocessing. At original
resolution, if the candidate has more foreground pixels than the baseline, return
the baseline instead. Otherwise return the candidate. This is an exact global
area constraint, with no fitted threshold, source detection or GT information.
It guarantees no new image-level false alarm, no increased area on any negative
(indeed any image), but can forfeit valid recall extensions and must be measured.

Development acceptance remains PCS positiveDice+0.02, IoU+, precision/recall>=P-.02,
negativecount/mean/maxarea nonincrease, CFD Dice>=P-.01, PaveSeg Dice>=P-.005. Freeze
only if passed, then all498officialvalidation without retuning, then untouched221
PCS test (exclude10seenIDs),40frozenDeepCracktestimages and predefined stress tests.
Same quality conditions, testpairedCI>0, runtime<=2xP. Rejected methods stay research.
