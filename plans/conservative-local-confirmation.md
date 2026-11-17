# Conservative local refinement: new prospective confirmation protocol

2026-11-16. The ranked local candidate passed training/official validation, but its
official-test positive Dice gain .0100 fell below the predeclared .0200 gate and
its positive-Dice bootstrap interval crossed zero. It is NOT admitted under that
protocol. All official val/test results are now observed development information.

Failure diagnosis: local-mask filtering can delete nearly all a previously strong
legacy detection, including a .7565-Dice test case becoming empty. Do not call a
conservative correction successful merely because some negative masks disappeared.

New hypothesis: preserve the legacy detector's image decisions and reject drastic
mask reductions. Start with the frozen ranked-local method. If candidate is empty
while P is nonempty, return P. Also test minimum retained-area fractions {0,.25,.5}:
if candidate native foreground is below that fraction of P, return P. No new
pixels are added to P's image area budget, and no GT is used in this fallback.
This is a non-learned consistency check between two classical masks.

Choose ONE fraction using the original150train and120seen development examples
only; require training positiveDice gain>=.02, IoU+, precision>=P+.02, recall>=P-.02,
and CFD Dice>=P-.01. Never select using newly reserved photographs. Record the
official val/test outcomes afterward only as observed failure-analysis evidence.

Acquire a fresh400-image reserve sampled from original PCS train, excluding both
the new150train IDs and historical P's300tuning IDs. Freeze IDs by seed20261116
before any pixel decoding. These are previously unused for new-method selection,
but capture-session independence is unknown. Source and exclusions must be explicit;
do not mislabel them as the original official test split.

Before reserve inference, freeze code/config. Accept this *smaller, precision-oriented*
improvement only if reserve positiveDice gain>=.01 AND paired image-bootstrap95%CI
is entirely >0 (10k,seed42), positiveIoUgain, precision gain>=.02, recall>=P-.02.
Negative image decisions must be exactly equal to P and area per image<=P by
construction/checking; report negative foreground reduction rather than a claimed
semantic classifier gain. All declared photometric/geometric stress conditions need
positive mean Dice gain and no new image alarms. Report every large regression.

Additionally use the frozen40DeepCrack positives as separate-source evidence:
mean Dice must not fall>.01; report its own CI and unknown-capture caveat. If it
regresses, do not claim a general default improvement. Runtime median<=2xP, original
resolution/bool output, unchanged P behavior, unit/integration tests and independent
review required. No algorithm/default promotion on a failed reserve check.

This new protocol deliberately targets a modest reliable precision improvement,
not the failed earlier .02-positive-Dice claim. Preserve both histories and do not
reinterpret the previously observed official test as fresh confirmation.

## Resolution scope, fixed before reserve decoding

The frozen40DeepCrack run exposed a real transfer regression: positiveDice .5796
->.5371, despite precisiongain. Reject an unrestricted/general replacement. Those
40cases are now observed evidence, not fresh confirmation for another method.

Narrow the method's admissible domain to LARGE inputs requiring downsampling by
the already-existing P size640. Inputs with native longside<=640 use the exact
legacy output (no new pixel algorithm), avoiding extrapolation to upsampled images
where no improvement was established. This is an explicit resolution-scope guard,
not dataset-name recognition, bottomcrop, or a claim DeepCrack confirms enhancement.
Its bypass is mechanically identical to P, and must be unit/regression tested.

Freeze this one scoped configuration before reserve400 pixels are opened. The
reserve supplies fresh evidence for the enhanced LARGE-image path, with the same
positiveDice>=.01 and CI>0, precisiongain>=.02, recall>=P-.02, unchanged detection
decisions and per-image-area<=P. Report all earlier official-test/DeepCrack failures
and the restricted scope. Do not claim cross-source or universal improvement;
additional high-resolution external sources remain future validation. No further
tuning or scope changes are allowed after reserve results for admission here.
