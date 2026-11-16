# Coarse/fine seeded fusion: prospective training experiment

Both symmetric-response waves and the native-intensity refinement failed their
training criteria; no official validation/test results have been opened. Their
failures stay in the research log. Merely deleting pixels loses recall.

New hypothesis: use median-5 detections as robust seeds, median-3 masks as a finer
support. Binary reconstruction keeps only fine connected regions touched by the
coarse baseline. This can extend an existing crack into finer adjacent segments
while avoiding objects disconnected from every baseline detection. It cannot
create a positive image when the baseline was empty, an exact image-alarm invariant.
It does not guarantee unchanged foreground area on already-false-positive images.
This is C3 median filtering and C7 connectivity/reconstruction/intersection/union.

Predeclare twelve combinations before scores: fine close {3,7}, fine elongation
{2,3}, blend {reconstructed fine alone, union with coarse, reconstructed fine plus
coarse pixels supported by a 3x3 dilation of fine}. Fine median=3, bg=71, size=640,
area=30, tmin=0; robust seeds are the complete legacy P mask Z at working scale.

Primary development: the fixed150PCS train pairs. Also report all seen CFD/PCS and
PaveSeg images. Require primary positive Dice+0.02 and IoU+, precision/recall each
no worse than baseline-0.02, no more negative image alarms, negative mean/max area
nonincrease; CFD mean Dice no worse than -0.01 and PaveSeg no worse than -0.005.
If none qualifies, reject without reading validation/test. If one qualifies,
freeze it and require the same principal criteria on all498officialvalidation.
Then freeze before original test excluding10seenIDs; require paired Dice95%CI>0,
same principal quality/false-alarm bounds, stress stability and >=40freshDeepCrack
test cases with no mean Dice regression>0.01. Unknown capture independence limits
inference to these releases. Runtime budget <=2x P, original output dimensions.
No app algorithm is added before these checks pass.
