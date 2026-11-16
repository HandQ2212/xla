# Native-resolution boundary refinement: prospective hypothesis

The symmetric-line waves are rejected: they either retain too much texture or
remove true cracks. No validation/test images or scores have been inspected.

New, narrower task: improve pixel delineation inside already-detected crack regions
on the original PCS task. This is not semantic recognition of sealed seams/joints
and cannot rescue wholly missed cracks. Keep the baseline instance search unchanged.

At native resolution, median-filter gray with 3x3, estimate local intensity with
a box mean at a window scaled from the 640-pixel working size. Intersect the
upscaled legacy mask with gray < mean + offset. Optional coarse-mask erosion keeps
the interior and applies intensity refinement only to its boundary. This uses
C2 interpolation, C3 local/median filtering, C5 thresholding and C7 intersection.
It produces a subset of the baseline mask, so no new negative-image alarm or
negative foreground pixel can be added. This does not prove positive quality.

Freeze 12 combinations before scoring: mean window {15,31} at working scale,
offset {0,4,8} gray levels, interior guard {off,on} using 3x3 erosion of the
coarse legacy mask before nearest-neighbor enlargement. All other P steps fixed.

Tune only on the frozen 150 PCS training pairs plus the 120 seen examples.
Choose one configuration before official validation. Positive macro Dice gain on
PCS train must be >=0.02 and IoU positive; precision gain >=0.02, recall loss <=0.03.
CFD seen mean Dice must not drop >0.01; PaveSeg development Dice must not drop >0.005.
Always report source-specific losses and the unchanged old-method failure cases.

Run the one frozen configuration on all 498 official validation images without
retuning. Require positive macro Dice gain >=0.02, IoU gain and precision gain >=0.02,
recall loss <=0.03; no increase in negative false alarms or area. If failed, reject.
If passed, freeze code+params and run original PCS test excluding all 10 observed
bundled IDs. The same quality conditions apply, and the paired image-bootstrap
95% CI for Dice delta must be >0 (10,000 resamples seed42; unknown source sessions
limit inference to this release, not all roads). Also check all previously declared
stress transforms with positive mean Dice delta and no greater negative alarm rate.

Use 40 fresh DeepCrack test images with authored pixel masks as separate-source
check: mean Dice must not fall >0.01, disclose scope/semantic mismatch. The new
method is admitted only for measured pixel refinement, never called a solution
to hard-negative semantic confusion. Median runtime over 3 repeats after warmup
must be <=2x baseline at original output resolution. Preserve legacy entry point.

If any principal validation/test gate fails, keep the candidate research-only.
