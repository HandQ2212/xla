# Local-contrast support with robust image seeds: prospective refinement

The twelve standalone local-contrast candidates completed development only. Window71,
k1.5 improves PCS training positiveDice .2879 -> .3565 (median3), but increases
negative alarms18->23 and loses CFD boundary recall. No official val/test was read.
Do not ship the standalone version.

New composition: retain legacy P's image decision as a prerequisite, use the richer
local-normalized dark mask only to segment/extend already-supported objects. This
cannot introduce a new positive image when P was empty; it can still alter pixel
false positives within existing positive images, which must be measured.

Before scoring, freeze twelve combinations: localmedian{3,5}, window{31,71}, k1.5
and blend{connected local components touched by P; connected local plus P pixels
within a3x3dilation of local; full localmask onlyif P image is nonempty}. All local
postprocessing retains close7/area30/elongation3/tmin0.3, and P is unchanged.
The supported blend aims to retain useful coarse boundaries while recovering
weak connected crack branches; the image-gated blend explicitly tests whether
image-level gating alone is sufficient. No per-dataset rule or bottomcrop.

Keep the local-contrast plan's training and officialvalidation/test quality gates:
PCS positiveDice+0.02, IoU+, precision/recall>=P-.02; no negative alarm/mean/maxarea
growth; CFD >=P-.01, PaveSeg >=P-.005 with no negative deterioration. Freeze one
qualifying candidate before all498officialvalidation, then unchanged test excluding
10seenIDs and40frozenDeepCrack cases; paired testCI>0 andstress checks. Any failed
candidate stays research-only. This is a new classical composition, not an assertion
that dark seams can be semantically classified from grayscale geometry alone.
