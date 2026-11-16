# Illumination-normalized local refinement: prospective new hypothesis

The previous raw-gray large-image candidate is REJECTED: original400reserve and
four stress types improve, but syntheticshadow Dice changes -.02017 (CIbelow0).
Its400reserve and all40DeepCrack cases are now development evidence only. Do not
retroactively admit that model or reuse those400as fresh confirmation.

Diagnosis: local standard deviation on raw gray is inflated by spatial illumination
transitions, suppressing real cracks near a shadow. Legacy P already computes
illumination-compensated ratio R. Use that SAME cached R as the input of local
mean/deviation thresholding, instead of raw gray/anothermedian. No extra model,
source-recognition or bottomcrop. C3 illumination division/localstatistics, C5
thresholding, C7 reconstruction/erosion and C8 shape filtering remain the scope.

Prospectively enumerate12configs: window{31,71}, k{1.0,1.5,2.0}, protectedcoarsecore
{off,on}. Worksize640fixed. Dark = R<mean(R)-k*std(R)-1. Post uses existingclose7,
area30,elongation3 but no second image-area gate (P already made that decision).
Keep only support components touched by P, plus P pixels within3x3dilatedsupport.
If core protection is on, preserve the3x3erosion of P as forced output pixels.
At native resolution, enforce P's foreground-count budget by retaining forcedcore
first and then strongest normalized-ratio contrast pixels; deterministicstableties.
If candidateempty, returnP. Native longside<=640 uses exactP; large-image scope
explicit, no claim that low-resolution bypass validates the enhanced path.

Develop on the fixed150train originals plus their same predefined syntheticshadow
transform, and report known examples separately. Require >=.015positiveDice gain
on originals AND >=.01gain on shadows, positiveIoU, precision/recall>=P-.02, exact
image decisions and area<=P. Choose one qualifying config using mean of original
and shadow positiveDice, freeze code+params before newreserve500 firstdecoding.

Fresh500reserve: originaltrain excluding oldnew150, historicalTUNE300, and old400.
Pool658, sample500seed20261117; frozenIDs beforedownload/view. Unknowncaptures mean
image-level evidence for this release only. No further tuning afternewreserve.
On its originals require positiveDice gain>=.01,pairedimagebootstrap95%CI>0
(10k,seed42), positiveIoU, precisiongain>=.02,recall>=P-.02. Each of the five fixed
stress conditions must also have positiveDicegain>0, sameimage decisions, native
foregroundarea<=P. Report every loss>.05 and any previously TP>0 case nowTP0;
do not confuse preserving a nonempty decision with preserving a correct crack hit.
Runtime median<=2xP afterwarmup/3repeats; exact-output optimizations may be audited.

No universal/semantic-crack classifier claim, no data-source-independent guarantee.
Small-image output must be bit-identical to P. Tests, sourcehashes, all failed
iterations, fullpairedmetrics and independentreview required before app admission.
If this newreserve fails, leave it research-only; do not change scope/cutoffs on it.
