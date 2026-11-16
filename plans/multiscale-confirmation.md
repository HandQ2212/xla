# Coarse/fine resolution consensus: prospective final quality hypothesis

Illumination-local ratio_w31_k1.0_c0 also FAILS its fresh500 gate: positiveDicegain
.006558 with CI[-.0000925,.012599], below .01 and crossingzero. Do not round the
lower bound into success. All previous samples/officialtests/400/500reserves are now
observed development evidence; none is fresh confirmation for another candidate.

New mechanism: cracks erased by640px resizing/median filtering may survive a finer
working resolution. Keep legacyP640 as image-level detector and seed. Run the SAME
classical P operations at fine960or1280 (cap at native longside to avoid upsampling),
using median5 but test fixed vs physicallyscaled other neighborhood/area parameters.
Physical mode scales bg,dil,close to nearest odd kernels and area by(scale^2).
Fine post's imagegate is0 because P already made the image-level decision.

Twelve configs before scoring: fine{960,1280} x geometry{fixed,physical} x
blend{fineconnectedcomponents touchedby P, connectedplusPnearfine, fullfineonlyif
Pnonempty}. Align masks at fine working size with nearest-neighbor interpolation;
supportedPneighborhood follows geometry scaling. Restore to native size. If new
maskempty returnP. Smallinputs<=640exactP. This is C2 sampling/interpolation plus
the existing C3/C5/C7/C8 operations; no learned/model detector or dataset-name rule.

Do NOT enforce the previous globalforegroundcount cap: it prevents legitimate
recall recovery where P has too few pixels. Instead measure negative pixel cost
empirically on original and shadow150train; do not claim a per-image area guarantee.
Image-positive decisions remain exactlyP by construction.

Select among configs with training positiveDicegain>=.02original and>=.01shadow,
IoUgainboth, P/R>=baseline-.02both, negative meanarea<=P and maxarea<=1.1xPmaxboth.
Choose highest mean positiveDice over bothconditions. Freeze one before final158
reserve is decoded. Runtime budget<=4xP prospectively for the fine-image pass;
report the accuracy/latency tradeoff, never lower output resolution to meet it.

Fresh final158: ALLremaining originaltrainIDs excluding historicalTUNE300,new150,
used400andused500 (union1322). Freeze sortedIDs beforedownload/view; unknowncapture
sessions constrain inference. Require positiveDicegain>=.015 AND imagepaired95%CI>0
(10kseed42), positiveIoU, P/R>=P-.02, sameimage decisions, negative meanarea<=P and
maxarea<=1.1xPmax. All five predefined stressconditions need positiveDicegain and
no additional imagefalsealarms; report every loss>.05 and newlylosttrue-pixel hits.

If no trainingconfig qualifies, do not open158. If confirmationfails, keep method
research-only and stop admission under this protocol. Exact-output engineering
optimizations are separately admissible if pixelparity and runtimebenefit proven;
do not present speed as an accuracy improvement. Keep all prior failures visible.
