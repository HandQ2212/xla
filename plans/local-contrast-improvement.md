# Local normalized contrast: prospective hypothesis

Previous line/native/fusion candidates failed their declared training criteria and
remain rejected. Official validation/test pixels and scores are still untouched.
Training diagnostics identify dark-region selection as the dominant loss:
D recall .4185, D-and-edge recall .3515, final recall .2981 on119trainpositives.
Median3 alone also worsened recall. These are development observations, not tests.

Use median filtering, local mean and local second moment (C3 filtering/statistics),
then C5 amplitude threshold on normalized contrast and existing C7/C8 postprocessing.
For denoised gray I, compute mean m and std s using box filters of I and I*I;
dark = I < m - k*s - 1. The exact combination is a custom classical composition,
not a claim that a named advanced detector occurs in the slides. No learning/model.
Drop the old globally normalized Sobel gate for this hypothesis: local variability
already controls texture, and a global high-gradient cutoff excludes faint cracks.

A monotone step has limited local dark contrast relative to std, whereas a narrow
valley may be many local deviations below its neighbors. This can reduce broad
shadow/paint-edge responses but does not universally separate tar seams from cracks.

Freeze twelve configurations before scores: median{3,5}, local window{31,71},
k{1.5,2.0,2.5}. Size640, close7, area30, rotated-boxelongation3 and imagegate0.3%
stay fixed initially. Report every configuration on fixed150PCS train and all120
seen development images. Require PCS train positiveDice>=baseline+.02, IoUgain,
precision/recall>=baseline-.02, negativefalsealarms/mean/maxarea nonincrease.
CFD meanDice>=baseline-.01; PaveSeg meanDice>=baseline-.005 andnegativealarms/area
nonincrease. Select only a qualifying configuration; do not retune using val/test.

Run one frozen candidate on all498officialval. Require same primaryqualitybounds
and negativecontrol. If passed freeze before PCS test excluding10seenIDs; same
bounds andpairedDice95%image-bootstrapCI>0 (10k,seed42), original-sizebooloutputs,
stresschecks with positiveDicegain andnonegativealarmgrowth. Also evaluate40fresh
DeepCracktest cases, meanDice maynotdecline>.01. Unknowncapturegroups limit claims
to tested releases. Runtime<=2xbaseline. Failure keeps newalgorithm research-only.
