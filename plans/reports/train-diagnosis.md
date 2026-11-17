# Training-only loss diagnosis

Baseline: existing `P` from `code/pipeline.py` (`size=640, median=5, bg=71,
hist_w=3, dil=3, close=7, a_min=30, e_min=3, tmin=0.3`). Used exactly the
150 frozen IDs in `docs/benchmarks/new-method/train-manifest.json`: **119 positive,
31 negative**. All 31 negatives are group C; A/B/C positive counts are 7/48/64.
No val, test or DeepCrack test image/GT was read by this diagnostic.

Every mask was restored to the original resolution with nearest-neighbor before
scoring. Dice/IoU/Precision/Recall below are macro means on positive images only;
negative false alarms are reported separately. `Empty+` is a wholly empty prediction;
`No TP+` additionally includes nonempty masks that miss the GT entirely.

## Existing stages

| Stage | Dice+ | IoU+ | Precision+ | Recall+ | Empty+ /119 | No TP+ /119 | Negative FA /31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| D | 0.2452 | 0.1517 | 0.1957 | 0.4185 | 0 | 0 | 31 |
| E | 0.0327 | 0.0169 | 0.0175 | 0.7239 | 0 | 0 | 31 |
| M | 0.2412 | 0.1489 | 0.2091 | 0.3515 | 0 | 0 | 31 |
| C | 0.2540 | 0.1585 | 0.2035 | 0.4274 | 0 | 0 | 31 |
| A | 0.3072 | 0.2037 | 0.2943 | 0.3988 | 0 | 3 | 31 |
| F | 0.3226 | 0.2228 | 0.4189 | 0.3216 | 1 | 12 | 30 |
| Z | 0.2879 | 0.2021 | 0.3336 | 0.2981 | 28 | 33 | 18 |

D = dark region after global triangle threshold; E = Sobel/Otsu/dilation support;
M = D intersection E; C = closing; A = area filtering; F = elongation filtering;
Z = image-area decision.

**Main pixel loss happens before postprocessing.** D recalls only 0.4185 of GT per
positive image on average, while E recalls 0.7239. Their intersection M recalls
0.3515. Pooled recall is D 0.4614, E 0.7576, M 0.3910: intersecting E removes
15.25% of D true pixels, but the dark-region threshold is the larger bottleneck.
This does not isolate resizing, denoising, illumination correction and thresholding
from one another; it shows their combined D representation is already restrictive.

Closing helps recover GT support (pooled TP 532,750 at M to 645,229 at C). Area
filtering then removes 5.94% of C true pixels while reducing positive-image FP.
Elongation removes 18.09% of A true pixels but removes 65.18% of A false pixels;
it is a real recall/precision tradeoff, not a plainly useless stage.

The final image gate removes only 4.92% of F true pixels in total but creates **27
additional empty positive images** (1 -> 28). It also turns 12 extra negative images
empty (negative FA 30 -> 18). Hence it disproportionately discards small cracks and
changes image-level results, even though its loss of total positive pixel mass is
smaller than the pre/post region-selection losses.

## Three requested ablations (diagnosis only)

| Configuration | Dice all150 | Dice+ | IoU+ | Precision+ | Recall+ | Empty+ /119 | Negative FA /31 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P | 0.3151 | 0.2879 | 0.2021 | 0.3336 | 0.2981 | 28 | 18 |
| no_image_gate | 0.2626 | 0.3226 | 0.2228 | 0.4189 | 0.3216 | 1 | 30 |
| no_elongation | 0.2636 | 0.2986 | 0.1985 | 0.2824 | 0.3916 | 6 | 27 |
| median3 | 0.2891 | 0.2720 | 0.1889 | 0.3453 | 0.2649 | 32 | 20 |

- Removing the image gate improves positive Dice by 0.0347 but adds 12 negative
  false alarms. It cannot simply replace P under the existing false-positive guard.
- Removing elongation improves recall by 0.0935, but gains only 0.0107 positive
  Dice, reduces precision, and adds 9 false alarms. Preserve selective shape/noise
  suppression rather than removing it wholesale.
- Median 3 is worse overall: lower positive Dice and recall, two more negative
  false alarms. Its D-stage recall is 0.3590 vs 0.4185 for median 5; raw M recall is
  0.3234 vs 0.3515. The intuitive thinner-filter explanation alone is insufficient:
  the denoised image and the automatically selected global threshold interact.

## Source-group differences

| Group | Positive / negative | P Dice+ | No-gate Dice+ | No-elongation Dice+ | Median3 Dice+ |
|---|---|---:|---:|---:|---:|
| A | 7 / 0 | 0.0022 | 0.0413 | 0.0674 | 0.0431 |
| B | 48 / 0 | 0.4676 | 0.5278 | 0.4930 | 0.4175 |
| C | 64 / 31 | 0.1844 | 0.1995 | 0.1781 | 0.1879 |

A fails strongly even before a generic postprocessing relaxation. B benefits most
from removing the image gate; C contains the negative/noise tradeoff. These are
analysis strata only: filenames/group IDs must not be used as detector inputs.

Outputs: `results/train-diagnosis/per_image.csv`, `summary.csv`, and the reproducible
training-only script `diagnose.py` in the same ignored directory. No production code
or parameters changed. No ablation was promoted. Conclusions are training
observations and do not establish held-out performance.
