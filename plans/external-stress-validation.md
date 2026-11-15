# External stress validation: predeclared acceptance gate

Status: protocol fixed before further algorithm selection. Supersedes the earlier plan's willingness to expose unvalidated presets. Scope: road-crack segmentation using only the image-processing course (chapters 2–5, 7–8). Baseline is the unchanged `P` configuration from mailinh. API/app correctness fixes are assessed separately from algorithm quality.

## 1. Collect and freeze evidence

Target a small external corpus of at least 32 distinct real photographs: 16 clearly cracked road surfaces with pixel GT, and 16 confidently noncracked hard negatives. Prefer publicly supplied expert dataset masks for positive examples. Include at least four independent capture/source groups in total and multiple source groups in each split. A repository mirroring the same dataset is the same source, not independent evidence.

Positive strata: thin/faint cracks; wide/branching or curved cracks; coarse aggregate; uneven illumination/shadow. Negative strata: sound coarse asphalt; painted road markings; construction joints or sealed repair seams; cables, twigs, tire marks, shadows or wet streaks that resemble cracks. Aim for at least two independent examples per stratum across the corpus; labels may overlap. Include difficult counterexamples, not only clean close-ups. Exclude uncertain scenes from scored negatives; retain them as qualitative ambiguity cases.

For every image record ID, landing-page URL, direct image URL, original dataset/capture/source group, download time, sha256, original dimensions, usage/license information when available, semantic label, strata, GT origin and split in a versioned manifest. Check exact duplicates by hash and inspect near duplicates / adjacent crops. Put every image, crop and transformation of one scene in one group. Do not fabricate ground truth from the pipeline output. Manual masks require image-only annotation before seeing candidate predictions, a documented annotation method and visible QA; an approximate mask cannot support a precise performance claim.

Freeze a discovery split (8 positives + 8 negatives) and untouched confirmation split (8 positives + 8 negatives), grouped by original source/capture. Confirmation groups must not appear in discovery or bundled samples. Assign splits and lock the manifest before running candidates. Preserve downloaded originals; do not silently resize before prediction or scoring. If the minimum corpus or trustworthy GT is unavailable, report insufficient evidence and keep baseline as default.

## 2. Candidate selection and freeze

Evaluate baseline and at most four documented classical variants on discovery. Existing P_RECON/P_THIN observations and the 20 bundled samples are discovery information only. Record all variants and failures, not only the winner. Select at most one candidate against the gate below, freeze the exact code commit and parameter JSON, then run confirmation once. Do not tune on confirmation. A failed confirmation means rejection for this iteration; another attempt needs new untouched confirmation groups.

Keep unverified candidates out of the normal app preset selector, recommended/default paths and improvement claims. They may remain in an explicitly research-only benchmark if useful for reproducibility. In particular, if reconstruction and reduced median fail this gate, remove their app options and retain baseline `P`. An honest outcome may be correctness improvements plus a rejected algorithm experiment.

## 3. Report separate metrics

On pixel-annotated positives, compute original-resolution per-image Dice, IoU, precision and recall; report their macro means, pooled TP/FP/FN and paired candidate-minus-baseline differences. Show per-source and per-stratum results, including every regression. A negative's empty-empty Dice of 1 must not inflate the positive segmentation average.

On confidently negative real images, report false-alarm count (any predicted pixel), foreground area percentage per image, mean and maximum. Do not redefine a smaller nonempty mask as a correct negative. Semantic crack/no-crack labels permit image-level metrics only; without verified spatial GT, never calculate or invent pixel Dice, IoU, precision or recall. Qualitative ambiguous images get original/prediction overlays and written failure observations, not accuracy scores.

Timing: warm each method, use identical inputs/machine/thread settings, run three repeats and report median wall time per original image. Save raw predictions, overlays, per-image CSV, summary, environment versions, source hashes, exact parameters and failures. Errors/unreadable files must appear as failures, not silently reduce the denominator.

## 4. Acceptance gate (all conditions)

The discovery candidate and its single untouched confirmation run must independently satisfy:

1. On annotated positives, mean paired Dice improvement is at least +0.02 absolute; mean paired IoU improves; macro precision and recall each fall by no more than 0.01 absolute.
2. No positive source group or declared positive stratum with at least two images loses more than 0.02 mean Dice. No previously detected positive image becomes a wholly empty prediction. An individual Dice loss greater than 0.05 is a material failure requiring explicit rejection or a new, narrower task definition before new data collection; do not hide it behind the mean.
3. Negative false-alarm count does not increase, no previously correct negative becomes nonempty, and mean/max predicted negative area do not increase. At least one existing false alarm must be eliminated if the claimed improvement is specifically false-positive reduction.
4. For the stronger claim that the improvement is reliable across external sources, a paired bootstrap over independent source/capture groups (10,000 resamples, seed 42) has a 95% percentile interval for mean positive Dice change entirely above zero. Too few independent groups or an interval touching zero means evidence is inconclusive, even if the point gain looks good. Report this uncertainty; do not reinterpret pixels or crops as independent samples.
5. All cases produce finite, correctly shaped boolean masks; regression tests pass; median runtime is at most 1.5 times baseline unless a measured, explicit quality/latency tradeoff is accepted separately. There must be no drop in output resolution to meet the limit.

Failure of any gate keeps `P` as default and the algorithm change out of the app. Passing this small corpus supports only the tested external conditions, not a universal or full-PCS/CFD improvement claim. Historical README full-test numbers remain clearly attributed to the old benchmark until that benchmark is rerun.

## 5. Derived adversarial/stability checks

After freezing the candidate, apply the same deterministic transformations to baseline and candidate originals: horizontal flip; gamma 0.75 and 1.25; JPEG quality 60. Keep original scene/source group and positive GT spatially aligned (flip mask only for flip). For each transform report its own paired metrics; never count transformed versions as new independent images or use them to inflate confidence. These preserve scene identity, but appearance can make very faint cracks genuinely unresolvable; record that limitation.

Also retain natural hard negatives where strong dark lines defeat the model: sealed seams, joints, shadows and foreign objects. These are separate stress categories; do not describe synthetically drawn lines as real-world validation. Any runtime crash, entirely lost previously detected crack, newly false-alarming clean scene, or positive-stratum Dice drop beyond the above bounds on a declared transformation blocks a robustness claim. If only aggregate natural-image acceptance passes, state that limited result and the failed stress categories plainly.

## Execution checklist

- [x] Collect all 100 public preview originals/labels and pinned provenance; screen duplicates and explain ambiguity.
- [x] Freeze manifest, ID blocks, splits and gate before scoring; record that capture/source independence is unavailable.
- [x] Run all fixed discovery candidates under six conditions; preserve all successes and failures.
- [x] Reject both candidates; remove their app options and production code, retain research-only reproductions.
- [x] Run baseline-only confirmation under six conditions after rejecting both; save artifacts. No candidate tested or selected on confirmation.
- [x] Report failed gates, one-source/GT limitations, and retain P without claiming improved robustness.

## Frozen execution scope (before any external candidate scores)

Use all 100 public PaveSeg preview images at commit dc21495f61867c94336283636959c754d192e18a.
Manifest: docs/benchmarks/external/manifest.json. Positives are labels 30 (crack) and
150 (alligator); sealed label 90 is a hard negative for open-crack detection, as are
patch/cover/joint pixels. This definition does not mean all pavement distress.

Seed 42 assigns ten consecutive-ID blocks: discovery 50 images (33 positive / 17
negative), confirmation 50 (31 positive / 19 negative). Exact hashes are unique;
cross-split 64x54 thumbnail MAE<5 screening flags no near duplicates. Capture-session
IDs are unavailable: blocks do not prove scene independence. This is ONE source and
does not satisfy the multi-source reliability gate. Results can reject candidates
and diagnose failures but cannot alone authorize a general improvement claim.

Compare existing fixed P_RECON and P_THIN with P on discovery, without tuning. If
neither passes, reject both and run only P on confirmation for baseline diagnostics.
No candidate selection after viewing confirmation. Test original, horizontal flip,
gamma powers 0.75/1.25, JPEG quality 60, and a smooth diagonal shadow (illumination
multiplier 0.35-1). Transformations remain in their original groups and are not new
photographs. Failure galleries are explanatory selections, never scoring subsets.

## Measured outcome

Discovery original positive Dice: P 0.017829, reconstruct 0.029656, thin 0.029053. False alarms: P 12/17, reconstruct 17/17, thin 16/17. Both rejected under every declared condition. Confirmation baseline original positive Dice 0.033706 and false alarms 15/19. All original negative cases together: 27/36 false alarms. No algorithm promoted. See docs/external-stress-report.md and per-image artifacts.
