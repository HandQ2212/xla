# Symmetric dark-line improvement: prospective plan

Protocol version: new research iteration, fixed before implementing/scoring this hypothesis. The earlier external-stress gate and rejection of reconstruct/thin remain valid historical results. All 100 PaveSeg preview images and all 20 bundled CFD/PCS examples have now been seen and are development data only; none can be called untouched confirmation again.

## Hypothesis and course scope

A crack is often locally darker than BOTH neighboring sides. A one-sided shadow or paint edge has this evidence on only one side. On a float gray image G after a small Gaussian blur, compute

`L(x) = max over d in {1,2,3}, u in {(1,0),(0,1),(1,1),(1,-1)} of max(0, min(G(x+d*u)-G(x), G(x-d*u)-G(x)))`.

The offsets cross a potential line; their perpendicular is its tangent. Threshold L, then use chapter-7 closing, connected components and area filtering. Compare the current box-elongation filter with a chapter-8 perimeter/area shape criterion to preserve curved/branching thin regions. First test the line evidence itself; add the alternative shape filter only if masks show the box filter is the limiting stage. Keep the grid modest and enumerate every configuration before running it; at most 12 candidates in this iteration. Work at a fixed declared resolution (initially current long side 640); always restore/score the mask at original resolution. Preserve the original image and all intermediate evidence for inspection.

Exact formula is a new composition of taught operations, not a named algorithm asserted to occur in the slides. Local course support:

- `../../tmp/slide-text/5_XLA5_Phat hien bien.txt` lines 149–220: directional finite differences, Prewitt/Sobel; lines 252–260: symmetric second differences / Laplacian; line 396: Gaussian convolution; lines 456–461: four directions 0/45/90/135.
- `../../tmp/slide-text/Chương 7-notebook.txt` section 3.3: constrained dilation and connected components; same notebook teaches closing and area filtering.
- `../../tmp/slide-text/8_XLA8_NhanDangAnh.txt` line 469: area, perimeter, aspect ratio, compactness C=4*pi*A/P^2.

Avoid Frangi, learned features, pretrained models, semantic classifiers or assumptions about a dataset's bottom crop. Skeletons are not needed for the first version; perimeter/area has clearer direct course support. Floating subtraction avoids uint8 wrap. Use reflect/replicate borders consistently and never wrap opposite image edges with `roll`. Flat images and a monotone intensity step should have no positive two-sided response away from numerical/blur effects; test that property. Diagonal offsets have sqrt(2) larger physical spacing, which must be documented rather than described as exact isotropy.

This cannot universally separate a dark sealed seam/joint from an open crack with the same local appearance. Bright paint boundaries may be rejected, but dark tar, cable, shadow intersections and coarse aggregate may still pass. Do not promise semantic discrimination from gray geometry alone.

## Development and scope fixed before results

Use all 120 seen examples for development, with source-specific reports and no rebranding old PaveSeg confirmation as independent evidence. PaveSeg positives continue to mean labels 30+150; negatives contain neither. Report crack-only (30), coarse alligator (150), and sealed/joint/patch/cover strata separately, keeping the original complete positive score as well. Do not remove alligator after scoring because its masks are coarse. Bundled CFD and PCS remain separate sources and are reported even if one loses.

Save baseline P and every enumerated candidate's original-resolution per-image CSV, masks/overlays, parameter JSON, input hashes and timing. Report mean positive Dice/IoU/precision/recall separately from negative false-alarm count and negative foreground area. Select one candidate only, freeze code+parameters, and then acquire/use new confirmation sources. A failed candidate is research-only; no app selector until the gate passes.

## Predeclared gate for this new iteration

An algorithm is admissible for the stated task of open-crack segmentation only if ALL applicable conditions below pass. This gate is declared now; do not modify it after observing the new candidate results.

Development screen:

1. Equal-source mean of positive Dice (CFD, PCS, PaveSeg each one vote) improves by at least +0.03 absolute over P; each source's mean Dice must be no worse than P minus 0.02. Equal-source positive IoU improves, and macro precision/recall each may fall by at most 0.02. Report all individual losses >0.05, their count and worst loss; at most 10% of positive images may lose >0.05. These losses are an explicit aggregate tradeoff for this iteration, not a claim every image improved.
2. On all 36 PaveSeg negative originals, total false alarms do not increase and mean/max predicted area do not increase. At most one previously empty negative may become nonempty, and only if at least two previous false alarms are eliminated (paired changes must be listed). If claiming improved false-positive control, require at least 20% fewer total false alarms as well.
3. For each declared stress condition (horizontal flip, gamma 0.75/1.25, JPEG60, existing smooth shadow), positive mean Dice delta remains positive and negative false-alarm count does not exceed P's count. A stress failure blocks a robustness claim and admission under this protocol; do not silently omit it.

Untouched external confirmation:

4. Before candidate inference, freeze a manifest of at least 40 previously unseen positive images with provided pixel GT from the official DeepCrack release and at least 24 previously unseen confidently noncracked road-surface images from a second source such as SDNET2018's noncrack pavement category. Select deterministic IDs without inspecting candidate predictions; do not pick aesthetically easy cases. Record original URLs, licenses, SHA256, dimensions, source/capture IDs when available and exact mask parsing. Inspect semantic compatibility and duplicates against every development source, using hashes plus near-duplicate visual screening. Dataset or source identity alone does not establish independent capture sessions. If unavailable, keep candidate research-only and say external confirmation is incomplete.
5. On confirmation positives, paired mean Dice improves >=+0.03 and paired mean IoU improves; macro precision/recall each may fall by at most0.02. At most10% of images may lose >0.05 Dice; show every such case and the worst loss. Paired bootstrap over available capture/source groups (10,000, seed42) must give a95% interval above0 for positive Dice delta for a reliable improvement claim. If capture IDs are unknown, an image bootstrap can describe sampled-image uncertainty only; explicitly limit the claim to the tested release and do not claim source-general reliability.
6. On confirmation negatives, false-alarm count, mean and max predicted foreground area must not increase. Because a negative-only new source can favor trivial empty masks, require the positive criterion simultaneously. No new threshold tuned from confirmation labels or outputs. Run the frozen transformations on confirmation too; require positive mean Dice delta and no greater false-alarm count for each. Failed confirmation ends this candidate; selecting another requires newly untouched data.
7. All masks have original shape/bool type; constant/tiny inputs and symmetric-response unit tests pass; baseline stays bit-identical; app/API checks pass. Median computation time after warm-up over three repeats must be <=2x P; report hardware and threads. The 2x budget is prospectively allowed for the additional directional filters, not purchased by lowering output resolution.

Only after passing development plus external confirmation expose the new algorithm alongside legacy P with its measured scope/tradeoffs. Keep historical full-PCS benchmark distinct; no claim of improving the complete PCS test without running it. If uncertainty or any gate fails, preserve research artifacts and rejected cases, continue with a new hypothesis only under a new predeclared protocol, and leave production P unchanged.

## TODO

- [ ] Freeze and list <=12 candidate parameter configurations before new scores.
- [ ] Implement symmetric response with finite-difference border tests; retain legacy P.
- [ ] Run all development sources/strata and stress conditions, preserve failures.
- [ ] Reject all or freeze exactly one qualifying candidate plus hashes.
- [ ] Acquire official untouched DeepCrack positives and independent hard negatives; freeze manifest before inference.
- [ ] Run confirmation once, save evidence and interpret uncertainty honestly.
- [ ] Independent test/review, docs, and app admission only if gates pass.

## Iterative development record

Wave 1 enumerated 12 configurations before scoring (results/line-study/configs.json).
Standalone line responses/augmentation retain too much texture and reduce the bundled
CFD/PCS scores; low-fraction component validation barely changes P. Reject these.
No new-source validation/test results have been read.

Wave 2, declared before scoring: only validate existing components, with widths
{2,3}, tangent persistence {3,7}, supported-area fraction {0.2,0.4,0.6}, exactly
12 configurations. Longer tangent erosion tests whether continuous cracks can be
separated from short aggregate responses. Keep the acceptance requirements above;
do not reinterpret a reduced foreground mask as successful positive segmentation.

The exact original PCS dataset has now been located at Zenodo record 20040434.
It supplies original train/val/test splits and binary GT. Additional tuning uses a
frozen random sample of 150 train pairs (seed 42, recorded train-manifest.json).
Choose among surviving hypotheses on the official validation split, then freeze
before evaluating test. Exclude the 10 already-observed bundled test IDs from any
untouched PCS test claim. The remaining approximately 221 test images include
approximately 44 unseen negatives, satisfying the new-negative requirement without
requiring concrete-wall negatives unrelated to this project. Verify actual counts
and source masks rather than assuming them. Add at least 40 untouched official
DeepCrack positive test images as a separate-source confirmation. Raw download and
CRC/hash verification do not authorize looking at validation/test pixels early.
