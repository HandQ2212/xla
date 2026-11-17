# Independent reserve numeric and provenance audit

2026-11-16. Audit used saved CSV/JSON and source-directory metadata only; no raw
photograph, segmentation mask or GT was opened/decoded. **Final admission result:
REJECTED**, because the predefined synthetic-shadow condition fails, despite a
real mean improvement on original photographs.

## Original 400: independent recomputation

Exactly 800 unique `(id,method)` rows: 400 originals, **327 positive / 73 negative**.
Positive Dice/IoU/Precision/Recall exclude all negative-image scores. Saved summary
and paired statistics match recomputation to numerical tolerance <1e-12.

| Metric | Legacy P | Scoped candidate |
|---|---:|---:|
| Positive macro Dice | 0.274008963 | 0.292421511 |
| Positive macro IoU | 0.195350789 | 0.209842241 |
| Positive macro precision | 0.311998028 | 0.372585569 |
| Positive macro recall | 0.285023585 | 0.281856825 |
| All-image macro Dice | 0.309002328 | 0.324054585 |
| Negative image false alarms /73 | 39.000000000 | 39.000000000 |
| Mean negative area (fraction of native image) | 0.007337145 | 0.002562385 |
| Median pipeline time (ms) | 16.965599993 | 27.483700003 |

Paired positive Dice gain is **+0.0184125475** (1.8413 percentage points). Independently
resampled the same 327 positive-image paired deltas 10,000 times using NumPy
`default_rng(42)`: percentile 95% CI **[0.0095495156, 0.0273725113]**, exactly matching
`paired.json`. This is an **image-level bootstrap**, not evidence that capture
sessions are independent or that the effect transfers to other sources.

Among positive images: **162 wins, 69 losses, 96 ties**. Positive predictions that
are completely empty remain **72/327** in both methods. However, nonempty does not
mean correct: **14 previously TP>0 images become TP=0**, with no TP=0 image becoming
TP>0. Thus positive no-TP cases increase **92 -> 106**. Preserved image decisions
must not be described as preserving every correctly detected crack.

All **400/400** image-level empty/nonempty decisions match P. All **400/400** native
foreground-area inequalities candidate<=P hold (maximum area delta 0). Saved areas
also equal `(TP+FP)/(native height*width)` and flags agree with TP/FP/FN. All 400 native
long sides exceed 640, so this reserve evaluates only the declared large-image path.
No new negative image is cured or newly false-alarming: false alarms remain **39/73**.
Mean negative foreground area falls from **0.7337% to 0.2562%**, a **65.08% relative
reduction of area**, not a reduction in false-alarm image count.

## Individual regressions: all 25 losses greater than 0.05

The worst is **B_00302, -0.5196581630 Dice**. The positive mean must not hide these:

| ID | Candidate minus P Dice |
|---|---:|
| B_00302 | -0.519658163 |
| B_00300 | -0.403582544 |
| B_00130 | -0.277780993 |
| B_00127 | -0.225320678 |
| C_01801 | -0.211106518 |
| C_00221 | -0.201152709 |
| C_00794 | -0.172789048 |
| C_01394 | -0.155594475 |
| C_01562 | -0.136102142 |
| C_00754 | -0.111483932 |
| C_00789 | -0.110326363 |
| B_00000 | -0.104551704 |
| C_00626 | -0.103337072 |
| C_00795 | -0.099618868 |
| B_00414 | -0.069959846 |
| C_01166 | -0.066320106 |
| B_00116 | -0.063231616 |
| C_00791 | -0.062780269 |
| C_00720 | -0.059265756 |
| C_00617 | -0.058790550 |
| C_00380 | -0.057584513 |
| B_00114 | -0.057572831 |
| A_00076 | -0.057445200 |
| C_02279 | -0.056052546 |
| C_00189 | -0.054739328 |

Positive-only group means all increase, but these are source-ID strata, not proof
of independent capture groups:

| Group | Positive / negative | P Dice+ | Candidate Dice+ |
|---|---|---:|---:|
| A | 21 / 0 | 0.118160 | 0.152010 |
| B | 108 / 0 | 0.554652 | 0.558768 |
| C | 198 / 73 | 0.137460 | 0.162034 |

## Performance-only refactor parity

Compared original-timing vs optimized CSVs by all 800 `(id,method)` keys. Every
mask SHA-256 and every saved quality/dimension/count/decision/area field is exactly
equal; only elapsed times differ. This verifies preserved outputs on this corpus.

The original-timing CSV has medians P **17.2396999988 ms**, candidate
**34.8651500099 ms**: **2.0223756801x**, which failed even the scoped 2x runtime gate.
The optimized timing rerun preserves all 800 output hashes and quality fields.

Recomputed medians: P **16.9655999925 ms**, candidate **27.4837000034 ms**; ratio
**1.6199662856x**, matching `performance-parity.json`. It passes the scoped protocol's
2x ceiling, **not** the earlier 1.5x ceiling. It is a ratio of per-method medians,
not a median paired latency ratio. This is not a zero-cost improvement.

## Prospective protocol and provenance

The applicable scope/gate is `plans/conservative-local-confirmation.md`: original
reserve positive Dice gain>=0.01 and image-bootstrap CI>0, IoU gain, precision
gain>=0.02, recall loss<=0.02, identical decisions, area<=P, runtime<=2x, plus positive
mean Dice gain on every declared stress condition. The original-image numeric
sub-gate passes. This is distinct from the failed prior +0.02/1.5x iteration.

Independently reconstructed original sorted 1,480 train IDs from cached ZIP central
metadata, then historical NumPy seed42 TUNE300 (A30/B90/C180). TUNE300 overlaps new
train150 by 28 IDs, so exclusion union=422 and eligible pool=1,058. Exact seed20261116
Python sampling reproduces all 400 reserve IDs; none overlap either excluded set.
The archive source is https://zenodo.org/records/20040434 with published MD5
`4b530eba05708a195355a0064e520db4` (published archive provenance; partial-download
member checks do not independently verify the whole ZIP MD5).

The frozen reserve raw SHA-256 matches its acquisition-time value
`81d002e05becefe6746904fdb247b2a7825b644bf1ef81997609d2fb56ead503`, and both large and
optimized freeze JSONs reference it. Both result metadata configurations match the
same scoped config; original metadata embeds the large freeze, optimized metadata
embeds the optimized freeze. Parameter/scope contents are consistent across runs.

Selection-before-retrieval was witnessed in the separate acquisition task. Exact
UTC algorithm-freeze timestamps are absent from these JSONs: date strings and source
hashes verify content, **not a timestamp-sealed chronology**. The pre-inference scope
ordering is documented by the protocol and parent execution record; it cannot be
independently proved from these artifact timestamps alone. Capture-session
independence remains unknown; all originals come from this PCS source.

## Five-condition stress audit: decisive failure

Independently recomputed all 4,000 stress rows (400 originals x five conditions x
both methods), their positive means and all five 10k/seed42 paired CIs. Decisions
and native area inequalities hold for all 2,000 condition-image pairs. Saved
`results/large-local-reserve-stress/paired.json` agrees with recomputation.

| Condition | P Dice+ | Candidate Dice+ | Paired gain | Image-bootstrap 95% CI | Losses >0.05 |
|---|---:|---:|---:|---|---:|
| reserve_flip | 0.273255 | 0.291636 | +0.018381 | [+0.009453, +0.027399] | 27 |
| reserve_gamma075 | 0.271741 | 0.290051 | +0.018310 | [+0.009776, +0.026927] | 25 |
| reserve_gamma125 | 0.271396 | 0.289428 | +0.018033 | [+0.009066, +0.027127] | 28 |
| reserve_jpeg60 | 0.274582 | 0.296551 | +0.021969 | [+0.013027, +0.030909] | 24 |
| reserve_shadow | 0.259395 | 0.239222 | -0.020174 | [-0.029486, -0.011035] | 88 |

Synthetic shadow has **negative mean gain -0.0201737158**, wholly negative CI
**[-0.0294862931, -0.0110348508]**, and 88 positive images losing more than 0.05 Dice.
It violates the prospective stress gate. **Do not admit this candidate**, label the
entire experiment successful, or select only its original-image score for a quality
claim. Earlier official-test and unrestricted DeepCrack failures also remain part
of the documented history; this audit does not reinterpret them as passes.

No implementation, model parameter, manifest or raw-data file was changed by this
audit. Detailed independently recomputed numbers are in ignored
`results/reserve-audit/{recomputed,stress-recomputed}.json`.
