"""Paired gates for external testing; negatives never inflate positive Dice."""
import numpy as np


def summary(rows):
    output = []
    for transform, variant in sorted({(r["transform"], r["variant"]) for r in rows}):
        subset = [r for r in rows if (r["transform"], r["variant"]) == (transform, variant)]
        pos = [r for r in subset if r["gt_pos"]]
        neg = [r for r in subset if not r["gt_pos"]]
        tp, fp, fn = (sum(r[k] for r in pos) for k in ("tp", "fp", "fn"))
        out = dict(transform=transform, variant=variant, images=len(subset), positives=len(pos),
                   positive_dice=float(np.mean([r["dice"] for r in pos])),
                   positive_iou=float(np.mean([r["iou"] for r in pos])),
                   precision=float(np.mean([r["precision"] for r in pos])),
                   recall=float(np.mean([r["recall"] for r in pos])),
                   pooled_positive_dice=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
                   positive_empty=sum(not r["pred_pos"] for r in pos),
                   negatives=len(neg), negative_false_alarms=sum(r["pred_pos"] for r in neg),
                   negative_mean_area_pct=float(np.mean([r["area_pct"] for r in neg])),
                   negative_max_area_pct=max(r["area_pct"] for r in neg),
                   median_ms=float(np.median([r["elapsed_ms"] for r in subset])))
        for tag in ("sealed", "patch", "cover", "joint"):
            count = sum(r[tag + "_pixels"] for r in subset)
            out[tag + "_fp_pct"] = 100 * sum(r[tag + "_fp"] for r in subset) / count if count else None
        output.append(out)
    return output


def gate(rows):
    """Every declared transform must pass; no averaging away catastrophic cases."""
    verdicts = []
    tables = {(r["transform"], r["variant"]): r for r in summary(rows)}
    baseline = {(r["id"], r["transform"]): r for r in rows if r["variant"] == "legacy"}
    for transform, variant in sorted(tables):
        if variant == "legacy":
            continue
        current, old = tables[transform, variant], tables[transform, "legacy"]
        subset = [r for r in rows if (r["transform"], r["variant"]) == (transform, variant)]
        pos = [r for r in subset if r["gt_pos"]]
        neg = [r for r in subset if not r["gt_pos"]]
        delta = lambda r: r["dice"] - baseline[r["id"], transform]["dice"]
        failures = []
        gain = current["positive_dice"] - old["positive_dice"]
        if gain < .02:
            failures.append("positive_dice_gain_below_0.02")
        if current["positive_iou"] <= old["positive_iou"]:
            failures.append("positive_iou_not_improved")
        for metric in ("precision", "recall"):
            if current[metric] < old[metric] - .01:
                failures.append(metric + "_regressed")
        if min(delta(r) for r in pos) < -.05:
            failures.append("individual_dice_loss_over_0.05")
        if any(not r["pred_pos"] and baseline[r["id"], transform]["pred_pos"] for r in pos):
            failures.append("new_empty_positive")
        if any(r["pred_pos"] and not baseline[r["id"], transform]["pred_pos"] for r in neg):
            failures.append("new_negative_false_alarm")
        for metric in ("negative_false_alarms", "negative_mean_area_pct", "negative_max_area_pct"):
            if current[metric] > old[metric] + 1e-12:
                failures.append(metric + "_increased")
        for tag in sorted({t for r in pos for t in r["tags"].split("|")}):
            stratum = [r for r in pos if tag in r["tags"].split("|")]
            if len(stratum) >= 2 and np.mean([delta(r) for r in stratum]) < -.02:
                failures.append("stratum_regressed:" + tag)
        for group in sorted({r["group"] for r in pos}):
            if np.mean([delta(r) for r in pos if r["group"] == group]) < -.02:
                failures.append("block_regressed:" + group)
        if current["median_ms"] > 1.5 * old["median_ms"]:
            failures.append("runtime_over_1.5x")
        verdicts.append(dict(transform=transform, variant=variant, positive_dice_gain=gain,
                             worst_individual_dice_delta=min(delta(r) for r in pos),
                             numerical_pass=not failures, failures=failures,
                             independent_source_gate_pass=False))
    return verdicts
