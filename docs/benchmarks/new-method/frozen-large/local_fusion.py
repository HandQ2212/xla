"""Research: locally normalized crack support seeded by the legacy detection."""
import cv2
import numpy as np

import experiments as exp
import local_contrast as local
import pipeline as pl

CONFIGS = [dict(name=f"seeded_m{m}_w{w}_{b}", median=m, window=w, k=1.5, blend=b)
           for m in (3, 5) for w in (31, 71) for b in ("connected", "supported", "imagegate")]


def fuse(coarse, support, blend):
    if not coarse.any():
        return np.zeros_like(coarse)
    if blend == "imagegate":
        return support
    connected = exp.reconstruct(coarse, support)
    if blend == "supported":
        neighborhood = cv2.dilate(support.astype(np.uint8), pl.ellipse(3)) > 0
        return connected | (coarse & neighborhood)
    return connected


def predict(image, config):
    coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
    if config.get("large_only") and max(image.shape[:2]) <= pl.P["size"]:
        return pl.to_orig(coarse, image.shape[:2])
    if not coarse.any():
        return pl.to_orig(coarse, image.shape[:2])
    raw, stages = local.raw_mask(image, config)
    support = pl.post(raw, pl.P)[0]
    mask = fuse(coarse, support, config["blend"])
    prediction = pl.to_orig(mask, image.shape[:2])
    baseline = pl.to_orig(coarse, image.shape[:2])
    if config.get("area_guard") == "ranked":
        contrast = (stages["mean"] - stages["den"]) / np.maximum(stages["deviation"], 1.)
        confidence = cv2.resize(contrast, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        prediction = area_budget(prediction, int(baseline.sum()), confidence)
    if config.get("preserve_detection") and (not prediction.any() or
            prediction.sum() < config.get("min_retained", 0) * baseline.sum()):
        return baseline
    return prediction


def area_budget(mask, budget, confidence):
    """Retain strongest candidate pixels under an exact original-resolution budget."""
    indices = np.flatnonzero(mask)
    if len(indices) <= budget:
        return mask
    order = np.argsort(-confidence.ravel()[indices], kind="stable")
    result = np.zeros(mask.shape, bool)
    result.ravel()[indices[order[:budget]]] = True
    return result
