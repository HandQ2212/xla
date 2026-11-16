"""Research: local statistics of illumination-normalized R with protected seeds."""
import cv2
import numpy as np

import experiments as exp
import pipeline as pl

CONFIGS = [dict(name=f"ratio_w{w}_k{k}_c{int(core)}", window=w, k=k, protect_core=core)
           for w in (31, 71) for k in (1., 1.5, 2.) for core in (False, True)]


def local_support(stages, config):
    ratio = stages["ratio"].astype(np.float32)
    shape = (config["window"],) * 2
    mean = cv2.boxFilter(ratio, cv2.CV_32F, shape, normalize=True, borderType=cv2.BORDER_REFLECT_101)
    square = cv2.boxFilter(ratio * ratio, cv2.CV_32F, shape, normalize=True, borderType=cv2.BORDER_REFLECT_101)
    deviation = np.sqrt(np.maximum(square - mean * mean, 0))
    raw = ratio < mean - config["k"] * deviation - 1
    support = pl.post(raw, {**pl.P, "tmin": 0})[0]
    confidence = (mean - ratio) / np.maximum(deviation, 1.)
    return support, confidence, dict(local_ratio=ratio, local_mean=mean,
                                      local_deviation=deviation, local_raw=raw, local_support=support)


def budget_mask(candidate, forced, budget, confidence):
    forced = forced & candidate
    slots = budget - int(np.count_nonzero(forced))
    if slots < 0:
        raise ValueError("Forced coarse core exceeds the legacy area budget")
    if np.count_nonzero(candidate) <= budget:
        return candidate
    indices = np.flatnonzero(candidate & ~forced)
    order = np.argsort(-confidence.ravel()[indices], kind="stable")
    result = forced.copy()
    result.ravel()[indices[order[:slots]]] = True
    return result


def refine(stages, shape, config):
    coarse = stages["Z"]
    baseline = pl.to_orig(coarse, shape)
    if max(shape[:2]) <= pl.P["size"] or not coarse.any():
        return baseline
    support, confidence, _ = local_support(stages, config)
    connected = exp.reconstruct(coarse, support)
    nearby = cv2.dilate(support.astype(np.uint8), pl.ellipse(3)) > 0
    core = (cv2.erode(coarse.astype(np.uint8), pl.ellipse(3)) > 0
            if config["protect_core"] else np.zeros_like(coarse))
    candidate = pl.to_orig(connected | (coarse & nearby) | core, shape)
    budget = int(np.count_nonzero(baseline))
    if np.count_nonzero(candidate) > budget:
        full_confidence = cv2.resize(confidence, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        candidate = budget_mask(candidate, pl.to_orig(core, shape), budget, full_confidence)
    return candidate if candidate.any() else baseline


def predict(image, config):
    stages = pl.pipeline(image, pl.P, steps=True)
    return refine(stages, image.shape[:2], config)
