"""Rejected research candidates, retained only for reproducing benchmark evidence.

The production app imports pipeline.py and does not expose these methods.
"""
import cv2
import numpy as np

import pipeline as pl

P_RECON = {**pl.P, "edge_mode": "reconstruct"}
P_THIN = {**pl.P, "median": 3}


def reconstruct(seed, support):
    """C7 binary reconstruction: keep 8-connected support regions touched by seeds."""
    _, labels = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
    keep = np.zeros(int(labels.max()) + 1, dtype=bool)
    keep[np.unique(labels[seed & support])] = True
    keep[0] = False
    return keep[labels]


def pipeline(image, params, steps=False):
    """Research entry point; the legacy route calls the unchanged production method."""
    if params.get("edge_mode") != "reconstruct":
        return pl.pipeline(image, params, steps=steps)
    raw, stages = pl.raw_mask(image, params)
    raw = reconstruct(raw, stages["D"])
    z, c, a, f = pl.post(raw, params) if params.get("post", True) else (raw,) * 4
    if steps:
        return dict(stages, M=raw, C=c, A=a, F=f, Z=z)
    return pl.to_orig(z, image.shape[:2])
