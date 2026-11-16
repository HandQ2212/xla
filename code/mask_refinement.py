"""Research: local-intensity boundary refinement within a legacy coarse mask."""
import cv2
import numpy as np

import pipeline as pl

CONFIGS = [dict(name=f"native_w{w}_o{o}_g{int(g)}", window=w, offset=o, guard=g)
           for w in (15, 31) for o in (0, 4, 8) for g in (False, True)]


def refine(image, coarse, config):
    gray = cv2.medianBlur(pl.to_gray(image), 3).astype(np.float32)
    scale = max(image.shape[:2]) / pl.P["size"]
    window = max(3, int(round(config["window"] * scale)) | 1)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window), normalize=True,
                         borderType=cv2.BORDER_REFLECT_101)
    support = gray < mean + config["offset"]
    if config["guard"]:
        core = cv2.erode(coarse.astype(np.uint8), pl.ellipse(3)) > 0
        support |= pl.to_orig(core, image.shape[:2])
    return pl.to_orig(coarse, image.shape[:2]) & support


def predict(image, config):
    coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
    return refine(image, coarse, config)
