"""Research: restore the same coarse labels using pixel-center interpolation."""
import cv2
import numpy as np

import pipeline as pl

CONFIGS = [dict(name="nearest_exact", interpolation="nearest_exact", threshold=.5)]
CONFIGS += [dict(name=f"linear_{t}", interpolation="linear", threshold=t) for t in (.35, .45, .5)]
CONFIGS += [dict(name="cubic_0.5", interpolation="cubic", threshold=.5)]


def restore(coarse, shape, config):
    baseline = pl.to_orig(coarse, shape)
    if max(shape[:2]) <= pl.P["size"] or not coarse.any():
        return baseline
    interpolation = {"nearest_exact": cv2.INTER_NEAREST_EXACT,
                     "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[config["interpolation"]]
    data = coarse.astype(np.uint8) if config["interpolation"] == "nearest_exact" else coarse.astype(np.float32)
    resized = cv2.resize(data, (shape[1], shape[0]), interpolation=interpolation)
    candidate = resized > config["threshold"]
    return candidate if candidate.any() else baseline


def predict(image, config):
    coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
    return restore(coarse, image.shape[:2], config)
