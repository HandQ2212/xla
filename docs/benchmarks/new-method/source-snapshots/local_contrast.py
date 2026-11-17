"""Research: dark regions selected using local mean and intensity deviation."""
import cv2
import numpy as np

import pipeline as pl

CONFIGS = [dict(name=f"local_m{m}_w{w}_k{k}", median=m, window=w, k=k)
           for m in (3, 5) for w in (31, 71) for k in (1.5, 2., 2.5)]


def raw_mask(image, config):
    small = pl.resize_long(image, pl.P["size"])
    gray = pl.to_gray(small)
    den = cv2.medianBlur(gray, config["median"]).astype(np.float32)
    window = (config["window"],) * 2
    mean = cv2.boxFilter(den, cv2.CV_32F, window, normalize=True, borderType=cv2.BORDER_REFLECT_101)
    square_mean = cv2.boxFilter(den * den, cv2.CV_32F, window, normalize=True, borderType=cv2.BORDER_REFLECT_101)
    deviation = np.sqrt(np.maximum(square_mean - mean * mean, 0))
    threshold = mean - config["k"] * deviation - 1.
    mask = den < threshold
    return mask, dict(small=small, gray=gray, den=den, mean=mean,
                      deviation=deviation, threshold=threshold)


def predict(image, config):
    raw, _ = raw_mask(image, config)
    final = pl.post(raw, pl.P)[0]
    return pl.to_orig(final, image.shape[:2])
