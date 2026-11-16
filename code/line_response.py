"""Research prototype: two-sided dark-line contrast from C3/C5/C7 operations."""
import cv2
import numpy as np

import pipeline as pl
from experiments import reconstruct

CONFIGS = [dict(name=f"line_w{width}_t{factor}", method="line", widths=width, factor=factor)
           for width in (2, 3) for factor in (.75, 1., 1.25)]
CONFIGS += [dict(name=f"gate_w{width}_f{fraction}", method="gate", widths=width, fraction=fraction)
            for width in (2, 3) for fraction in (.1, .2)]
CONFIGS += [dict(name=f"augment_w{width}", method="augment", widths=width, fraction=.1)
            for width in (2, 3)]


def shifted(array, dx, dy):
    margin = max(abs(dx), abs(dy))
    h, w = array.shape
    padded = cv2.copyMakeBorder(array, margin, margin, margin, margin, cv2.BORDER_REFLECT_101)
    return padded[margin + dy:margin + dy + h, margin + dx:margin + dx + w]


def response(image, width=3, persistence=3):
    gray = pl.to_gray(image).astype(np.float32)
    den = cv2.GaussianBlur(gray, (3, 3), .6)
    background = cv2.GaussianBlur(den, (0, 0), 12)
    result = np.zeros_like(den)
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        directional = np.zeros_like(den)
        for distance in range(1, width + 1):
            first = shifted(den, dx * distance, dy * distance) - den
            second = shifted(den, -dx * distance, -dy * distance) - den
            contrast = np.minimum(first, second)
            directional = np.maximum(directional, contrast)
        # Grayscale erosion along the line suppresses short texture responses.
        directional = np.minimum.reduce([shifted(directional, -dy * step, dx * step)
                                          for step in range(-persistence // 2 + 1, persistence // 2 + 1)])
        result = np.maximum(result, directional)
    return np.clip(128 * result / np.maximum(background, 30), 0, 255).astype(np.uint8)


def line_mask(strength, factor=1.):
    otsu, _ = cv2.threshold(strength, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(3., otsu * factor)
    strong = strength > threshold
    support = strength > max(1., threshold * .5)
    raw = reconstruct(strong, support)
    closed = pl.closing(raw, 3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < 30:
            continue
        component = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        # C8 compactness: long thin or branched regions, rather than a long bounding box only.
        keep[i] = max(w, h) >= 25 and perimeter * perimeter / (4 * area) >= 6
    return keep[labels], threshold


def gate_components(mask, evidence, fraction):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    support = np.bincount(labels[evidence], minlength=count)
    keep = support / np.maximum(stats[:, cv2.CC_STAT_AREA], 1) >= fraction
    keep[0] = False
    return keep[labels]


def predict(image, config, cached=None):
    if cached is None:
        small = pl.resize_long(image, pl.P["size"])
        old = pl.pipeline(image, pl.P, steps=True)["Z"]
        strengths = {config["widths"]: response(small, config["widths"], config.get("persistence", 3))}
        cached = (old, strengths)
    old, strengths = cached
    key = (config["widths"], config.get("persistence", 3))
    strength = strengths[key] if key in strengths else strengths[config["widths"]]
    candidate, threshold = line_mask(strength, config.get("factor", 1.))
    if config["method"] in {"gate", "augment"}:
        evidence = cv2.dilate((strength > threshold).astype(np.uint8), pl.ellipse(3)) > 0
        validated = gate_components(old, evidence, config["fraction"])
        candidate = validated if config["method"] == "gate" else validated | candidate
    return pl.to_orig(candidate, image.shape[:2])
