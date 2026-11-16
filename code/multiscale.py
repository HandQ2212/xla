"""Research: finer classical segmentation anchored to the unchanged P detector."""
import cv2
import numpy as np

import experiments as exp
import pipeline as pl

CONFIGS = [dict(name=f"scale_{s}_{g}_{b}", fine_size=s, geometry=g, blend=b)
           for s in (960, 1280) for g in ("fixed", "physical")
           for b in ("connected", "supported", "imagegate")]


def odd(value):
    return max(1, int(round(value)) | 1)


def fine_params(image, config):
    size = min(config["fine_size"], max(image.shape[:2]))
    params = {**pl.P, "size": size, "tmin": 0}
    scale = size / pl.P["size"]
    if config["geometry"] == "physical":
        params.update({key: odd(pl.P[key] * scale) for key in ("bg", "dil", "close")})
        params["a_min"] = round(pl.P["a_min"] * scale * scale)
    return params


def fuse(coarse, fine, config, params):
    if not coarse.any():
        return np.zeros_like(fine)
    mapped = pl.to_orig(coarse, fine.shape)
    if config["blend"] == "imagegate":
        return fine
    connected = exp.reconstruct(mapped, fine)
    if config["blend"] == "supported":
        scale = params["size"] / pl.P["size"]
        radius = odd(3 * scale) if config["geometry"] == "physical" else 3
        nearby = cv2.dilate(fine.astype(np.uint8), pl.ellipse(radius)) > 0
        return connected | (mapped & nearby)
    return connected


def refine(image, coarse, fine, config, params):
    baseline = pl.to_orig(coarse, image.shape[:2])
    if max(image.shape[:2]) <= pl.P["size"] or not coarse.any():
        return baseline
    candidate = pl.to_orig(fuse(coarse, fine, config, params), image.shape[:2])
    return candidate if candidate.any() else baseline


def predict(image, config):
    coarse = pl.pipeline(image, pl.P, steps=True)["Z"]
    if max(image.shape[:2]) <= pl.P["size"] or not coarse.any():
        return pl.to_orig(coarse, image.shape[:2])
    params = fine_params(image, config)
    fine = pl.pipeline(image, params, steps=True)["Z"]
    return refine(image, coarse, fine, config, params)
