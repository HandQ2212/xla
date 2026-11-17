"""Exact tiled uint8 median filtering; local executors retain no forked pool."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


def tiled_median(image, k, workers=2, axis=1):
    """Match cv2.medianBlur for a 2-D uint8 image, including replicated borders.

    Tiles have a k//2 halo on both sides. Only the tile interiors are copied back,
    so internal tile borders cannot affect a returned pixel. Small windows/images
    use OpenCV directly because threading/overlap would add work.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8:
        raise TypeError("Expected a two-dimensional uint8 NumPy array")
    if image.size == 0:
        raise ValueError("Image must be nonempty")
    if not isinstance(k, (int, np.integer)) or k < 1 or k % 2 == 0:
        raise ValueError("Median window must be a positive odd integer")
    if not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if axis not in (0, 1):
        raise ValueError("axis must be 0 or 1")
    extent = image.shape[axis]
    n = min(int(workers), extent)
    if n == 1 or k <= 5 or extent < 2 * k:
        return cv2.medianBlur(image, int(k))
    radius = int(k) // 2
    edges = np.linspace(0, extent, n + 1, dtype=int)

    def part(index):
        start, stop = int(edges[index]), int(edges[index + 1])
        top, bottom = max(0, start - radius), min(extent, stop + radius)
        tile = image[top:bottom] if axis == 0 else image[:, top:bottom]
        filtered = cv2.medianBlur(tile, int(k))
        return filtered[start - top:stop - top] if axis == 0 else filtered[:, start - top:stop - top]

    # A fresh pool is intentional: no persistent executor or PID-sensitive cache.
    with ThreadPoolExecutor(max_workers=n) as pool:
        pieces = list(pool.map(part, range(n)))
    return np.concatenate(pieces, axis=axis)
