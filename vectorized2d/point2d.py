from __future__ import annotations

import numpy as np

from vectorized2d import Array2D


class Point2D(Array2D):
    def euclid_dist(self, other: Point2D) -> np.ndarray:
        return (self - other).norm

    def euclid_dist_squared(self, other: Point2D) -> np.ndarray:
        return (self - other).norm_squared
