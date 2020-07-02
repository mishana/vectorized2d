from __future__ import annotations

import numpy as np
from fast_enum import FastEnum

from vectorized2d import Array2D


class Point2D(Array2D):
    class Pairing(metaclass=FastEnum):
        ALL = 0
        ALIGNED = 1

    @staticmethod
    def _pairwise_diff(self: Point2D, other: Point2D) -> Point2D:
        self_reshaped = self.reshape((len(self), 1, 2))
        other_reshaped = other.reshape((1, len(other), 2))
        return (self_reshaped - other_reshaped).reshape(-1, 2)

    def euclid_dist(self, other: Point2D, *, pairing: Pairing = Pairing.ALL) -> np.ndarray:
        """
        Calculate the euclidean distance(s) between self and other.

        Note: ALIGNED pairing mode is supported only when self and other have same shape.

        :param other: The target point(s) for distance calculations
        :param pairing: An enum, specifies whether to calculate distances between
                        ALL (pairwise) or ALIGNED (corresponding points).
        :return: If pairing mode is ALIGNED:
                    a 1D numpy array of euclidean distance(s) between corresponding pairs of self and other.
                 Otherwise, if pairing mode is ALL:
                    a 2D numpy array of shape=(len(self), len(other)), of euclidean distance(s) between
                    all pairs of self and other.

        """
        if pairing is self.Pairing.ALIGNED or len(self) == 1 or len(other) == 1:
            dists = (self - other).norm
        else:
            dists = self._pairwise_diff(self, other).norm

        if pairing is self.Pairing.ALL:
            dists = dists.reshape(len(self), len(other))

        return dists

    def euclid_dist_squared(self, other: Point2D, *, pairing: Pairing = Pairing.ALL) -> np.ndarray:
        """
        Calculate the euclidean distance(s) squared between self and other.

        Note: ALIGNED pairing mode is supported only when self and other have same shape.

        :param other: The target point(s) for distance calculations
        :param pairing: An enum, specifies whether to calculate distances between
                        ALL (pairwise) or ALIGNED (corresponding points).
        :return: If pairing mode is ALIGNED:
                    a 1D numpy array of euclidean distance(s) squared between corresponding pairs of self and other.
                 Otherwise, if pairing mode is ALL:
                    a 2D numpy array of shape=(len(self), len(other)), of euclidean distance(s) squared between
                    all pairs of self and other.

        """
        if pairing is self.Pairing.ALIGNED or len(self) == 1 or len(other) == 1:
            dists = (self - other).norm_squared
        else:
            dists = self._pairwise_diff(self, other).norm_squared

        if pairing is self.Pairing.ALL:
            dists = dists.reshape(len(self), len(other))

        return dists
