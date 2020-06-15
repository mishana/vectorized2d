from __future__ import annotations

from typing import List, Sequence

import numpy as np
from numba import njit


class Array2D(np.ndarray):
    """
    This is a user-friendly interface to numpy arrays of shape=Nx2

    Examples:
    ---------

    Creation:
    >>> v1 = Array2D([1, 2])
    >>> v2 = Array2D([[1,2], [3,4]])
    >>> v3 = Array2D(np.random.random(size=(1000, 2)))
    >>> v4 = np.random.random(size=(1000, 2)).view(Array2D)

    >>> v1.shape
    (1, 2)
    >>> v2.shape
    (2, 2)
    >>> v3.shape
    (1000, 2)
    >>> v4.shape
    (1000, 2)
    """

    def __new__(cls, input_array) -> Array2D:
        # the underlying ndarray is always of shape Nx2
        return np.asarray(input_array).reshape(-1, 2).view(cls)

    @classmethod
    def concat(cls, arrays: Sequence[Array2D]) -> Array2D:
        """
        Concatenates a sequence of Array2D objects - vertically.
        That is, if v1 is of shape (N1x2) and v2 is of shape (N2x2)
        then Array2D.concat([v1, v2]) is of shape ((N1+N2)x2).

        :param arrays: a sequence of Array2D objects to concatenate
        :return: a Array2D object that holds all the 2D-arrays in arrays, stacked up vertically - (N_totalx2) shape
        """
        return np.concatenate(arrays).view(cls)

    def __hash__(self):
        return hash(self.tobytes())

    def __eq__(self, other):
        return np.array_equal(self, other)

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, item) -> Array2D:
        if isinstance(item, (int, np.integer)):
            return super().__getitem__(item)[np.newaxis]  # expand dimension, make sure the Coordinate shape is (1x2)
        return super().__getitem__(item)

    @property
    def x1(self):
        """
        This property holds the first axis value(s) of the 2D vector(s)
        :return: a 1D numpy array of values along the first axis
        """
        return self.view(np.ndarray)[:, 0]

    @property
    def x2(self):
        """
        This property holds the second axis value(s) of the 2D vector(s)
        :return: a 1D numpy array of values along the second axis
        """
        return self.view(np.ndarray)[:, 1]

    def split(self) -> List[Array2D]:
        """
        Split a (Nx2) Array2D object into a list of N (1x2) Array2D objects.

        :return: a list of (1x2) array2d objects
        """
        return [self[i:(i + 1)] for i in range(len(self))]

    @staticmethod
    @njit
    def _norm(a: Array2D) -> np.ndarray:
        return np.sqrt(a[:, 0] ** 2 + a[:, 1] ** 2)

    @property
    def norm(self) -> np.ndarray:
        # TODO: add a conditional parallel jit for larger arrays (~len > 500_000)
        return self._norm(self)

    @staticmethod
    @njit
    def _norm_squared(a: Array2D) -> np.ndarray:
        return a[:, 0] ** 2 + a[:, 1] ** 2

    @property
    def norm_squared(self) -> np.ndarray:
        return self._norm_squared(self)

    def normalized(self) -> Array2D:
        norm = self.norm[:, np.newaxis]
        norm[norm == 0] = 1
        return self / norm
