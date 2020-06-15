from __future__ import annotations

from typing import Iterable, Union

import numpy as np
from fast_enum import FastEnum
from numba import njit

from vectorized2d import Array2D


class Vector2D(Array2D):
    """"
        This is a user-friendly wrapper for arrays of 2D vectors that represent physical quantities.
    """

    class Units(metaclass=FastEnum):
        RADIANS = 0
        DEGREES = 1

    def __new__(cls,
                *,  # make magnitude and direction keyword-only arguments
                magnitude: Union[float, np.ndarray, Iterable[float]],
                direction: Union[float, np.ndarray, Iterable[float]],
                direction_units: Units = Units.RADIANS) -> Vector2D:
        """

        :param magnitude: magnitude(s) of a physical quantity vector(s).
        :param direction: direction(s) of a physical quantity vector(s).
        :param direction_units: an enum, specifies whether the input direction is given in radians or degrees.

        Examples
        --------

        >>> v1 = Vector2D(magnitude=1, direction=np.pi/2)
        >>> v1
        Vector2D([[6.123234e-17, 1.000000e+00]])

        >>> v2 = Vector2D(magnitude=2, direction=np.pi/2)
        >>> v2
        Vector2D([[1.2246468e-16, 2.0000000e+00]])

        >>> v3 = Vector2D(magnitude=1, direction=90, direction_units=Vector2D.Units.DEGREES)
        >>> v3
        Vector2D([[6.123234e-17, 1.000000e+00]])

        >>> v4 = Vector2D(magnitude=2, direction=[0, 90, 180], direction_units=Vector2D.Units.DEGREES)
        >>> v4
        Vector2D([[ 2.0000000e+00,  0.0000000e+00],
                  [ 1.2246468e-16,  2.0000000e+00],
                  [-2.0000000e+00,  2.4492936e-16]])

        >>> v5 = Vector2D(magnitude=[1, 2, 3], direction=90, direction_units=Vector2D.Units.DEGREES)
        >>> v5
        Vector2D([[6.1232340e-17, 1.0000000e+00],
                  [1.2246468e-16, 2.0000000e+00],
                  [1.8369702e-16, 3.0000000e+00]])

        """
        if direction_units is Vector2D.Units.DEGREES:
            direction = np.deg2rad(direction)
        input_array = np.array([np.asarray(magnitude) * np.cos(direction),
                                np.asarray(magnitude) * np.sin(direction)]).T

        return super().__new__(cls, input_array=input_array)

    @staticmethod
    @njit
    def _project_onto(v: Vector2D, onto_unit: Vector2D) -> np.ndarray:
        projection_magnitude = (v[:, 0] * onto_unit[:, 0] + v[:, 1] * onto_unit[:, 1]).reshape(-1, 1)
        return projection_magnitude * onto_unit

    def project_onto(self, onto: Vector2D) -> Vector2D:
        onto_unit = onto.normalized()
        return self._project_onto(self, onto_unit).view(Vector2D)

    @staticmethod
    @njit
    def _direction(v: Vector2D) -> np.ndarray:
        return np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)

    @property
    def direction(self) -> np.ndarray:
        """
        Returns the (positive - between 0 and 2*pi) direction of the vector(s) in radians.

        """
        # TODO: add a conditional parallel (and fastmath) jit for larger arrays (~len > 5_000)
        return self._direction(self)
