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
        """
        Calculate a projection of itself onto another vector.

        :param onto: a direction vector to project itself onto.
        :return: the projected vector.
        """
        onto_unit = onto.normalized()
        return self._project_onto(self, onto_unit).view(Vector2D)

    def rotated(self, rotation_angle: Union[float, np.ndarray, Iterable[float]],
                rotation_units: Units = Units.RADIANS) -> Vector2D:
        """
        Calculates a rotated vector(s) by given rotation angle(s)
        """
        if rotation_units is Vector2D.Units.DEGREES:
            rotation_angle = np.deg2rad(rotation_angle)
        new_direction = self.direction + rotation_angle
        return Vector2D(magnitude=self.norm, direction=new_direction)

    @staticmethod
    @njit
    def _calc_angle_diff(direction_from: np.ndarray, direction_to: np.ndarray) -> np.ndarray:
        raw_diff = direction_to - direction_from
        diff = np.abs(raw_diff) % (2 * np.pi)
        diff[diff > np.pi] = 2 * np.pi - diff[diff > np.pi]

        mask = ((raw_diff >= -np.pi) & (raw_diff <= 0)) | ((raw_diff >= np.pi) & (raw_diff <= 2 * np.pi))
        diff[mask] *= -1
        return diff

    def angle_to(self, v_towards: Vector2D):
        """
        Returns the angle between the current vector(s) and v_towards.
        The angle is defined such that [(self.direction + angle) % 2*pi = v_towards.direction]
        """
        return self._calc_angle_diff(direction_from=self.direction, direction_to=v_towards.direction)

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
