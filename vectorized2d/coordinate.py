from __future__ import annotations

import math
from typing import Tuple, Iterable, Union

import numpy as np
from fast_enum import FastEnum
from numba import njit

from vectorized2d import Point2D
from vectorized2d.utils import units as units


class Coordinate(Point2D):
    """"
    This is a user-friendly wrapper for arrays of 2D vectors that represent 2D spatial coordinates
    (longitude and latitude) in radians.
    """

    class Units(metaclass=FastEnum):
        RADIANS = 0
        DEGREES = 1

    def __new__(cls,
                *,  # make lat, lon and units keyword-only arguments
                lat: Union[float, np.ndarray, Iterable[float]],
                lon: Union[float, np.ndarray, Iterable[float]],
                units: Units = Units.RADIANS) -> Coordinate:
        """

        :param lat: latitude(s) of a coordinate(s).
        :param lon: longitude(s) of a coordinate(s).
        :param units: an enum, specifies whether the input lan/lon is given in radians or degrees.

        Examples
        --------
        >>> c1 = Coordinate(lat=45, lon=180, units=Coordinate.Units.DEGREES)
        >>> c1
        Coordinate([[0.78539816, 3.14159265]])

        >>> c2 = Coordinate(lat=[1,2,3], lon=[4,5,6])
        >>> c2
        Coordinate([[1, 4],
                    [2, 5],
                    [3, 6]])

        """
        input_array = np.array([lat, lon]).T
        if units is Coordinate.Units.DEGREES:
            input_array = np.deg2rad(input_array)

        return super().__new__(cls, input_array=input_array)

    @property
    def lat(self):
        """
        This property holds the latitude value(s) of the Coordinate(s)
        :return: a 1D numpy array of latitude values
        """
        return self.x1

    @property
    def lon(self):
        """
        This property holds the longitude value(s) of the Coordinate(s)
        :return: a 1D numpy array of longitude values
        """
        return self.x2

    @staticmethod
    @njit
    def _delta_east_and_north_jit(self_lat: np.ndarray, self_lon: np.ndarray, other_lat: np.ndarray,
                                  other_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_lat = np.rad2deg(other_lat - self_lat)
        d_lon = np.rad2deg(other_lon - self_lon)
        d_north = d_lat * 60
        d_east = d_lon * 60 * np.cos((self_lat + other_lat) / 2)

        return d_east * units.NM_TO_METERS, d_north * units.NM_TO_METERS

    def _delta_east_and_north(self, other: Coordinate) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates an approximation of the delta between self and other on the east and north axes, respectively.

        Note: Supports coordinates (self, other) with matching sizes,
        and one-to-many or many-to-one using standard broadcasting.

        :param other: the target coordinate(s) for delta calculations
        :return: a Tuple of two 1D numpy arrays of the delta on the east and north axes, respectively [meters]
        """
        return self._delta_east_and_north_jit(self.lat, self.lon, other.lat, other.lon)

    @staticmethod
    @njit
    def _dist(d_east: np.ndarray, d_north: np.ndarray) -> np.ndarray:
        return np.sqrt(d_east ** 2 + d_north ** 2)

    def geo_dist(self, other: Coordinate) -> np.ndarray:
        """
        Calculates an approximation of the geographical distance(s) between self and other.

        Note: Supports coordinates (self, other) with matching sizes,
        and one-to-many or many-to-one using standard broadcasting.

        :param other: the target coordinate(s) for distance calculations
        :return: a 1D numpy array of geographical distance(s) between self and other [meters]
        """
        d_east, d_north = self._delta_east_and_north(other)
        return self._dist(d_north, d_east)

    @staticmethod
    @njit
    def _dist_squared(d_east: np.ndarray, d_north: np.ndarray) -> np.ndarray:
        return d_east ** 2 + d_north ** 2

    def geo_dist_squared(self, other: Coordinate) -> np.ndarray:
        """
        Calculates an approximation the geographical distance(s) squared between self and other.

        Note: Supports coordinates (self, other) with matching sizes,
        and one-to-many or many-to-one using standard broadcasting.

        :param other: the target coordinate(s) for distance calculations
        :return: a 1D numpy array of geographical distance(s) squared between self and other [meters**2]
        """
        d_east, d_north = self._delta_east_and_north(other)
        return d_north * d_north + d_east * d_east

    @staticmethod
    @njit
    def _bearing(d_east: np.ndarray, d_north: np.ndarray) -> np.ndarray:
        return np.arctan2(d_east, d_north) % (2 * math.pi)

    def bearing(self, other: Coordinate) -> np.ndarray:
        """
        Calculates an approximation of the bearing(s) between self and other.

        Note: Supports coordinates (self, other) with matching sizes,
        and one-to-many or many-to-one using standard broadcasting.

        :param other: the target coordinate(s) for bearing calculations
        :return: a 1D numpy array of bearing(s) between self and other [radians]
        """
        d_east, d_north = self._delta_east_and_north(other)
        return self._bearing(d_east, d_north)

    def geo_dist_and_bearing(self, other: Coordinate) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates an approximation of the geographical distance(s) and bearing(s) between self and other.

        Note: Supports coordinates (self, other) with matching sizes,
        and one-to-many or many-to-one using standard broadcasting.

        :param other: the target coordinate(s) for distance and bearing calculations
        :return: a Tuple of two 1D numpy arrays of geographical distance(s) and bearing(s)
                 between self and other ([meters], [radians])
        """
        d_east, d_north = self._delta_east_and_north(other)
        dist = self._dist(d_east, d_north)
        bearing = self._bearing(d_east, d_north)
        return dist, bearing

    @staticmethod
    @njit
    def _shifted(self_lat: np.ndarray, self_lon: np.ndarray, geo_dist: Union[float, np.ndarray],
                 bearing: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        earth_radius = 6_378_100
        angular_dist = geo_dist / earth_radius

        sin_angular_dist = np.sin(angular_dist)
        cos_angular_dist = np.cos(angular_dist)
        sin_lat = np.sin(self_lat)
        cos_lat = np.cos(self_lat)

        sin_shifted_lat = sin_lat * cos_angular_dist + cos_lat * sin_angular_dist * np.cos(bearing)

        shifted_lat = np.arcsin(sin_shifted_lat)
        shifted_lon = self_lon + np.arctan2(np.sin(bearing) * sin_angular_dist * cos_lat,
                                            cos_angular_dist - sin_lat * sin_shifted_lat)

        return shifted_lat, shifted_lon

    def shifted(self, geo_dist: Union[float, np.ndarray], bearing: Union[float, np.ndarray]) -> Coordinate:
        """
        Calculates a coordinate(s) shifted by given distance(s) and bearing(s).

        :param geo_dist: the distance(s) to the shifted coordinate(s) [meters]
        :param bearing: the bearing(s) to the shifted coordinate(s) [radians]
        :return: a Coordinate object that represents the coordinate(s) shifted by given distance(s) and bearing(s)
        """
        shifted_lat, shifted_lon = self._shifted(self.lat, self.lon, geo_dist, bearing)
        return Coordinate(lat=shifted_lat, lon=shifted_lon)

    def circle_around(self, radius: float, number_of_points: int) -> Coordinate:
        """
        Return a multi-coordinate with shape=(number_of_points, 2), representing a circle around the Coordinate (self).

        Note: the behaviour is only well-defined for a single Coordinate as the center of the circle.

        :param radius: radius of the circle [meters]
        :param number_of_points: amount of points to sample from the circle
        :return: a Coordinate with shape=(number_of_points, 2), that holds coordinates of samples from
                 the surrounding circle
        """
        assert self.shape[0] == 1, "circle_around() method is undefined for multi-coordinates"
        return self.shifted(geo_dist=radius, bearing=np.arange(0, math.pi * 2, (math.pi * 2) / number_of_points))
