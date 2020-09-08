from random import random, randint

import math
import numpy as np
import pytest

from vectorized2d.utils import units as units
from vectorized2d import Array2D, Coordinate


def _rand_degree():
    return random() * randint(0, 360)


def test_create_single_coordinate_with_lat_lon():
    lat = _rand_degree()
    lon = _rand_degree()
    c = Coordinate(lat=lat, lon=lon)

    assert np.array_equal(c, Array2D([lat, lon]))


def test_create_multi_coordinate_with_coordinates():
    lat1, lon1 = _rand_degree(), _rand_degree()
    lat2, lon2 = _rand_degree(), _rand_degree()
    lat3, lon3 = _rand_degree(), _rand_degree()

    c1 = Coordinate(lat=lat1, lon=lon1)
    c2 = Coordinate(lat=lat2, lon=lon2)
    c3 = Coordinate(lat=lat3, lon=lon3)
    c = Coordinate.concat([c1, c2, c3])

    assert np.array_equal(c, Array2D([[lat1, lon1],
                                      [lat2, lon2],
                                      [lat3, lon3]]))


def test_create_single_coordinate_with_lat_lon_degrees():
    lat = _rand_degree()
    lon = _rand_degree()
    c = Coordinate(lat=lat, lon=lon, units=Coordinate.Units.DEGREES)

    assert np.array_equal(c, np.deg2rad(Array2D([lat, lon])))


def test_create_multi_coordinate_with_coordinates_degrees():
    lat1, lon1 = _rand_degree(), _rand_degree()
    lat2, lon2 = _rand_degree(), _rand_degree()
    lat3, lon3 = _rand_degree(), _rand_degree()

    c1 = Coordinate(lat=lat1, lon=lon1, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=lat2, lon=lon2, units=Coordinate.Units.DEGREES)
    c3 = Coordinate(lat=lat3, lon=lon3, units=Coordinate.Units.DEGREES)
    c = Coordinate.concat([c1, c2, c3])

    assert np.array_equal(c, np.deg2rad(Array2D([[lat1, lon1],
                                                 [lat2, lon2],
                                                 [lat3, lon3]])))


def test_lat_lon_scalars_for_single_coordinate():
    lat, lon = _rand_degree(), _rand_degree()
    c = Coordinate(lat=lat, lon=lon)

    assert c.lat == lat
    assert c.lon == lon


def test_lat_lon_arrays_for_multi_coordinate():
    lat1, lon1 = _rand_degree(), _rand_degree()
    lat2, lon2 = _rand_degree(), _rand_degree()
    lat3, lon3 = _rand_degree(), _rand_degree()

    c1 = Coordinate(lat=lat1, lon=lon1)
    c2 = Coordinate(lat=lat2, lon=lon2)
    c3 = Coordinate(lat=lat3, lon=lon3)
    c = Coordinate.concat([c1, c2, c3])

    assert np.array_equal(c.lat, np.array([lat1, lat2, lat3]))
    assert np.array_equal(c.lon, np.array([lon1, lon2, lon3]))


def test_geo_distance_same_shape():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])
    c4 = Coordinate.concat([c2, c2, c1, c1])

    dists_c3_c4 = c3.geo_dist(c4)
    dists_c3_c3 = c3.geo_dist(c3)

    assert np.all(dists_c3_c4 == dists_c3_c4[0])
    assert np.all(dists_c3_c3 == 0.0)
    assert np.allclose(dists_c3_c4[:2], 72_497.1, rtol=0.01)
    assert np.allclose(dists_c3_c4[2:], 72_497.1, rtol=0.01)


def test_geo_distance_squared_same_shape():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])
    c4 = Coordinate.concat([c2, c2, c1, c1])

    dists_c3_c4 = c3.geo_dist_squared(c4)
    dists_c3_c3 = c3.geo_dist_squared(c3)

    assert np.all(dists_c3_c4 == dists_c3_c4[0])
    assert np.all(dists_c3_c3 == 0.0)
    assert np.allclose(dists_c3_c4[:2], 72_497.1**2, rtol=0.01)
    assert np.allclose(dists_c3_c4[2:], 72_497.1**2, rtol=0.01)


def test_geo_distance_broadcast():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])

    dists_c3_c1 = c3.geo_dist(c1)
    dists_c1_c3 = c1.geo_dist(c3)

    assert np.all(dists_c3_c1 == dists_c1_c3)
    assert np.all(dists_c1_c3[:2] == 0.0)
    assert not np.any(dists_c1_c3[2:] == 0.0)
    assert np.allclose(dists_c3_c1[2], 72_497.1, rtol=0.01)
    assert np.allclose(dists_c1_c3[2:], 72_497.1, rtol=0.01)


def test_bearing_same_shape():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])
    c4 = Coordinate.concat([c2, c2, c1, c1])

    bearing_c3_c4 = c3.bearing(c4)
    bearing_c3_c3 = c3.bearing(c3)

    assert np.all(bearing_c3_c3 == 0.0)
    assert np.all(bearing_c3_c4[0] == bearing_c3_c4[1])
    assert np.all(bearing_c3_c4[2] == bearing_c3_c4[3])
    assert np.abs(np.rad2deg(bearing_c3_c4[0] - bearing_c3_c4[2])) == 180.0
    assert np.allclose(np.rad2deg(bearing_c3_c4)[:2], 39.91, rtol=0.001)
    assert np.allclose(np.rad2deg(bearing_c3_c4)[2:], 219.91, rtol=0.001)


def test_bearing_broadcast():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])

    bearing_c3_c1 = c3.bearing(c1)
    bearing_c1_c3 = c1.bearing(c3)

    assert np.all(bearing_c1_c3[:2] == 0)
    assert np.all(bearing_c3_c1[:2] == 0)
    assert np.abs(np.rad2deg(bearing_c1_c3[2] - bearing_c3_c1[2])) == 180.0
    assert np.allclose(np.rad2deg(bearing_c1_c3)[2:], 39.91, rtol=0.001)
    assert np.allclose(np.rad2deg(bearing_c3_c1)[2:], 219.91, rtol=0.001)


def test_geo_distance_and_bearing_same_shape():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])
    c4 = Coordinate.concat([c2, c2, c1, c1])

    dists_c3_c4, bearing_c3_c4 = c3.geo_dist_and_bearing(c4)
    dists_c3_c3, bearing_c3_c3 = c3.geo_dist_and_bearing(c3)

    assert np.all(dists_c3_c4 == dists_c3_c4[0])
    assert np.all(dists_c3_c3 == 0.0)
    assert np.allclose(dists_c3_c4[:2], 72_497.1, rtol=0.01)
    assert np.allclose(dists_c3_c4[2:], 72_497.1, rtol=0.01)

    assert np.all(bearing_c3_c3 == 0.0)
    assert np.all(bearing_c3_c4[0] == bearing_c3_c4[1])
    assert np.all(bearing_c3_c4[2] == bearing_c3_c4[3])
    assert np.abs(np.rad2deg(bearing_c3_c4[0] - bearing_c3_c4[2])) == 180.0
    assert np.allclose(np.rad2deg(bearing_c3_c4)[:2], 39.91, rtol=0.001)
    assert np.allclose(np.rad2deg(bearing_c3_c4)[2:], 219.91, rtol=0.001)


def test_geo_distance_and_bearing_broadcast():
    c1 = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    c2 = Coordinate(lat=33.5, lon=34.5, units=Coordinate.Units.DEGREES)
    c3 = Coordinate.concat([c1, c1, c2, c2])

    dists_c3_c1, bearing_c3_c1 = c3.geo_dist_and_bearing(c1)
    dists_c1_c3, bearing_c1_c3 = c1.geo_dist_and_bearing(c3)

    assert np.all(dists_c3_c1 == dists_c1_c3)
    assert np.all(dists_c1_c3[:2] == 0.0)
    assert not np.any(dists_c1_c3[2:] == 0.0)

    assert np.all(bearing_c1_c3[:2] == 0)
    assert np.all(bearing_c3_c1[:2] == 0)
    assert np.abs(np.rad2deg(bearing_c1_c3[2] - bearing_c3_c1[2])) == 180.0


def test_shifted_one_to_many():
    c = Coordinate(lat=33, lon=34, units=Coordinate.Units.DEGREES)
    dist = np.random.random(size=(1000,)) * 60 * units.NM_TO_METERS
    bearing = np.deg2rad(np.random.random(size=(1000,)) * randint(0, 360))

    shifted = c.shifted(geo_dist=dist, bearing=bearing)

    assert np.allclose(c.geo_dist(shifted), dist, rtol=0.01)
    assert np.allclose(c.bearing(shifted), bearing, rtol=0.01)


def test_shifted_many_to_one():
    c = Coordinate.concat(
        [Coordinate(lat=33 + i, lon=34 - i, units=Coordinate.Units.DEGREES) for i in np.random.random(size=(1000,))])
    dist = random() * 60 * units.NM_TO_METERS
    bearing = np.deg2rad(_rand_degree())

    shifted = c.shifted(geo_dist=dist, bearing=bearing)

    assert np.allclose(c.geo_dist(shifted), dist, rtol=0.01)
    assert np.allclose(c.bearing(shifted), bearing, rtol=0.01)


def test_shifted_same_shape():
    c = Coordinate.concat(
        [Coordinate(lat=33 + i, lon=34 - i, units=Coordinate.Units.DEGREES) for i in np.random.random(size=(1000,))])
    dist = np.random.random(size=(1000,)) * 60 * units.NM_TO_METERS
    bearing = np.deg2rad(np.random.random(size=(1000,)) * randint(0, 360))

    shifted = c.shifted(geo_dist=dist, bearing=bearing)

    assert np.allclose(c.geo_dist(shifted), dist, rtol=0.01)
    assert np.allclose(c.bearing(shifted), bearing, rtol=0.01)


def test_circle_around_fails_for_multi_coordinate():
    c = Coordinate(lat=np.deg2rad(np.random.random(size=(1000,)) * randint(0, 360)),
                   lon=np.deg2rad(np.random.random(size=(1000,)) * randint(0, 360)),
                   units=Coordinate.Units.DEGREES)
    random_indices = np.random.choice(c.shape[0], randint(2, 1000))
    c_rand = c[random_indices]

    with pytest.raises(AssertionError):
        c_rand.circle_around(radius=random(), number_of_points=randint(1, 1000))


def test_circle_around():
    c = Coordinate(lat=33 + random(), lon=34 - random(), units=Coordinate.Units.DEGREES)
    radius = random() * 60 * units.NM_TO_METERS
    number_of_points = randint(1, 1000)

    circle = c.circle_around(radius=radius, number_of_points=number_of_points)
    dists, bearings = c.geo_dist_and_bearing(circle)

    assert np.allclose(dists, radius, rtol=0.01)
    assert np.allclose(bearings, np.arange(0, math.pi * 2, (math.pi * 2) / number_of_points), rtol=0.01, atol=1e-3)
    assert len(circle) == number_of_points


def test_ellipse_around():
    c = Coordinate(lat=33 + random(), lon=34 - random(), units=Coordinate.Units.DEGREES)
    minor_radius = random() * 60 * units.NM_TO_METERS
    major_radius = random() * 60 * units.NM_TO_METERS + minor_radius
    major_axis_bearing = np.deg2rad(_rand_degree())
    number_of_points = randint(1, 1000)

    ellipse = c.ellipse_around(major_radius=major_radius, minor_radius=minor_radius,
                               major_axis_bearing=major_axis_bearing, number_of_points=number_of_points)
    dists, bearings = c.geo_dist_and_bearing(ellipse)
    atol = 1e-8
    rtol = 0.01
    lower_bound = minor_radius - atol - minor_radius * rtol
    upper_bound = major_radius + atol + major_radius * rtol

    assert np.all((dists >= lower_bound) & (dists <= upper_bound))
    assert np.allclose(bearings, np.arange(0, math.pi * 2, (math.pi * 2) / number_of_points), rtol=0.01, atol=1e-3)
    assert len(ellipse) == number_of_points
