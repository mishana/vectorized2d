from random import random, randint

import numpy as np

from vectorized2d.array2d import Array2D
from vectorized2d.vector2d import Vector2d


def _rand_degree():
    return random() * randint(0, 360)


def test_create_single_vector_with_magnitude_and_direction():
    magnitude = random()
    direction = np.deg2rad(_rand_degree())
    c = Vector2d(magnitude=magnitude, direction=direction)

    assert np.array_equal(c, Array2D([magnitude * np.cos(direction), magnitude * np.sin(direction)]))


def test_create_multi_vector_with_vectors():
    m1, d1 = random(), np.deg2rad(_rand_degree())
    m2, d2 = random(), np.deg2rad(_rand_degree())
    m3, d3 = random(), np.deg2rad(_rand_degree())

    v1 = Vector2d(magnitude=m1, direction=d1)
    v2 = Vector2d(magnitude=m2, direction=d2)
    v3 = Vector2d(magnitude=m3, direction=d3)
    v = Vector2d.concat([v1, v2, v3])

    assert np.array_equal(v, Array2D([[m1 * np.cos(d1), m1 * np.sin(d1)],
                                      [m2 * np.cos(d2), m2 * np.sin(d2)],
                                      [m3 * np.cos(d3), m3 * np.sin(d3)]]))


def test_create_single_vector_with_direction_degrees():
    magnitude = random()
    direction = _rand_degree()
    c = Vector2d(magnitude=magnitude, direction=direction, direction_units=Vector2d.Units.DEGREES)

    assert np.array_equal(c, Array2D([magnitude * np.cos(np.deg2rad(direction)),
                                      magnitude * np.sin(np.deg2rad(direction))]))


def test_create_multi_vector_with_vectors_degrees():
    m1, d1 = random(), _rand_degree()
    m2, d2 = random(), _rand_degree()
    m3, d3 = random(), _rand_degree()

    v1 = Vector2d(magnitude=m1, direction=d1, direction_units=Vector2d.Units.DEGREES)
    v2 = Vector2d(magnitude=m2, direction=d2, direction_units=Vector2d.Units.DEGREES)
    v3 = Vector2d(magnitude=m3, direction=d3, direction_units=Vector2d.Units.DEGREES)
    v = Vector2d.concat([v1, v2, v3])

    assert np.array_equal(v, Array2D([[m1 * np.cos(np.deg2rad(d1)), m1 * np.sin(np.deg2rad(d1))],
                                      [m2 * np.cos(np.deg2rad(d2)), m2 * np.sin(np.deg2rad(d2))],
                                      [m3 * np.cos(np.deg2rad(d3)), m3 * np.sin(np.deg2rad(d3))]]))


def test_create_multi_vector_with_single_mag_multi_dir():
    m = random()
    d1, d2, d3 = _rand_degree(), _rand_degree(), _rand_degree()

    v = Vector2d(magnitude=m, direction=[d1, d2, d3], direction_units=Vector2d.Units.DEGREES)

    np.array_equal(v, Array2D([[m * np.cos(np.deg2rad(d1)), m * np.sin(np.deg2rad(d1))],
                               [m * np.cos(np.deg2rad(d2)), m * np.sin(np.deg2rad(d2))],
                               [m * np.cos(np.deg2rad(d3)), m * np.sin(np.deg2rad(d3))]]))


def test_create_multi_vector_with_multi_mag_single_dir():
    m1, m2, m3 = random(), random(), random()
    d = np.deg2rad(_rand_degree())

    v = Vector2d(magnitude=[m1, m2, m3], direction=d)
    assert np.array_equal(v, Array2D([[m1 * np.cos(d), m1 * np.sin(d)],
                                      [m2 * np.cos(d), m2 * np.sin(d)],
                                      [m3 * np.cos(d), m3 * np.sin(d)]]))


def test_project_v1_onto_v2():
    a = np.random.random(size=(5000, 2))
    b = np.random.random(size=(1, 2))
    v1 = Array2D(a).view(Vector2d)
    v2 = Array2D(b).view(Vector2d)

    v_1proj2 = v1.project_onto(v2)
    a_proj_b = b * np.dot(a, b.T) / np.dot(b, b.T)  # using formula from wikipedia, which is less efficient empirically

    assert np.allclose(v_1proj2, a_proj_b)


def test_direction():
    direction = np.random.random(size=(5000,)) * randint(0, 360)
    magnitude = np.random.random(size=(5000,)) * randint(0, 20)
    v = Vector2d(magnitude=magnitude, direction=direction, direction_units=Vector2d.Units.DEGREES)

    assert np.allclose(v.direction(units=Vector2d.Units.DEGREES) % 360, direction)