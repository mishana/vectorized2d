from random import randint

import numpy as np

from vectorized2d import Point2D


def test_euclidean_distance_same_shape_aligned():
    a1 = np.random.random(size=(100, 2))
    a2 = np.random.random(size=(100, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert p1.euclid_dist(p2, pairing=Point2D.Pairing.ALIGNED).shape == (len(p1),) == (len(p2),)
    assert np.array_equal(p1.euclid_dist(p2, pairing=Point2D.Pairing.ALIGNED),
                          np.linalg.norm(a1 - a2, axis=1, keepdims=True).ravel())
    assert np.array_equal(p1.euclid_dist(p2, pairing=Point2D.Pairing.ALIGNED), np.linalg.norm(a1 - a2, axis=1))


def test_euclidean_distance_same_shape_pairwise():
    a1 = np.random.random(size=(100, 2))
    a2 = np.random.random(size=(100, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    a1_repeated = p1.repeat(len(p2)).view(np.ndarray)
    a2_tiled = p2.tile(len(p1)).view(np.ndarray)

    assert a1_repeated.shape == a2_tiled.shape == (len(p1) * len(p2), 2)

    assert p1.euclid_dist(p2).shape == (len(p1), len(p2))
    assert np.array_equal(p1.euclid_dist(p2),
                          np.linalg.norm(a1_repeated - a2_tiled, axis=1, keepdims=True).reshape(len(p1), len(p2)))
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1_repeated - a2_tiled, axis=1).reshape(len(p1), len(p2)))


def test_euclidean_distance_different_shapes():
    a1 = np.random.random(size=(100, 2))
    a2 = np.random.random(size=(50, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    a1_repeated = p1.repeat(len(p2)).view(np.ndarray)
    a2_tiled = p2.tile(len(p1))

    assert a1_repeated.shape == a2_tiled.shape == (len(p1) * len(p2), 2)

    assert p1.euclid_dist(p2).shape == (len(p1), len(p2))
    assert np.array_equal(p1.euclid_dist(p2),
                          np.linalg.norm(a1_repeated - a2_tiled, axis=1, keepdims=True).reshape(len(p1), len(p2)))
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1_repeated - a2_tiled, axis=1).reshape(len(p1), len(p2)))


def test_euclidean_distance_broadcast():
    a1 = np.random.random(size=(10000, 2))
    a2 = np.random.random(size=(1, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert p1.euclid_dist(p2).shape == (len(p1), len(p2))
    assert p2.euclid_dist(p1).shape == (len(p2), len(p1))
    assert np.array_equal(p1.euclid_dist(p2),
                          np.linalg.norm(a1 - a2, axis=1, keepdims=True).reshape((len(p1), len(p2))))
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1 - a2, axis=1).reshape((len(p1), len(p2))))


def test_euclidean_distance_squared():
    a1 = np.random.random(size=(randint(1, 100), 2))
    a2 = np.random.random(size=(randint(1, 100), 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert np.allclose(p1.euclid_dist(p2) ** 2, p1.euclid_dist_squared(p2))
    assert np.allclose(p2.euclid_dist(p1) ** 2, p2.euclid_dist_squared(p1))


def test_euclidean_distance_squared_aligned():
    a1 = np.random.random(size=(100, 2))
    a2 = np.random.random(size=(100, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert np.allclose(p1.euclid_dist(p2, pairing=Point2D.Pairing.ALIGNED) ** 2,
                       p1.euclid_dist_squared(p2, pairing=Point2D.Pairing.ALIGNED))
    assert np.allclose(p2.euclid_dist(p1, pairing=Point2D.Pairing.ALIGNED) ** 2,
                       p2.euclid_dist_squared(p1, pairing=Point2D.Pairing.ALIGNED))
