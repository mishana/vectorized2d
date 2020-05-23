import numpy as np

from vectorized2d.point2d import Point2D


def test_euclidean_distance_same_shape():
    a1 = np.random.random(size=(10000, 2))
    a2 = np.random.random(size=(10000, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1 - a2, axis=1, keepdims=True).ravel())
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1 - a2, axis=1))


def test_euclidean_distance_broadcast():
    a1 = np.random.random(size=(10000, 2))
    a2 = np.random.random(size=(1, 2))
    p1 = Point2D(a1)
    p2 = Point2D(a2)

    assert p1.euclid_dist(p2).shape == (p1.shape[0],)
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1 - a2, axis=1, keepdims=True).ravel())
    assert np.array_equal(p1.euclid_dist(p2), np.linalg.norm(a1 - a2, axis=1))
