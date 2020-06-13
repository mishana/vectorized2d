from vectorized2d import Array2D
import numpy as np


def test_construct_single_2d_array_from_array():
    a = np.random.random(size=(2,))
    a2d = Array2D(a)

    assert np.array_equal(a2d, a.reshape(-1, 2))


def test_construct_single_2d_array_from_list():
    a = np.random.random(size=(2,))
    a2d = Array2D(a.tolist())

    assert np.array_equal(a2d, a.reshape(-1, 2))


def test_construct_multi_2d_array_from_array():
    a = np.random.random(size=(1000, 2))
    a2d = Array2D(a)

    assert np.array_equal(a2d, a.reshape(-1, 2))


def test_construct_multi_2d_array_from_list():
    a = np.random.random(size=(2000, 2))
    a2d = Array2D(a.tolist())

    assert np.array_equal(a2d, a.reshape(-1, 2))


def test_x1_scalar_for_single_2d_array():
    a = np.random.random(size=(2,))
    a2d = Array2D(a)

    assert a2d.x1 == a[0]


def test_x1_array_for_multi_2d_array():
    a = np.random.random(size=(1000, 2))
    a2d = Array2D(a)

    assert np.array_equal(a2d.x1, a[:, 0])


def test_x2_scalar_for_single_2d_array():
    a = np.random.random(size=(2,))
    a2d = Array2D(a)

    assert a2d.x2 == a[1]


def test_x2_array_for_multi_2d_array():
    a = np.random.random(size=(2000, 2))
    a2d = Array2D(a)

    assert np.array_equal(a2d.x2, a[:, 1])


def test_norm():
    a = np.random.random(size=(3000, 2))
    a2d = Array2D(a)

    assert np.array_equal(a2d.norm[:, np.newaxis], np.linalg.norm(a, axis=1, keepdims=True))
    assert np.array_equal(a2d.norm, np.linalg.norm(a, axis=1))


def test_normalize():
    a = np.random.random(size=(4000, 2))
    a2d = Array2D(a)
    n = a2d.normalized()
    norms = np.linalg.norm(n, axis=1)

    assert np.allclose(norms, 1)
