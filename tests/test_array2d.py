import random

import numpy as np

from vectorized2d import Array2D


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


def test_concat():
    a1 = np.random.random(size=(1992, 2)).view(Array2D)
    a2 = np.random.random(size=(817, 2)).view(Array2D)

    a12 = Array2D.concat((a1, a2))
    a21 = Array2D.concat((a2, a1))

    assert a12.shape[1] == 2
    assert a21.shape[1] == 2
    assert a12.shape[0] == a1.shape[0] + a2.shape[0]
    assert a21.shape[0] == a1.shape[0] + a2.shape[0]
    assert a12[:a1.shape[0]] == a1
    assert a12[a1.shape[0]:] == a2
    assert a21[:a2.shape[0]] == a2
    assert a21[a2.shape[0]:] == a1


def test_repeat_scalar_repeats():
    a = np.random.random(size=(random.randint(1, 1000), 2)).view(Array2D)
    repeats = random.randint(1, 100)
    ar = a.repeat(repeats)

    assert ar.shape[1] == 2
    assert ar.shape[0] == a.shape[0] * repeats

    for i in range(len(a)):
        assert np.all(a[i].view(np.ndarray) == ar[(i * repeats):((i + 1) * repeats)].view(np.ndarray))


def test_repeat_array_repeats():
    a = np.random.random(size=(random.randint(1, 1000), 2)).view(Array2D)
    repeats = np.random.random_integers(low=1, high=100, size=a.shape[0])
    ar = a.repeat(repeats)

    assert ar.shape[1] == 2
    assert ar.shape[0] == sum(repeats)

    for i in range(len(a)):
        assert np.all((a[i] == ar_i for ar_i in ar[(i * sum(repeats[:i])):((i + 1) * repeats[i])]))


def test_reduce_ufuncs_same_as_ndarray():
    a_np = np.random.random(size=(random.randint(1, 1000), 2))
    a_2d = Array2D(a_np)

    assert a_np.min() == a_2d.min() and np.min(a_2d) == a_2d.min()
    assert a_np.max() == a_2d.max() and np.max(a_2d) == a_2d.max()
    assert a_np.sum() == a_2d.sum() and np.sum(a_2d) == a_2d.sum()
    assert a_np.prod() == a_2d.prod() and np.prod(a_2d) == a_2d.prod()
    assert a_np.mean() == a_2d.mean() and np.mean(a_2d) == a_2d.mean()
    assert a_np.std() == a_2d.std() and np.std(a_2d) == a_2d.std()
    assert a_np.var() == a_2d.var() and np.var(a_2d) == a_2d.var()
