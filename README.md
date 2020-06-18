# Vectorized2D

![Tests with conda](https://github.com/mishana/vectorized2d/workflows/Tests%20with%20Conda/badge.svg?branch=master)
![Tests with pip](https://github.com/mishana/vectorized2d/workflows/Tests%20with%20pip/badge.svg?branch=master)

[![codecov](https://codecov.io/gh/mishana/vectorized2d/branch/master/graph/badge.svg)](https://codecov.io/gh/mishana/vectorized2d)

This is a user-friendly wrapper to numpy arrays, for dealing with numerical problems in a vectorized fashion - in the 2D world.

Provided objects include:
1. `Array2D` - a user-friendly interface to numpy arrays of shape=Nx2.
2. `Vector2D` - a user-friendly wrapper for arrays of 2D vectors that represent physical quantities.
3. `Point2D` - a user-friendly wrapper to arrays of 2D points that represent spatial locations in a cartesian coordinate system.
4. `Coordinate` - a user-friendly wrapper for arrays of 2D points that represent 2D spatial (geographical) coordinates
    (longitude and latitude) in radians.
    

## Installation
The easiest way to install [vectorized2d](https://pypi.org/project/vectorized2d/) and get updates is via the Python Package Index (PyPI):

```bash
$ pip install vectorized2d
```

## Performance 
Vectorized2d uses [Numba](http://numba.pydata.org/) to gain enhanced performance compared to vanilla numpy.

For example, (per-row) norm calculation:
```python
  >>> import numpy as np
  >>> from vectorized2d import Array2D
  
  >>> a_np = np.random.random(size=(1000, 2))
  >>> a_2d = a_np.view(Array2D)
  
  >>> %timeit np.linalg.norm(a_np, axis=1)
  23.1 µs ± 1.25 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
  
  >>> %timeit np.sqrt(np.einsum('ij,ij->i', a_np, a_np))
  10.5 µs ± 40.2 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
  
  >>> %timeit a_2d.norm
  2.63 µs ± 67.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```
