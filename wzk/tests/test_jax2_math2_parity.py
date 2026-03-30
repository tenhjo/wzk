import numpy as np

from wzk import jax2
from wzk import math2


def test_normalize_and_discretize():
    x = np.arange(20).reshape(4, 5)
    x[-1, -1] = 20
    n0 = math2.normalize_01(x)
    n1 = jax2.normalize_01(x)
    assert np.allclose(np.asarray(n0), np.asarray(n1))

    y = np.linspace(0, 2, num=20)
    d0 = math2.discretize(y, step=0.17)
    d1 = jax2.discretize(y, step=0.17)
    assert np.allclose(np.asarray(d0), np.asarray(d1))


def test_divisors_and_mean_pair():
    for n in [1, 2, 10, 12, 24, 81, 99]:
        assert math2.divisors(n) == jax2.divisors(n)
        assert math2.get_mean_divisor_pair(n) == jax2.get_mean_divisor_pair(n)


def test_dxnorm_dx():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(128, 7))

    j0 = math2.dxnorm_dx(x)
    j1 = jax2.dxnorm_dx(x)
    assert np.allclose(np.asarray(j0), np.asarray(j1), atol=1e-6, rtol=1e-6)


def test_numeric_derivative():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(20, 5))

    def linalg_norm(q):
        return q / np.linalg.norm(q, axis=-1, keepdims=True)

    j0 = math2.numeric_derivative(fun=linalg_norm, x=x, axis=-1)
    j1 = jax2.numeric_derivative(fun=linalg_norm, x=x, axis=-1)
    assert np.allclose(np.asarray(j0), np.asarray(j1), atol=5e-5, rtol=5e-5)


def test_matrix_sqrt():
    A = np.array([[1, 2], [2, 4]], dtype=float)
    s0 = math2.matrix_sqrt(A)
    s1 = jax2.matrix_sqrt(A)
    assert np.allclose(np.asarray(s0), np.asarray(s1), atol=1e-6, rtol=1e-6)
