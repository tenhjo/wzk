import numpy as np

from wzk import jax2
from wzk import random2


def test_p_normal_skew():
    x = np.linspace(-3.0, 3.0, num=201)
    p0 = random2.p_normal_skew(x=x, loc=0.1, scale=1.4, a=0.8)
    p1 = jax2.p_normal_skew(x=x, loc=0.1, scale=1.4, a=0.8)
    assert np.allclose(np.asarray(p0), np.asarray(p1), atol=1e-6, rtol=1e-6)


def test_normal_skew_int_and_random_uniform_ndim():
    np.random.seed(0)
    s0 = random2.normal_skew_int(loc=3, scale=2.0, a=0.3, size=64)
    np.random.seed(0)
    s1 = jax2.normal_skew_int(loc=3, scale=2.0, a=0.3, size=64)
    assert np.array_equal(np.asarray(s0), np.asarray(s1))

    low = np.array([-1.0, 0.0, 10.0])
    high = np.array([+1.0, 2.0, 20.0])
    np.random.seed(1)
    u0 = random2.random_uniform_ndim(low=low, high=high, shape=50)
    np.random.seed(1)
    u1 = jax2.random_uniform_ndim(low=low, high=high, shape=50)
    assert np.allclose(np.asarray(u0), np.asarray(u1), atol=1e-6, rtol=1e-6)


def test_noise_modes():
    for mode in ("constant", "plusminus", "uniform", "normal"):
        np.random.seed(2)
        n0 = random2.noise(shape=(40, 3), scale=0.5, mode=mode)
        np.random.seed(2)
        n1 = jax2.noise(shape=(40, 3), scale=0.5, mode=mode)
        assert np.allclose(np.asarray(n0), np.asarray(n1), atol=1e-6, rtol=1e-6)


def test_get_n_in2():
    args = dict(n_in=100, n_out=24, n_total=1000, n_current=220)
    assert random2.get_n_in2(**args) == jax2.get_n_in2(**args)
    args = dict(n_in=10, n_out=0, n_total=1000, n_current=0)
    assert random2.get_n_in2(**args) == jax2.get_n_in2(**args)


def test_fun2n():
    def fun(n: int):
        r = 0.5
        w = 0.05
        x = np.random.uniform(low=-1.0, high=+1.0, size=(n, 2))
        d = np.linalg.norm(x, axis=-1)
        b = np.logical_and(r - w < d, d < r + w)
        return x[b]

    nn = 500
    np.random.seed(4)
    y0 = random2.fun2n(fun=fun, n=nn, verbose=0)
    np.random.seed(4)
    y1 = jax2.fun2n(fun=fun, n=nn)
    assert np.allclose(np.asarray(y0), np.asarray(y1), atol=1e-6, rtol=1e-6)
    assert len(np.asarray(y1)) == nn


def test_choose_from_sections_and_uniform_grid():
    np.random.seed(5)
    c0 = random2.choose_from_sections(n_total=120, n_sections=6, n_choose_per_section=3, flatten=True)
    np.random.seed(5)
    c1 = jax2.choose_from_sections(n_total=120, n_sections=6, n_choose_per_section=3, flatten=True)
    assert np.array_equal(np.asarray(c0), np.asarray(c1))

    rng = np.random.default_rng(6)
    x = rng.normal(size=(200, 2))
    np.random.seed(6)
    i0 = random2.choose_from_uniform_grid(x=x, n=20)
    np.random.seed(6)
    i1 = jax2.choose_from_uniform_grid(x=x, n=20)
    assert np.array_equal(np.asarray(i0), np.asarray(i1))


def test_block_shuffle():
    arr = np.arange(24)

    np.random.seed(7)
    b0 = random2.block_shuffle(arr=arr.copy(), block_size=4, inside=False)
    np.random.seed(7)
    b1 = jax2.block_shuffle(arr=arr.copy(), block_size=4, inside=False)
    assert np.array_equal(np.asarray(b0), np.asarray(b1))

    np.random.seed(8)
    c0 = random2.block_shuffle(arr=arr.copy(), block_size=4, inside=True)
    np.random.seed(8)
    c1 = jax2.block_shuffle(arr=arr.copy(), block_size=4, inside=True)
    assert np.array_equal(np.asarray(c0), np.asarray(c1))

    np.random.seed(9)
    d0 = random2.block_shuffle(arr=12, block_size=3, inside=False)
    np.random.seed(9)
    d1 = jax2.block_shuffle(arr=12, block_size=3, inside=False)
    assert np.array_equal(np.asarray(d0), np.asarray(d1))
