import numpy as np

from wzk import jax2
from wzk import np2


def test_initialize_array():
    shape = (3, 4)

    a0 = np.asarray(np2.initialize_array(shape=shape, mode="zeros", dtype=float, order="C"))
    a1 = np.asarray(jax2.initialize_array(shape=shape, mode="zeros", dtype=float, order="C"))
    assert np.array_equal(a0, a1)

    np.random.seed(7)
    r0 = np.asarray(np2.initialize_array(shape=shape, mode="random", dtype=float, order="F"))
    np.random.seed(7)
    r1 = np.asarray(jax2.initialize_array(shape=shape, mode="random", dtype=float, order="F"))
    assert np.allclose(r0, r1)


def test_scalar2array_and_isinstance():
    arr0 = np2.scalar2array(1, shape=3)
    arr1 = jax2.scalar2array(1, shape=3)
    assert np.array_equal(np.asarray(arr0), np.asarray(arr1))

    assert np2.np_isinstance(np.ones(4, dtype=int), int) == jax2.np_isinstance(np.ones(4, dtype=int), int)
    assert np2.np_isinstance(4.4, float) == jax2.np_isinstance(4.4, float)


def test_insert_extract_argmax_argmin():
    np.random.seed(0)
    a = np.random.randint(100, size=(3, 4, 5, 6))

    a0 = a.copy()
    a1 = a.copy()
    np2.insert(a=a0, idx=(1, 2), axis=(0, 2), val=123)
    out = jax2.insert(a=a1, idx=(1, 2), axis=(0, 2), val=123)
    if out is not None:
        a1 = np.asarray(out)
    assert np.array_equal(a0, a1)

    axis = (0, 2)
    i0 = np2.argmax(a, axis=axis)
    i1 = jax2.argmax(a, axis=axis)
    assert np.array_equal(np.asarray(i0), np.asarray(i1))

    e0 = np2.extract(a=a, idx=i0, axis=axis, mode="orange")
    e1 = jax2.extract(a=a, idx=i1, axis=axis, mode="orange")
    assert np.array_equal(np.asarray(e0), np.asarray(e1))

    axis = (1, 3)
    i0 = np2.argmin(a, axis=axis)
    i1 = jax2.argmin(a, axis=axis)
    assert np.array_equal(np.asarray(i0), np.asarray(i1))


def test_find_values_and_intervals():
    arr = np.array([3, 5, 5, 6, 7, 8, 8, 8, 10, 11, 1])
    values = [3, 5, 8]

    r0 = np2.find_values(arr=arr, values=values)
    r1 = jax2.find_values(arr=arr, values=values)
    assert np.array_equal(np.asarray(r0), np.asarray(r1))

    bool_arr = np.array([1, 1, 0, 1, 0, 0, 1], dtype=bool)
    g0 = np2.get_interval_indices(bool_arr)
    g1 = jax2.get_interval_indices(bool_arr)
    assert np.array_equal(np.asarray(g0), np.asarray(g1))


def test_tile_offset():
    a = np.arange(12).reshape(4, 3)
    r0 = np2.tile_offset(a=a, reps=(2, 3), offsets=(100, 1000))
    r1 = jax2.tile_offset(a=a, reps=(2, 3), offsets=(100, 1000))
    assert np.array_equal(np.asarray(r0), np.asarray(r1))


def test_construct_array_diag_slicen():
    b0 = np2.construct_array(shape=10, val=[1, 2, 3], idx=[2, 4, 5], dtype=None, insert_mode=None)
    b1 = jax2.construct_array(shape=10, val=[1, 2, 3], idx=[2, 4, 5], dtype=None, insert_mode=None)
    assert np.allclose(np.asarray(b0), np.asarray(b1))

    d0 = np2.diag_wrapper(n=3, x=[2, 3, 4])
    d1 = jax2.diag_wrapper(n=3, x=[2, 3, 4])
    assert np.allclose(np.asarray(d0), np.asarray(d1))

    assert np2.slicen([1], [2]) == jax2.slicen([1], [2])


def test_clip_periodic_and_clip2():
    x = np.linspace(-10, 30, num=500)
    a_min, a_max = 5, 17

    c0 = np2.clip_periodic(x=x, a_min=a_min, a_max=a_max)
    c1 = jax2.clip_periodic(x=x, a_min=a_min, a_max=a_max)
    assert np.allclose(np.asarray(c0), np.asarray(c1))

    y = np.random.default_rng(1).normal(size=300)
    u0 = np2.clip2(y, clip=0.5, mode="value")
    u1 = jax2.clip2(y, clip=0.5, mode="value")
    assert np.allclose(np.asarray(u0), np.asarray(u1), atol=1e-6, rtol=1e-6)
