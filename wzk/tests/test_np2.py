from __future__ import annotations

from typing import Literal

import numpy as np

from wzk import np2, testing
from wzk.logger import log_print


def test_dummy_array() -> None:
    arr = np.random.random((4, 4))
    idx = (1, 2, 3, 4)
    d = np2.DummyArray(arr=arr, shape=(4, 5, 6, 6))
    assert np.allclose(arr, d[idx])

    d = np2.DummyArray(arr=1, shape=(2, 2))
    assert np.allclose(1, d[1, :])


def test_initialize_array() -> None:
    shape = [4, (4,), (1, 2, 3, 4)]
    dtype = [float, int, bool]
    order = ["C", "F"]

    for s in shape:
        for d in dtype:
            for o in order:
                o: Literal["C", "F"]

                assert testing.compare_arrays(
                    a=np2.initialize_array(shape=s, dtype=d, order=o, mode="zeros"),
                    b=np.zeros(shape=s, dtype=d, order=o),
                )

                assert testing.compare_arrays(
                    a=np2.initialize_array(shape=s, dtype=d, order=o, mode="ones"),
                    b=np.ones(shape=s, dtype=d, order=o),
                )

                assert testing.compare_arrays(
                    a=np2.initialize_array(shape=s, dtype=d, order=o, mode="empty"),
                    b=np.empty(shape=s, dtype=d, order=o),
                )

                np.random.seed(0)
                a = np2.initialize_array(shape=s, dtype=d, order=o, mode="random")
                np.random.seed(0)
                b = np.random.random(s).astype(dtype=d, order=o)
                assert testing.compare_arrays(a=a, b=b)


def test_np_isinstance() -> None:
    assert np2.np_isinstance(4.4, float)
    assert not np2.np_isinstance(4.4, int)

    assert np2.np_isinstance(("this", "that"), tuple)
    assert np2.np_isinstance(("this", "that"), tuple)

    assert np2.np_isinstance(np.full((4, 4), "bert"), str)
    assert np2.np_isinstance(np.ones((5, 5), dtype=bool), bool)

    assert np2.np_isinstance(np.ones(4, dtype=int), int)
    assert not np2.np_isinstance(np.ones(4, dtype=int), float)


def test_insert() -> None:
    a = np.ones((4, 5, 3))
    val = 2

    np2.insert(a=a, idx=(1, 2), axis=(0, 2), val=val)

    assert np.allclose(a[1, :, 2], val)


def test_argmax() -> None:
    n = 100
    axis = (0, 2)
    size = (3, 4, 5, 6)

    a = np.random.randint(n, size=size)
    i = np2.argmax(a, axis=axis)

    e = np2.extract(a=a, axis=axis, idx=i, mode="orange")
    amax = np.max(a, axis=axis)
    assert np.allclose(amax, e)


def test_argmin() -> None:
    n = 1000
    axis = (1, 3, 5)
    size = (3, 4, 5, 6, 7, 8, 9)

    a = np.random.randint(n, size=size)
    i = np2.argmin(a, axis=axis)

    e = np2.extract(a=a, axis=axis, idx=i, mode="orange")
    amin = np.min(a, axis=axis)
    assert np.allclose(amin, e)


def test_scalar2array() -> None:
    assert np.array_equal([np.array([1])], np2.scalar2array(1, shape=1, squeeze=False))
    assert np.array_equal(np.array([1]), np2.scalar2array(1, shape=1, squeeze=True))

    assert np.array_equal(
        [
            np.array(["a", "a", "a"], dtype="<U1"),
            np.array(["b", "b", "b"], dtype="<U1"),
            np.array(["c", "c", "c"], dtype="<U1"),
        ],
        np2.scalar2array("a", "b", "c", shape=3),
    )

    assert np.array_equal(
        [np.array([1, 1, 1]), np.array([None, None, None], dtype=object), np.array(["a", "a", "a"], dtype="<U1")],
        np2.scalar2array(1, None, "a", shape=3),
    )


def test_find_values() -> None:
    arr = np.array([3, 5, 5, 6, 7, 8, 8, 8, 10, 11, 1])
    values = [3, 5, 8]
    res = np2.find_values(arr=arr, values=values)
    true = np.array([True, True, True, False, False, True, True, True, False, False, False])

    assert np.array_equal(res, true)


def test_tile_offset() -> None:
    a = np.arange(3)
    res = np2.tile_offset(a=a, reps=3, offsets=10)
    true = np.array([0, 1, 2, 10, 11, 12, 20, 21, 22])
    assert np.array_equal(res, true)

    a = np.arange(12).reshape(3, 4)
    res = np2.tile_offset(a=a, reps=2, offsets=(100, 1000))
    true = np.array([
        [0, 1, 2, 3, 1000, 1001, 1002, 1003],
        [4, 5, 6, 7, 1004, 1005, 1006, 1007],
        [8, 9, 10, 11, 1008, 1009, 1010, 1011],
    ])
    assert np.array_equal(res, true)

    a = np.arange(12).reshape(4, 3)
    res = np2.tile_offset(a=a, reps=(2, 3), offsets=(100, 1000))
    true = np.array([
        [0, 1, 2, 1000, 1001, 1002, 2000, 2001, 2002],
        [3, 4, 5, 1003, 1004, 1005, 2003, 2004, 2005],
        [6, 7, 8, 1006, 1007, 1008, 2006, 2007, 2008],
        [9, 10, 11, 1009, 1010, 1011, 2009, 2010, 2011],
        [100, 101, 102, 1100, 1101, 1102, 2100, 2101, 2102],
        [103, 104, 105, 1103, 1104, 1105, 2103, 2104, 2105],
        [106, 107, 108, 1106, 1107, 1108, 2106, 2107, 2108],
        [109, 110, 111, 1109, 1110, 1111, 2109, 2110, 2111],
    ])
    assert np.array_equal(res, true)


def test_construct_array() -> None:
    b = np2.construct_array(shape=10, val=[1, 2, 3], idx=[2, 4, 5], dtype=None, insert_mode=None)
    assert np.allclose(b, [0, 0, 1, 0, 2, 3, 0, 0, 0, 0])


def test_round_dict() -> None:
    d = {
        "a": 1.123,
        "b": {
            "c": np.arange(5) / 27,
            "d": {"e": "why", "f": ["why", "not", "why"]},
        },
    }

    d_round = np2.round_dict(d=d, decimals=1)
    log_print(d_round)


def test_get_interval_indices() -> None:
    arr = [
        np.array([0, 0, 0, 0]),
        np.array([0, 0, 0, 1]),
        np.array([0, 1, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([1, 0, 0, 1]),
        np.array([1, 1, 0, 1]),
        np.array([1, 1, 1, 1]),
    ]
    res = [
        np.zeros((0, 2)),
        np.array([[3, 4]]),
        np.array([[1, 3]]),
        np.array([[0, 1]]),
        np.array([[0, 1], [3, 4]]),
        np.array([[0, 2], [3, 4]]),
        np.array([[0, 4]]),
    ]

    for aa, rr in zip(arr, res, strict=True):
        assert np.array_equal(np2.get_interval_indices(aa), rr)


def test_clip_periodic() -> None:
    a_min = 5
    a_max = 37

    x = np.linspace(start=0, stop=100, num=10000)
    x2 = np2.clip_periodic(x=x, a_min=a_min, a_max=a_max)

    assert np.all(x2 >= a_min)
    assert np.all(x2 <= a_max)


def test_diag_wrapper() -> None:
    a = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    b = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 4]])

    assert np.allclose(np2.diag_wrapper(n=4, x=2), a)
    assert np.allclose(np2.diag_wrapper(n=3, x=[2, 3, 4]), b)

    assert np.allclose(np2.diag_wrapper(n=4, x=a), a)
    assert np.allclose(np2.diag_wrapper(n=3, x=b), b)


def test_slicen() -> None:
    sl = np2.slicen([1], [2])
    assert sl == (slice(1, 2),)


def _reference_convolve_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Frozen copy of the original O(H*W*k^2) loop — the behavioural oracle the
    fast (FFT) ``np2.convolve_2d`` must reproduce bit-for-bit on integer inputs."""
    s = np.array(img.shape)
    ks = np.array(kernel.shape)
    ks2 = ks // 2
    out = np.zeros(s, float)
    for i0 in range(ks2[0], s[0] - ks2[0]):
        for i1 in range(ks2[1], s[1] - ks2[1]):
            out[i0, i1] = np.sum(img[i0 - ks2[0]: i0 + ks2[0] + 1, i1 - ks2[1]: i1 + ks2[1] + 1] * kernel)
    return out


def test_convolve_2d_matches_reference_loop() -> None:
    rng = np.random.default_rng(0)
    cases = [
        (rng.integers(0, 2, (60, 55)).astype(float), np.ones((7, 5))),
        (rng.integers(0, 2, (50, 40)).astype(bool), rng.integers(0, 2, (9, 9)).astype(bool)),
        (rng.integers(0, 4, (31, 33)).astype(float), np.ones((1, 1))),
        (np.ones((20, 24)), np.ones((3, 7))),
        (rng.integers(0, 3, (45, 45)).astype(float), rng.integers(0, 2, (5, 11)).astype(float)),
    ]
    for img, kernel in cases:
        assert np.array_equal(np2.convolve_2d(img, kernel), _reference_convolve_2d(img, kernel))


def test_convolve_2d_preserves_full_fit_positions() -> None:
    # juwo's actual use: cells where the whole kernel footprint lands on free
    # space (``conv == kernel.sum()``) must be identical to the reference.
    free = np.ones((80, 70))
    free[20:40, 30:55] = 0
    free[10:15, 5:25] = 0
    kernel = np.ones((11, 9))
    out = np2.convolve_2d(free, kernel)
    ref = _reference_convolve_2d(free, kernel)
    assert np.array_equal(out == kernel.sum(), ref == kernel.sum())


def test_convolve_2d_zeros_border() -> None:
    out = np2.convolve_2d(np.ones((10, 10)), np.ones((5, 3)))
    assert np.all(out[:2, :] == 0)
    assert np.all(out[-2:, :] == 0)
    assert np.all(out[:, :1] == 0)
    assert np.all(out[:, -1:] == 0)


def test_convolve_2d_rejects_even_kernel() -> None:
    import pytest

    with pytest.raises(AssertionError):
        np2.convolve_2d(np.ones((6, 6)), np.ones((2, 2)))


def test_convolve_2d_is_fast() -> None:
    # 300x300 with a 31x21 footprint: the O(H*W*k^2) Python loop took ~0.6 s,
    # the FFT path is ~0.01 s. Guards against regressing to the loop.
    import time

    img = np.ones((300, 300))
    kernel = np.ones((31, 21))
    t0 = time.perf_counter()
    np2.convolve_2d(img, kernel)
    dt = time.perf_counter() - t0
    assert dt < 0.2, f"convolve_2d took {dt * 1e3:.0f} ms — regressed to an O(H*W*k^2) loop?"
