from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ._types import ArrayLike, int32
from .basics import args2arrays, rolling_window


def find_subarray(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    """Find sequence b in a and return start indices."""
    a, b = np.atleast_1d(a, b)

    window = len(b)
    a_window = np.asarray(rolling_window(a=a, window=window))
    idx = np.nonzero(np.array(np.sum(a_window == b, axis=-1) == window, bool))[0]
    return idx


def find_values(arr: ArrayLike, values: ArrayLike) -> np.ndarray:
    arr = np.asarray(arr)
    res = np.zeros_like(arr, dtype=bool)
    for v in values:
        res[~res] = arr[~res] == v
    return res


def find_common_values(a: ArrayLike, b: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """If values repeat, first match in b is used."""
    i_a = []
    i_b = []
    for i, aa in a:
        for j, bb in b:
            if np.allclose(aa, bb):
                i_a.append(i)
                i_b.append(j)
                break
    return np.array(i_a, dtype=int32), np.array(i_b, dtype=int32)


def find_array_occurrences(a: ArrayLike, o: ArrayLike) -> np.ndarray:
    a = np.asarray(a)
    o = np.asarray(o)
    assert a.ndim == o.ndim
    assert a.shape[-1] == o.shape[-1]

    if a.ndim == 2:
        b = a[:, np.newaxis, :] == o[np.newaxis, :, :]
        b = np.sum(b, axis=-1) == o.shape[-1]
        i = np.array(np.nonzero(np.array(b))).T
    else:
        raise ValueError

    return i


def get_element_overlap(arr1: ArrayLike,
                        arr2: ArrayLike | None = None,
                        verbose: int = 0) -> np.ndarray:
    if arr2 is None:
        arr2 = arr1

    overlap = np.zeros((len(arr1), len(arr2)), dtype=int32)
    for i, arr_i in enumerate(arr1):
        if verbose > 0:
            print(f"{i} / {len(arr1)}")
        for j, arr_j in enumerate(arr2):
            for k in arr_i:
                if k in arr_j:
                    overlap[i, j] += 1

    return overlap


def get_first_row_occurrence(bool_arr: ArrayLike) -> np.ndarray:
    nz_i, nz_j = np.nonzero(bool_arr)
    u, idx = np.unique(nz_i, return_index=True)
    res = np.full(np.shape(bool_arr)[0], fill_value=-1, dtype=int32)
    res[u] = nz_j[idx]
    return res


def fill_interval_indices(interval_list: list[list[int]] | np.ndarray, n: int) -> np.ndarray:
    if isinstance(interval_list, np.ndarray):
        interval_list = interval_list.tolist()

    if np.size(interval_list) == 0:
        return np.array([[1, n]], dtype=int32)

    if interval_list[0][0] != 0:
        interval_list.insert(0, [0, interval_list[0][0]])

    if interval_list[-1][1] != n:
        interval_list.insert(len(interval_list), [interval_list[-1][1], n])

    i = 1
    while i < len(interval_list):
        if interval_list[i - 1][1] != interval_list[i][0]:
            interval_list.insert(i, [interval_list[i - 1][1], interval_list[i][0]])
        i += 1

    return np.array(interval_list, dtype=int32)


def get_interval_indices(bool_array: ArrayLike, expand: bool = False) -> np.ndarray | list[list[int]]:
    bool_array = np.asarray(bool_array).astype(bool)
    assert bool_array.ndim == 1

    interval_list = np.where(np.diff(bool_array) != 0)[0] + 1
    if bool_array[0]:
        interval_list = np.concatenate([[0], interval_list])
    if bool_array[-1]:
        interval_list = np.concatenate([interval_list, bool_array.shape])
    interval_list = interval_list.reshape(-1, 2)

    if expand:
        return [list(range(i0, i1)) for (i0, i1) in interval_list]

    return interval_list


def get_cropping_indices(pos: ArrayLike,
                         shape_small: ArrayLike,
                         shape_big: ArrayLike,
                         mode: str = "lower_left") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape_small, shape_big = args2arrays(shape_small, shape_big)

    if mode == "center":
        assert np.all(np.array(shape_small) % 2 == 1), shape_small
        shape_small2 = (np.array(shape_small) - 1) // 2
        ll_big = pos - shape_small2
        ur_big = pos + shape_small2 + 1

    elif mode == "lower_left":
        ll_big = pos
        ur_big = pos + shape_small

    elif mode == "upper_right":
        ll_big = pos - shape_small
        ur_big = pos

    else:
        raise ValueError(f"Invalid position mode {mode}")

    ll_small = np.where(ll_big < 0, -ll_big, 0)
    ur_small = np.where(shape_big - ur_big < 0, shape_small + (shape_big - ur_big), shape_small)

    ll_big = np.where(ll_big < 0, 0, ll_big)
    ur_big = np.where(shape_big - ur_big < 0, shape_big, ur_big)

    return ll_big, ur_big, ll_small, ur_small


def find_closest(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    d = jnp.linalg.norm(x[:, jnp.newaxis, :] - y[jnp.newaxis, :, :], axis=-1)
    i_x = jnp.argmin(d, axis=0)
    i_y = jnp.argmin(d, axis=1)
    return np.asarray(i_x, dtype=int32), np.asarray(i_y, dtype=int32)


def find_consecutives(x: ArrayLike, n: int) -> np.ndarray:
    if n == 1:
        return np.arange(len(x), dtype=int32)
    assert n > 1
    c = np.convolve(np.abs(np.diff(x)), v=np.ones(n - 1), mode="valid")
    return np.nonzero(np.equal(c, 0))[0]


def find_largest_consecutives(x: ArrayLike) -> tuple[int, np.ndarray]:
    c = np.convolve(np.abs(np.diff(x)), v=np.ones(1), mode="valid")
    i2 = np.nonzero(np.equal(c, 0))[0]
    i2 -= np.arange(len(i2))
    _, c2 = np.unique(i2, return_counts=True)
    n = 1 if c2.size == 0 else int(c2.max() + 1)
    return n, find_consecutives(x, n=n)


def find_block_shuffled_order(a: ArrayLike,
                              b: ArrayLike,
                              block_size: int,
                              threshold: float,
                              verbose: int = 1) -> np.ndarray:
    n = len(a)
    m = len(b)
    assert n == m
    assert n % block_size == 0

    nn = n // block_size
    idx = np.empty(nn)

    for i in range(nn):
        for j in range(nn):
            d = (a[i * block_size:(i + 1) * block_size] -
                 b[j * block_size:(j + 1) * block_size])

            d = np.abs(d).max()
            if d < threshold:
                idx[i] = j
                if verbose > 0:
                    print(i, j, d)

    return idx


def align_shapes(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    idx = find_subarray(a=a, b=b).item()
    aligned_shape = np.full(shape=len(a), fill_value=-1, dtype=int32)
    aligned_shape[idx:idx + len(b)] = 1
    return aligned_shape
