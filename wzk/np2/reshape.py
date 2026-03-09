from __future__ import annotations

import numpy as np

from ._types import ArrayLike, ShapeLike
from .basics import scalar2array
from .find import align_shapes
from .range import slicen


def repeat2new_shape(img: ArrayLike, new_shape: ShapeLike) -> np.ndarray:
    reps = np.ceil(np.array(new_shape) / np.array(img.shape)).astype(int)
    for i in range(img.ndim):
        img = np.repeat(img, repeats=reps[i], axis=i)

    img = img[slicen(end=new_shape)]
    return img


def change_shape(arr: np.ndarray, mode: str = "even") -> np.ndarray:
    s = np.array(arr.shape)

    if mode == "even":
        s_new = s + s % 2
    elif mode == "odd":
        s_new = (s // 2) * 2 + 1
    else:
        raise ValueError(f"Unknown mode {mode}")

    arr_odd = np.zeros(s_new, dtype=arr.dtype)
    fill_with_air_left(arr=arr, out=arr_odd)
    return arr_odd


def flatten_without_last(x: ArrayLike) -> np.ndarray:
    return np.reshape(x, (-1, np.shape(x)[-1]))


def flatten_without_first(x: ArrayLike) -> np.ndarray:
    return np.reshape(x, (np.shape(x)[0], -1))


def fill_with_air_left(arr: ArrayLike, out: np.ndarray) -> None:
    assert arr.ndim == out.ndim
    out[slicen(end=arr.shape)] = arr


def array2array(a: np.ndarray, shape: ShapeLike, fill_value: str = "empty") -> np.ndarray:
    a = np.atleast_1d(a)
    if np.size(a) == 1:
        return scalar2array(a.item(), shape=shape)

    s = align_shapes(shape, a.shape)
    s = tuple([slice(None) if ss == 1 else np.newaxis for ss in s])
    if fill_value == "empty":
        b = np.empty(shape, dtype=a.dtype)
    else:
        b = np.full(shape, fill_value, dtype=a.dtype)

    b[:] = a[s]
    return b
