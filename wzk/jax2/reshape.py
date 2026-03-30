from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ._types import ArrayLike, ShapeLike, int32
from .range import slicen
from .basics import scalar2array
from .find import align_shapes


def repeat2new_shape(img: ArrayLike, new_shape: ShapeLike) -> jax.Array:
    img = np.asarray(img)
    reps = np.ceil(np.array(new_shape) / np.array(img.shape)).astype(int)
    for i in range(img.ndim):
        img = np.repeat(img, repeats=reps[i], axis=i)

    img = img[slicen(end=np.asarray(new_shape, dtype=int32))]
    return jnp.asarray(img)


def change_shape(arr: ArrayLike, mode: str = "even") -> jax.Array:
    arr = np.asarray(arr)
    s = np.array(arr.shape)

    if mode == "even":
        s_new = s + s % 2
    elif mode == "odd":
        s_new = (s // 2) * 2 + 1
    else:
        raise ValueError(f"Unknown mode {mode}")

    arr_odd = np.zeros(s_new, dtype=arr.dtype)
    fill_with_air_left(arr=arr, out=arr_odd)
    return jnp.asarray(arr_odd)


def flatten_without_last(x: ArrayLike) -> jax.Array:
    return jnp.reshape(x, (-1, np.shape(x)[-1]))


def flatten_without_first(x: ArrayLike) -> jax.Array:
    return jnp.reshape(x, (np.shape(x)[0], -1))


def fill_with_air_left(arr: ArrayLike, out: np.ndarray) -> None:
    assert np.ndim(arr) == np.ndim(out)
    out[slicen(end=np.asarray(np.shape(arr), dtype=int32))] = arr


def array2array(a: np.ndarray, shape: ShapeLike, fill_value: str = "empty") -> jax.Array:
    a = np.atleast_1d(a)
    shape_arr = np.atleast_1d(np.asarray(shape, dtype=int32))
    shape_tuple = tuple(int(v) for v in shape_arr.tolist())

    if np.size(a) == 1:
        return jnp.asarray(scalar2array(a.item(), shape=shape_tuple))

    s = align_shapes(shape_arr, np.asarray(a.shape, dtype=int32))
    s = tuple(slice(None) if ss == 1 else np.newaxis for ss in s)
    if fill_value == "empty":
        b = np.empty(shape_tuple, dtype=a.dtype)
    else:
        b = np.full(shape_tuple, fill_value, dtype=a.dtype)

    b[:] = a[s]
    return jnp.asarray(b)
