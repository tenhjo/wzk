from __future__ import annotations

from itertools import product
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from ._types import ArrayLike, AxisLike, BoolArray, float32, int32
from . import shape as sh


def object2numeric_array(arr: ArrayLike) -> np.ndarray:
    s = np.shape(arr)
    arr = np.array([v for v in np.ravel(arr)])
    arr = np.reshape(arr, s + np.shape(arr)[1:])
    return arr


def numeric2object_array(arr: ArrayLike) -> np.ndarray:
    arr = np.asarray(arr)
    n = arr.shape[0]
    arr_obj = np.zeros(n, dtype=object)
    for i in range(n):
        arr_obj[i] = arr[i]
    return arr_obj


def scalar2array(*val_or_arr: Any,
                 shape: int | tuple[int, ...] | list[int],
                 squeeze: bool = True,
                 safe: bool = True) -> Any:
    """Broadcast scalar-like values to numpy arrays with the provided shape."""
    shape = sh.shape_wrapper(shape)

    res = []
    for voa in val_or_arr:
        try:
            if isinstance(voa, str):
                dtype = np.array(voa).dtype
            elif isinstance(voa, np.ndarray):
                dtype = voa.dtype
            elif isinstance(voa, jax.Array):
                dtype = np.asarray(voa).dtype
            else:
                dtype = type(voa)

            res_i = np.empty(shape=shape, dtype=dtype)
            res_i[:] = np.array(voa).copy()
            res.append(res_i)

        except ValueError:
            if safe:
                assert np.all(np.shape(voa) == shape), f"{np.shape(voa)} != {shape}"
            res.append(voa)

    if len(res) == 1 and squeeze:
        return res[0]
    return res


def args2arrays(*args: Any) -> list[np.ndarray]:
    return [np.array(a) for a in args]


def unify(x: ArrayLike) -> np.generic:
    x = np.atleast_1d(x)
    assert np.allclose(x, x.mean())
    x_mean = np.mean(x)
    return x_mean.astype(x.dtype)


def __fill_index_with(idx: Any,
                      axis: AxisLike,
                      shape: tuple[int, ...],
                      mode: str = "slice") -> tuple[Any, ...] | Any:
    """
    orange <-> orth-range
    """
    axis = sh.axis_wrapper(axis=axis, n_dim=len(shape))
    if mode == "slice":
        idx_with_ = [slice(None) for _ in range(len(shape) - len(axis))]
    elif mode == "orange":
        ogrid_idx = tuple(slice(0, int(s)) for i, s in enumerate(shape) if i not in axis)
        idx_with_ = np.ogrid[ogrid_idx]
        idx_with_ = list(idx_with_)
    elif mode is None:
        return idx
    else:
        raise ValueError(f"Unknown mode {mode}")

    idx = np.array(idx)
    for i, ax in enumerate(axis):
        idx_with_.insert(ax, idx[..., i])

    return tuple(idx_with_)


def insert(a: ArrayLike,
           val: Any,
           idx: Any,
           axis: AxisLike,
           mode: str = "slice") -> jax.Array | np.ndarray:
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    if isinstance(a, jax.Array):
        if isinstance(idx, list):
            idx = jnp.asarray(idx)
        return a.at[idx].set(val)
    a[idx] = val
    return a


def extract(a: ArrayLike, idx: Any, axis: AxisLike, mode: str = "slice") -> Any:
    idx = __fill_index_with(idx=idx, axis=axis, shape=a.shape, mode=mode)
    return a[idx]


def __argfun(a: ArrayLike,
             axis: AxisLike,
             fun: Callable[..., np.ndarray]) -> np.ndarray | tuple[np.ndarray, ...]:
    a = np.asarray(a)
    axis = sh.axis_wrapper(axis=axis, n_dim=a.ndim)
    if len(axis) == 1:
        return fun(a, axis=axis[0])
    if len(axis) == a.ndim:
        return np.unravel_index(fun(a), shape=a.shape)

    axis_inv = sh.axis_wrapper(axis=axis, n_dim=a.ndim, invert=True)
    shape_inv = sh.get_subshape(shape=a.shape, axis=axis_inv)
    shape = sh.get_subshape(shape=a.shape, axis=axis)

    a2 = np.transpose(a, axes=np.hstack((axis_inv, axis))).reshape(shape_inv + (-1,))
    idx = fun(a2, axis=-1)
    idx = np.array(np.unravel_index(idx, shape=shape))

    return np.transpose(idx, axes=np.roll(np.arange(idx.ndim), -1))


def argmax(a: ArrayLike, axis: AxisLike = None) -> np.ndarray | tuple[np.ndarray, ...]:
    return __argfun(a=a, axis=axis, fun=np.argmax)


def argmin(a: ArrayLike, axis: AxisLike = None) -> np.ndarray | tuple[np.ndarray, ...]:
    return __argfun(a=a, axis=axis, fun=np.argmin)


def allclose(a: ArrayLike,
             b: ArrayLike,
             rtol: float = 1.0e-5,
             atol: float = 1.0e-8,
             axis: AxisLike = None) -> BoolArray | bool:
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    axis = np.array(sh.axis_wrapper(axis=axis, n_dim=a.ndim))
    assert len(axis) <= len(a.shape)
    if np.isscalar(a) and np.isscalar(b):
        return np.allclose(a, b)

    shape = np.array(a.shape)[axis]
    bool_arr = np.zeros(shape, dtype=bool)
    for i in product(*(range(s) for s in shape)):
        bool_arr[i] = np.allclose(extract(a, idx=i, axis=axis),
                                  extract(b, idx=i, axis=axis),
                                  rtol=rtol, atol=atol)
    return bool_arr


def __wrapper_pair2list_fun(*args: ArrayLike,
                            fun: Callable[[Any, Any], Any]) -> Any:
    assert len(args) >= 2
    res = fun(args[0], args[1])
    for a in args[2:]:
        res = fun(res, a)
    return res


def minimum(*args: ArrayLike) -> jax.Array:
    return __wrapper_pair2list_fun(*args, fun=jnp.minimum)


def maximum(*args: ArrayLike) -> jax.Array:
    return __wrapper_pair2list_fun(*args, fun=jnp.maximum)


def logical_or(*args: ArrayLike) -> jax.Array:
    return __wrapper_pair2list_fun(*args, fun=jnp.logical_or)


def logical_and(*args: ArrayLike) -> jax.Array:
    return __wrapper_pair2list_fun(*args, fun=jnp.logical_and)


def max_size(*args: ArrayLike) -> int:
    return int(np.max([np.size(a) for a in args]))


def min_size(*args: ArrayLike) -> int:
    return int(np.min([np.size(a) for a in args]))


def argmax_size(*args: ArrayLike) -> int:
    return int(np.argmax([np.size(a) for a in args]))


def max_len(*args: ArrayLike) -> int:
    return int(np.max([len(a) for a in args]))


def squeeze_all(*args: ArrayLike) -> list[jax.Array]:
    return [jnp.squeeze(a) for a in args]


def round2(x: ArrayLike, decimals: int | None = None) -> jax.Array | np.ndarray:
    try:
        decimals_i = 0 if decimals is None else decimals
        return jnp.round(x, decimals=decimals_i)
    except Exception:  # noqa: BLE001
        return np.array(x)


def clip_periodic(x: ArrayLike, a_min: float, a_max: float) -> jax.Array:
    x = jnp.asarray(x)
    x = x - a_min
    x = jnp.mod(x, a_max - a_min)
    x = x + a_min
    return x


def clip2(x: ArrayLike, clip: float, mode: str, axis: int = -1) -> jax.Array | ArrayLike:
    if not mode:
        return x

    x = jnp.asarray(x)
    if mode == "value":
        return jnp.clip(x, min=-clip, max=+clip)

    if mode == "norm":
        n = jnp.linalg.norm(x, axis=axis, keepdims=True)
        scale = jnp.where(n > clip, clip / (n + 1e-12), 1.0)
        return x * scale

    if mode == "norm-force":
        n = jnp.linalg.norm(x, axis=axis, keepdims=True)
        return x * (clip / (n + 1e-12))

    raise ValueError(f"Unknown mode: '{mode}'")


def load_dict(file: str) -> dict[str, Any]:
    d = np.load(file, allow_pickle=True)
    try:
        d = d.item()
    except AttributeError:
        d = d["arr"]
        d = d.item()

    assert isinstance(d, dict)
    return d


def round_dict(d: dict[str, Any], decimals: int | None = None) -> dict[str, Any]:
    for key in d.keys():
        value = d[key]
        if isinstance(value, dict):
            d[key] = round_dict(d=value, decimals=decimals)
        else:
            d[key] = round2(x=value, decimals=decimals)
    return d


def rolling_window(a: ArrayLike, window: int) -> jax.Array:
    """https://stackoverflow.com/a/6811241/7570817"""
    a = np.array(a)
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    out = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if np.issubdtype(out.dtype, np.floating):
        return jnp.asarray(out, dtype=float32)
    if np.issubdtype(out.dtype, np.integer):
        return jnp.asarray(out, dtype=int32)
    return jnp.asarray(out)
