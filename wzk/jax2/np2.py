from __future__ import annotations

from types import EllipsisType
from typing import Any, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._types import ArrayLike, AxisLike, ShapeLike, float32, int32
from . import basics
from . import dtypes2
from . import find
from . import reshape
from .range import slicen


class DummyArray:
    """Allows indexing but always returns the same dummy value."""

    def __init__(self, arr: Any, shape: ShapeLike):
        self.arr = arr
        self.shape = shape

    def __assert_int(self, item: int, i: int) -> None:
        assert item in range(-self.shape[i], self.shape[i])

    def __assert_slice(self, item: slice, i: int) -> None:
        _ = item, i

    def __assert_ellipsis(self, item: EllipsisType, i: int) -> None:
        _ = item, i

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, int):
            self.__assert_int(item=item, i=0)
        elif isinstance(item, slice):
            self.__assert_slice(item=item, i=0)
        elif isinstance(item, type(...)):
            self.__assert_ellipsis(item=item, i=0)
        else:
            assert len(item) == len(self.shape), f"Incompatible index {item} for array with shape {self.shape}"
            for i, item_i in enumerate(item):
                if isinstance(item_i, int):
                    self.__assert_int(item=item_i, i=i)
                elif isinstance(item_i, slice):
                    self.__assert_slice(item=item_i, i=i)
                elif isinstance(item, type(...)):
                    self.__assert_ellipsis(item=item, i=i)
                else:
                    raise ValueError

        return self.arr


def initialize_array(shape: ShapeLike,
                     mode: str = "zeros",
                     dtype: Any = None,
                     order: Literal["C", "F"] = "C") -> jax.Array:
    if mode == "zeros":
        arr = np.zeros(shape, dtype=dtype, order=order)
    elif mode == "ones":
        arr = np.ones(shape, dtype=dtype, order=order)
    elif mode == "empty":
        arr = np.empty(shape, dtype=dtype, order=order)
    elif mode == "random":
        arr = np.random.random(shape).astype(dtype=(float32 if dtype is None else dtype), order=order)
    else:
        raise ValueError(f"Unknown initialization method {mode}")
    return jnp.asarray(arr)


def np_isinstance(o: Any, c: Any) -> bool:
    if isinstance(o, (np.ndarray, jax.Array)):
        if isinstance(c, tuple):
            c2 = tuple(dtypes2.c2np[cc] for cc in c)
        else:
            c2 = dtypes2.c2np[c]
        return isinstance(np.asarray(o).flat[0], c2)
    return isinstance(o, c)


def delete_args(*args: ArrayLike, i: int, axis: AxisLike = None) -> tuple[np.ndarray, ...]:
    return tuple(np.delete(np.asarray(a), obj=i, axis=axis) for a in args)


def interleave(arrays: Sequence[ArrayLike], axis: int = 0, out: np.ndarray | None = None) -> jax.Array:
    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis + 1] + [len(arrays)] + shape[axis + 1:])
    shape[axis] = -1
    res = np.stack([np.asarray(a) for a in arrays], axis=axis + 1, out=out).reshape(shape)
    return jnp.asarray(res)


def digitize_group(x: ArrayLike, bins: ArrayLike, right: bool = False) -> list[np.ndarray]:
    from scipy.sparse import csr_matrix

    idx_x = np.digitize(x=x, bins=bins, right=right)
    n, m = len(x), len(bins) + 1
    s = csr_matrix((np.arange(n), [idx_x, np.arange(n)]), shape=(m, n))
    return [group for group in np.split(s.data, s.indptr[1:-1])]


def sort_args(idx: ArrayLike, *args: ArrayLike) -> list[np.ndarray]:
    return [np.asarray(a)[idx] for a in args]


def make_odd(arr: ArrayLike) -> jax.Array:
    arr = np.asarray(arr)
    mo = (np.array(arr.shape) + 1) % 2
    arr_new = np.zeros(np.array(arr.shape) + mo, dtype=bool)
    arr_new[slicen(end=arr.shape)] = arr
    return jnp.asarray(arr_new)


def convolve_2d(img: ArrayLike, kernel: ArrayLike) -> jax.Array:
    img = np.asarray(img)
    kernel = np.asarray(kernel)
    s = np.array(img.shape)
    ks = np.array(kernel.shape)
    assert np.all(ks % 2 == 1)

    ks2 = ks // 2
    out = np.zeros(s, dtype=float32)
    for i0 in range(ks2[0], s[0] - ks2[0]):
        for i1 in range(ks2[1], s[1] - ks2[1]):
            out[i0, i1] = np.sum(img[i0 - ks2[0]:i0 + ks2[0] + 1, i1 - ks2[1]:i1 + ks2[1] + 1] * kernel)

    return jnp.asarray(out)


def add_small2big(idx: ArrayLike,
                  small: ArrayLike,
                  big: ArrayLike,
                  mode_crop: str = "center",
                  mode_add: str = "add") -> jax.Array | None:
    idx = reshape.flatten_without_last(idx)
    n_samples, n_dim = idx.shape
    ll_big, ur_big, ll_small, ur_small = find.get_cropping_indices(pos=idx, mode=mode_crop,
                                                                   shape_small=np.shape(small)[-n_dim:],
                                                                   shape_big=np.shape(big))

    is_jax = isinstance(big, jax.Array)
    big_arr = np.array(big) if is_jax else big

    if np.ndim(small) > n_dim:
        for ll_b, ur_b, ll_s, ur_s, s in zip(ll_big, ur_big, ll_small, ur_small, small):
            big_arr[slicen(ll_b, ur_b)] += np.asarray(s)[slicen(ll_s, ur_s)]
    else:
        for ll_b, ur_b, ll_s, ur_s in zip(ll_big, ur_big, ll_small, ur_small):
            if mode_add == "add":
                big_arr[slicen(ll_b, ur_b)] += np.asarray(small)[slicen(ll_s, ur_s)]
            elif mode_add == "replace":
                big_arr[slicen(ll_b, ur_b)] = np.asarray(small)[slicen(ll_s, ur_s)]

    if is_jax:
        return jnp.asarray(big_arr)
    return None


def get_exclusion_mask(a: ArrayLike, exclude_values: ArrayLike) -> np.ndarray:
    exclude_values = np.atleast_1d(exclude_values)
    bool_a = np.ones_like(a, dtype=bool)
    for v in exclude_values:
        bool_a[np.asarray(a) == v] = False

    return bool_a


def matmul(a: ArrayLike,
           b: ArrayLike,
           axes_a: tuple[int, int] = (-2, -1),
           axes_b: tuple[int, int] = (-2, -1)) -> jax.Array:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    if axes_a == (-2, -1) and axes_b == (-2, -1):
        return a @ b

    if axes_a == (-3, -2) and axes_b == (-2, -1) and np.ndim(a) == np.ndim(b) + 1:
        return jnp.concatenate([(a[..., i] @ b)[..., jnp.newaxis] for i in range(a.shape[-1])], axis=-1)
    if axes_a == (-2, -1) and axes_b == (-3, -2) and np.ndim(b) == np.ndim(a) + 1:
        return jnp.concatenate([(a @ b[..., i])[..., jnp.newaxis] for i in range(b.shape[-1])], axis=-1)
    raise NotImplementedError


def matsort(mat: ArrayLike, order_j: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = np.asarray(mat)

    def idx2interval(idx):
        idx = np.sort(idx)
        interval = np.zeros((len(idx) - 1, 2), dtype=int32)
        interval[:, 0] = idx[:-1]
        interval[:, 1] = idx[1:]
        return interval

    n, m = mat.shape

    if order_j is None:
        order_j = np.argsort(np.sum(mat, axis=0))[::-1]

    order_i = np.argsort(mat[:, order_j[0]])[::-1]

    interval_idx = np.zeros(2, dtype=int32)
    interval_idx[1] = n
    for i in range(0, m - 1):
        interval_idx = np.unique(np.hstack([interval_idx,
                                            find.get_interval_indices(mat[order_i, order_j[i]]).ravel()]))

        for j, il in enumerate(idx2interval(idx=interval_idx)):
            slice_j = slice(il[0], il[1])
            if j % 2 == 0:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])[::-1]]
            else:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])]

    return mat[order_i, :][:, order_j], order_i, order_j


def idx2boolmat(idx: ArrayLike, n: int = 100) -> np.ndarray:
    s = np.shape(idx)[:-1]

    mat = np.zeros(s + (n,), dtype=bool)

    for i, idx_i in enumerate(np.asarray(idx).reshape(-1, np.shape(idx)[-1])):
        print(i, np.unravel_index(i, shape=s))
        mat[np.unravel_index(i, shape=s)][idx_i] = True
    return mat


def construct_array(shape: ShapeLike,
                    val: ArrayLike,
                    idx: ArrayLike,
                    init_mode: str = "zeros",
                    dtype: Any = None,
                    axis: AxisLike = None,
                    insert_mode: str | None = None) -> jax.Array:
    a = initialize_array(shape=shape, mode=init_mode, dtype=dtype)
    out = basics.insert(a=a, val=val, idx=idx, axis=axis, mode=insert_mode)
    return a if out is None else out


def block_view(a: ArrayLike,
               shape: ShapeLike,
               aslist: bool = False,
               require_aligned_blocks: bool = True) -> np.ndarray | list[np.ndarray]:
    a = np.asarray(a)
    assert a.flags["C_CONTIGUOUS"], "This function relies on the memory layout of the array."
    shape = tuple(shape)
    outershape = tuple(np.array(a.shape) // shape)
    view_shape = outershape + shape

    if require_aligned_blocks:
        assert np.all(np.mod(a.shape, shape) == 0), (
            f"blockshape {shape} must divide evenly into array shape {a.shape}")

    intra_block_strides = a.strides
    inter_block_strides = tuple(a.strides * np.array(shape))
    view = np.lib.stride_tricks.as_strided(a, shape=view_shape, strides=(inter_block_strides + intra_block_strides))

    if aslist:
        return [view[idx] for idx in np.ndindex(outershape)]
    return view


def expand_block_indices(idx_block: ArrayLike, block_size: int, squeeze: bool = True) -> np.ndarray:
    idx_block = np.atleast_1d(idx_block)
    if np.size(idx_block) == 1:
        ii = int(np.asarray(idx_block).item())
        return np.arange(block_size * ii, block_size * (ii + 1))

    idx2 = np.array([expand_block_indices(i, block_size=block_size, squeeze=squeeze) for i in idx_block])
    if squeeze:
        return idx2.flatten()
    return idx2


def replace(arr: ArrayLike,
            r_dict: dict[Any, Any],
            copy: bool = True,
            dtype: Any = None) -> np.ndarray | None:
    arr_np = np.asarray(arr)
    if copy:
        arr2 = arr_np.copy()
        if dtype is not None:
            arr2 = arr2.astype(dtype=dtype)
        for key in r_dict:
            arr2[arr_np == key] = r_dict[key]
        return arr2

    for key in r_dict:
        arr_np[arr_np == key] = r_dict[key]
    return None


def replace_tail_roll(a: ArrayLike, b: ArrayLike) -> np.ndarray:
    n_a, n_b = np.shape(a)[0], np.shape(b)[0]
    assert n_a > n_b

    a_np = np.array(a)
    a_np[-n_b:] = b
    return np.roll(a_np, n_b, axis=0)


def replace_tail_roll_list(arr_list: list[ArrayLike], arr_new_list: list[ArrayLike]):
    assert len(arr_list) == len(arr_new_list)
    return (replace_tail_roll(a=arr, b=arr_new) for (arr, arr_new) in zip(arr_list, arr_new_list))


def diag_wrapper(x: ArrayLike, n: int | None = None) -> jax.Array:
    x = jnp.asarray(x)
    if n is None:
        n = x.shape[0]

    if np.all(np.shape(x) == (n, n)):
        return x

    d = jnp.eye(n, dtype=float32)
    d = d.at[jnp.arange(n), jnp.arange(n)].set(x)
    return d


def create_constant_diagonal(n: int, m: int, v: ArrayLike, k: int) -> jax.Array:
    diag = jnp.eye(N=n, M=m, k=k) * v[0]
    for i in range(1, len(v)):
        diag = diag + jnp.eye(N=n, M=m, k=k + i) * v[i]
    return diag


def banded_matrix(v_list: list[ArrayLike], k0: int) -> jax.Array:
    m = jnp.diag(jnp.asarray(v_list[0]), k=k0)
    for i, v in enumerate(v_list[1:], start=1):
        m = m + jnp.diag(jnp.asarray(v), k=k0 + i)
    return m


def get_stats(x: ArrayLike, axis: AxisLike = None, return_array: bool = False) -> dict[str, Any] | jax.Array:
    x = jnp.asarray(x)
    stats = {
        "size": int(np.size(np.asarray(x), axis=axis)),
        "mean": jnp.mean(x, axis=axis),
        "std": jnp.std(x, axis=axis),
        "median": jnp.median(x, axis=axis),
        "min": jnp.min(x, axis=axis),
        "max": jnp.max(x, axis=axis),
    }

    if return_array:
        return jnp.array([stats["size"], stats["mean"], stats["std"], stats["median"], stats["min"], stats["max"]],
                         dtype=float32)

    return stats


def verbose_reject_x(title: str, x: ArrayLike, b: np.ndarray) -> np.ndarray:
    if b.size == 0:
        mean = 0
    else:
        mean = b.mean()
    print(f"{title}: {b.sum()}/{b.size} ~ {np.round(mean * 100, 3)}%")
    return np.asarray(x)[b].copy()


def get_points_inbetween(x: ArrayLike, extrapolate: bool = False) -> np.ndarray:
    x = np.asarray(x)
    assert x.ndim == 1

    delta = x[1:] - x[:-1]
    x_new = np.zeros(np.size(x) + 1, dtype=float32)
    x_new[1:-1] = x[:-1] + delta / 2
    if extrapolate:
        x_new[0] = x_new[1] - delta[0]
        x_new[-1] = x_new[-2] + delta[-1]
        return x_new
    return x_new[1:-1]


from wzk.np2 import np2 as _np_np2  # noqa: E402


def __getattr__(name: str) -> Any:
    return getattr(_np_np2, name)
