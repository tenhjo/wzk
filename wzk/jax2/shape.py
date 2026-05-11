from __future__ import annotations

from typing import Any

import numpy as np

from ._types import AxisLike, ShapeLike, int32


def axis_wrapper(axis: AxisLike, n_dim: int, invert: bool = False) -> tuple[int, ...]:
    axis_arr = np.arange(n_dim, dtype=int32) if axis is None else np.atleast_1d(np.asarray(axis, dtype=int32))
    axis_arr %= n_dim
    axis_arr = np.sort(axis_arr)

    if invert:
        axis_inv = np.setxor1d(np.arange(n_dim, dtype=int32), axis_arr).astype(int32)
        return tuple(int(v) for v in axis_inv.tolist())
    return tuple(int(v) for v in axis_arr.tolist())


def shape_wrapper(shape: ShapeLike | None = None) -> tuple[int, ...]:
    """Normalize scalar/iterable shape arguments to a shape tuple."""
    if shape is None:
        return ()
    if isinstance(shape, (int, np.integer)):
        return (int(shape),)
    if isinstance(shape, tuple):
        return tuple(int(s) for s in shape)
    if isinstance(shape, (list, np.ndarray)):
        return tuple(int(s) for s in shape)
    raise ValueError(f"Unknown 'shape': {shape}")


def get_max_shape(*args: Any) -> tuple[int, ...] | int:
    shapes = [-1 if a is None else np.shape(a) for a in args]
    sizes = [np.prod(shape) for shape in shapes]
    return shapes[int(np.argmax(sizes))]


def get_subshape(shape: ShapeLike, axis: AxisLike) -> tuple[int, ...]:
    shape_np = np.asarray(shape)
    axis_tuple = axis_wrapper(axis=axis, n_dim=shape_np.size)
    return tuple(shape_np[np.array(axis_tuple)])
