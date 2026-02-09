from __future__ import annotations

from typing import Any

import numpy as np

from ._types import AxisLike, ShapeLike, int32


def axis_wrapper(axis: AxisLike, n_dim: int, invert: bool = False) -> tuple[int, ...]:
    if axis is None:
        axis = np.arange(n_dim)

    axis = np.atleast_1d(axis).astype(int32)
    axis %= n_dim
    axis = np.sort(axis)

    if invert:
        return tuple(np.setxor1d(np.arange(n_dim), axis).astype(int32))
    return tuple(axis)


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
    return tuple(np.array(shape)[np.array(axis)])
