from __future__ import annotations

from typing import Any

import numpy as np

from .basics import max_size, scalar2array
from ._types import ArrayLike, ShapeLike, int32


def arangen(start: ArrayLike | ShapeLike | None = None,
            end: ArrayLike | ShapeLike | None = None,
            step: ArrayLike | ShapeLike | None = None) -> list[np.ndarray]:
    """N-dimensional arange."""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return [np.arange(start[i], end[i], step[i]) for i in range(n)]


def arange_between(i: ArrayLike, n: int) -> np.ndarray:
    i = np.hstack([np.array(i, dtype=int32), [n]])
    if i[0] != 0:
        i = np.insert(i, 0, 0)

    j = np.zeros(n, dtype=int32)
    for v, (i0, i1) in enumerate(zip(i[:-1], i[1:])):
        j[i0:i1] = v

    return j


def slicen(start: ArrayLike | ShapeLike | None = None,
           end: ArrayLike | ShapeLike | None = None,
           step: ArrayLike | ShapeLike | None = None) -> tuple[slice, ...]:
    """N-dimensional slice tuple."""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return tuple(map(slice, start, end, step))


def range2slice(r: range) -> slice:
    return slice(r.start, r.stop, r.step)


def slice2range(s: slice) -> range:
    return range(0 if s.start is None else s.start,
                 s.stop,
                 1 if s.step is None else s.step)


def __slice_or_range2tuple(sor: slice | range | int | tuple[int, ...] | None,
                           type2: str) -> tuple[Any, Any, Any]:
    if type2 == "slice":
        default1, default2, default3 = None, None, None
    elif type2 == "range":
        default1, default2, default3 = 0, 1, 1
    else:
        raise ValueError(f"Unknown {type2}")

    if isinstance(sor, (slice, range)):
        return sor.start, sor.stop, sor.step
    if isinstance(sor, int):
        return default1, sor, default3
    if sor is None:
        return default1, default2, default3
    if isinstance(sor, tuple):
        if len(sor) == 1:
            return default1, sor[0], default3
        if len(sor) == 2:
            return sor[0], sor[1], default3
        if len(sor) == 3:
            return sor[0], sor[1], sor[2]
        raise ValueError("tuple must be have length={1, 2, 3}")
    raise TypeError("r must be {slice, range, int, tuple}")


def slice2tuple(s: slice | int | tuple[int, ...] | None) -> tuple[Any, Any, Any]:
    return __slice_or_range2tuple(s, "slice")


def range2tuple(r: range | int | tuple[int, ...] | None) -> tuple[int, int, int]:
    return __slice_or_range2tuple(r, "range")


def slice_add(a: slice, b: slice) -> slice:
    a_t = slice2tuple(a)
    b_t = slice2tuple(b)
    return slice(a_t[0] + b_t[0], a_t[1] + b_t[1], max(a_t[2], b_t[2]))


def range_add(a: range, b: range) -> range:
    a_t = range2tuple(a)
    b_t = range2tuple(b)
    return range(a_t[0] + b_t[0], a_t[1] + b_t[1], max(a_t[2], b_t[2]))
