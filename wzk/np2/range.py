import numpy as np

from .basics import max_size, scalar2array


def arangen(start=None, end=None, step=None):
    """n dimensional slice"""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return [np.arange(start[i], end[i], step[i]) for i in range(n)]


def arange_between(i, n):
    i = np.hstack([np.array(i, dtype=int), [n]])
    if i[0] != 0:
        i = np.insert(i, 0, 0)

    j = np.zeros(n, dtype=int)
    for v, (i0, i1) in enumerate(zip(i[:-1], i[1:])):
        j[i0:i1] = v

    return j


#  --- Slice, Ellipsis, Range ------------------------------------------------------------------------------------------
# Slice and range
def slicen(start=None, end=None, step=None):
    """n dimensional slice"""
    n = max_size(start, end, step)
    start, end, step = scalar2array(start, end, step, shape=n)
    return tuple(map(slice, start, end, step))  # noqa


def range2slice(r):
    return slice(r.start, r.stop, r.step)


def slice2range(s):
    return range(0 if s.start is None else s.start,
                 s.stop,
                 1 if s.step is None else s.step)


def __slice_or_range2tuple(sor, type2):
    if type2 == "slice":
        default1, default2, default3 = None, None, None
    elif type2 == "range":
        default1, default2, default3 = 0, 1, 1
    else:
        raise ValueError(f"Unknown {type2}")

    if isinstance(sor, (slice, range)):
        return sor.start, sor.stop, sor.step

    elif isinstance(sor, int):
        return default1, sor, default3

    elif sor is None:
        return default1, default2, default3

    elif isinstance(sor, tuple):
        if len(sor) == 1:
            return default1, sor[0], default3
        elif len(sor) == 2:
            return sor[0], sor[1], default3
        elif len(sor) == 3:
            return sor
        else:
            raise ValueError("tuple must be have length={1, 2, 3}")
    else:
        raise TypeError("r must be {slice, range, int, tuple}")


def slice2tuple(s):
    return __slice_or_range2tuple(s, "slice")


def range2tuple(r):
    return __slice_or_range2tuple(r, "range")


def slice_add(a, b):
    a = slice2tuple(a)
    b = slice2tuple(b)
    return slice(a[0] + b[0], a[1] + b[1], max(a[2], b[2]))


def range_add(a, b):
    a = range2tuple(a)
    b = range2tuple(b)
    return range(a[0] + b[0], a[1] + b[1], max(a[2], b[2]))
