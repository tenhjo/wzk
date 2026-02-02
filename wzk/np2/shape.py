import numpy as np

from typing import Iterable


def axis_wrapper(axis, n_dim, invert=False):
    if axis is None:
        axis = np.arange(n_dim)

    axis = np.atleast_1d(axis)
    axis %= n_dim
    np.sort(axis)

    if invert:
        return tuple(np.setxor1d(np.arange(n_dim), axis).astype(int))
    else:
        return tuple(axis)


def shape_wrapper(shape: int | np.integer | Iterable[int] | None = None
                  ) -> tuple[int, ...]:
    """
    Note the inconsistent usage of shape / size as function arguments in numpy.
    https://stackoverflow.com/questions/44804965/numpy-size-vs-shape-in-function-arguments
    -> use shape
    """
    if shape is None:
        return ()

    elif isinstance(shape, (int, np.int_)):
        return int(shape),

    elif isinstance(shape, tuple):
        return shape

    elif isinstance(shape, (list, np.ndarray)):
        return tuple(shape)

    else:
        raise ValueError(f"Unknown 'shape': {shape}")


def get_max_shape(*args):
    shapes = [-1 if a is None else np.shape(a) for a in args]
    sizes = [np.prod(shape) for shape in shapes]
    return shapes[np.argmax(sizes)]


def get_subshape(shape, axis):
    return tuple(np.array(shape)[np.array(axis)])
