import numpy as np
from wzk import np2


def x2limits(x, axis=-1):
    axis = np2.axis_wrapper(axis=axis, n_dim=x.ndim, invert=True)
    limits = np.stack([x.min(axis=axis), x.max(axis=axis)], axis=1)
    return limits


def limits2size(limits):
    return limits[:, 1] - limits[:, 0]


def limits2center(limits):
    s2 = limits2size(limits)/2
    return limits[:, 0] + s2


def spheres2limits(x, r):
    x = np.atleast_2d(x)
    r = np.atleast_1d(r)

    lower = x - r[:, np.newaxis]
    upper = x + r[:, np.newaxis]

    lower = np.min(lower, axis=0)
    upper = np.max(upper, axis=0)

    limits = np.array([lower, upper]).T
    return limits


def combine_limits(limits_a, limits_b, mode="largest"):
    if mode == "largest":
        lower = np.minimum(limits_a[:, 0], limits_b[:, 0])
        upper = np.maximum(limits_a[:, 1], limits_b[:, 1])
    elif mode == "smallest":
        lower = np.maximum(limits_a[:, 0], limits_b[:, 0])
        upper = np.minimum(limits_a[:, 1], limits_b[:, 1])
    else:
        raise ValueError

    limits = np.array([lower, upper]).T
    return limits


def make_limits_symmetrical(limits, mode="largest"):
    s2 = limits2size(limits=limits) / 2
    c = limits2center(limits)

    if mode == "largest":
        s2 = np.max(s2)
    elif mode == "smallest":
        s2 = np.min(s2)
    else:
        raise ValueError

    limits = limits.copy()
    limits[:, 0] = c - s2
    limits[:, 1] = c + s2
    return limits


def add_safety_limits(limits: np.ndarray, factor=None, offset=None):
    limits = np.asarray(limits)
    squeeze = limits.ndim == 1
    if squeeze:
        limits = limits[np.newaxis, :]

    s = limits2size(limits=limits)

    if offset is None:
        assert factor is not None
        offset = factor * s

    offset = np.asarray(offset)
    if offset.ndim == 0:
        offset = np.full_like(s, fill_value=float(offset), dtype=float)

    assert np.all(offset > -s/2)

    limits_safe = np.array([limits[..., 0] - offset,
                            limits[..., 1] + offset]).T
    if squeeze:
        return limits_safe[0]
    return limits_safe


# --- Use limits -------------------------------------------------------------------------------------------------------
def remove_outside_limits(x, limits, safety_factor=None, return_idx=False):
    if safety_factor is not None:
        limits = add_safety_limits(limits=limits, factor=safety_factor)

    below_lower = np.sum(x < limits[:, 0], axis=-1) > 0
    above_upper = np.sum(x > limits[:, 1], axis=-1) > 0
    outside_limits = np.logical_or(below_lower, above_upper)
    inside_limits = ~outside_limits
    x = x[inside_limits]
    if return_idx:
        return x, inside_limits
    return x
