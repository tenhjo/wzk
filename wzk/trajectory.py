"""
bee nomenclature
bee:   direct line between a and b
dbee:  difference to the bee-line
dbees: scaled difference to the bee-line
"""
# TODO move to mopla at some point

import numpy as np

from scipy.interpolate import UnivariateSpline

from wzk import printing, math2


def full2inner(x):
    return x[..., 1:-1, :].copy()


def inner2full(inner, start=None, end=None):
    n_samples = inner.shape[0]

    def __repeat(x):
        x = np.atleast_2d(x)
        if inner.ndim != 3:
            return x
        else:
            if x.ndim == 2:
                x = x[:, np.newaxis, :]

            return x.repeat(n_samples // x.shape[0], axis=0)

    if start is not None:
        start = __repeat(start)
        inner = np.concatenate([start, inner], axis=-2)

    if end is not None:
        end = __repeat(end)
        inner = np.concatenate([inner, end], axis=-2)

    return inner


def full2start_end(x, mode=""):
    if mode == "1":
        return x[..., [0, -1], :].copy()
    if mode == "20":
        return x[..., 0, :].copy(), x[..., -1, :].copy()
    elif mode == "21":
        return x[..., :1, :].copy(), x[..., -1:, :].copy()
    else:
        raise NotImplementedError


def x2se(x):
    return x[..., [0, -1], :]


def path_mode(x, mode=None):
    if mode is None or mode == "full":
        return x
    elif mode == "inner":
        return x[..., 1:-1, :]
    elif mode == "wo_start":
        return x[..., 1:, :]
    elif mode == "wo_end":
        return x[..., :-1, :]
    else:
        raise ValueError


def flat2full(flat, n_dof):
    if flat.ndim == 1:
        flat = flat.reshape(-1, n_dof)
    elif flat.ndim == 2:
        flat = flat.reshape(len(flat), -1, n_dof)
    return flat


def full2flat(x):
    if x.ndim == 2:
        x = x.reshape(-1)
    elif x.ndim == 3:
        x = x.reshape(len(x), -1)
    return x


# ----------------------------------------------------------------------------------------------------------------------
# Substeps
def periodic_dof_wrapper(x,
                         is_periodic=None):
    if is_periodic is not None and any(is_periodic):
        x[..., is_periodic] = math2.angle2minuspi_pluspi(x[..., is_periodic])
    return x


def get_steps(q,
              is_periodic=None):
    return periodic_dof_wrapper(np.diff(q, axis=-2), is_periodic=is_periodic)


def get_steps_norm(q,
                   is_periodic=None):
    return np.linalg.norm(get_steps(q=q, is_periodic=is_periodic), axis=-1)


def get_substeps(x: np.ndarray, n: int,
                 is_periodic=None, include_start: bool = True):

    *shape, m, d = x.shape

    # only fill in substeps if the number is greater 1,
    if n <= 1 or m <= 1:
        if include_start:
            return x
        else:  # How often does this happen? once?: fill_linear_connection
            return x[..., 1:, :]

    steps = get_steps(q=x, is_periodic=is_periodic)
    delta = (np.arange(n-1, -1, -1)/n) * steps[..., np.newaxis]
    x_ss = x[..., 1:, :, np.newaxis] - delta
    x_ss = np.swapaxes(x_ss, -2, -1)
    x_ss = x_ss.reshape(shape + [(m-1) * n, d])

    if include_start:
        x_ss = np.concatenate([x[..., :1, :], x_ss], axis=-2)

    x_ss = periodic_dof_wrapper(x_ss, is_periodic=is_periodic)
    return x_ss


def get_steps_between(start: np.ndarray, end: np.ndarray, n: int,
                      is_periodic=None):
    q = np.concatenate([start[..., np.newaxis, :], end[..., np.newaxis, :]], axis=-2)
    q = get_substeps(x=q, n=n-1, is_periodic=is_periodic, include_start=True)
    return q


def get_substeps_adjusted(x: np.ndarray, n: int,
                          is_periodic=None, weighting=None, enforce_equal_steps: bool = False):

    *shape, m, d = x.shape

    if m == 1:
        return np.ones(shape + [n, d]) * x

    if shape:
        shape = tuple(shape)
        x_n = np.zeros(shape+(n, d))
        for i in np.ndindex(*shape):
            x_n[i] = get_substeps_adjusted(x=x[i], n=n, is_periodic=is_periodic, weighting=weighting)
        return x_n

    m1 = m - 1
    steps = get_steps(q=x, is_periodic=is_periodic)

    if weighting is not None:
        steps *= weighting

    # Distribute the waypoints equally along the linear sequences of the initial path
    steps_length = np.linalg.norm(steps, axis=-1)
    if enforce_equal_steps:
        steps_length[:] = 1

    if np.sum(steps_length) == 0:
        relative_steps_length = np.full(m1, fill_value=1 / m1)
    else:
        relative_steps_length = steps_length / np.sum(steps_length)

    # Adjust the number of waypoints for each step to make the initial guess as equally spaced as possible
    n_sub_exact = relative_steps_length * (n - 1)
    n_sub = np.round(n_sub_exact).astype(int)

    # If the number of points do not match, change the substeps where the rounding was worst
    n_diff = (n-1) - np.sum(n_sub)
    if n_diff != 0:
        n_sub_acc = n_sub_exact - n_sub
        n_sub_acc = 0.5 + np.sign(n_diff) * n_sub_acc
        idx = np.argsort(n_sub_acc)[-np.abs(n_diff):]
        n_sub[idx] += np.sign(n_diff)

    n_sub_cs = np.hstack((0, n_sub.cumsum())) + 1

    # Add the linear interpolation between the waypoints - step by step for each dimension
    x_n = np.empty((n, d))
    x_n[0, :] = x[0, :].copy()
    for i in range(m1):
        x_n[n_sub_cs[i]:n_sub_cs[i + 1], :] = \
            get_substeps(x=x[i:i + 2, :], n=n_sub[i], is_periodic=is_periodic, include_start=False)

    x_n = periodic_dof_wrapper(x=x_n, is_periodic=is_periodic)
    return x_n


def get_path_adjusted(x: np.ndarray, n: int = None,
                      is_periodic=None, weighting=None, enforce_equal_steps: bool = False, __m: int = 5):
    n0 = x.shape[-2]
    if n is None:
        n = n0
    else:
        pass

    n = int(n)
    return get_substeps_adjusted(x=x, n=(n - 1) * (n0*__m) + 1,
                                 is_periodic=is_periodic, weighting=weighting,
                                 enforce_equal_steps=enforce_equal_steps)[..., ::(n0*__m), :]


def order_path(x, start=None, end=None, is_periodic=None, weights=1.):
    """
    Order the points given by 'x' [2d: (n, d)] according to a weighted Euclidean distance
    so that always the nearest point comes next.
    Start with the first point in the array and end with the last if 'x_start' or 'x_end' aren't given.
    """

    # Handle different input combinations
    d = None
    if start is None:
        start = x[..., 0, :]
        x = np.delete(x, 0, axis=-2)
    else:
        d = np.size(start)

    if x is None:
        n = 0
    else:
        n, d = x.shape

    if end is None:
        x_o = np.zeros((n + 1, d))
    else:
        x_o = np.zeros((n + 2, d))
        x_o[-1, :] = end.ravel()

    # Order the points, so that always the nearest is visited next, according to the Euclidean distance
    x_o[0, :] = start.ravel()
    for i in range(n):
        x_diff = np.linalg.norm(periodic_dof_wrapper(x - start, is_periodic=is_periodic) * weights, axis=-1)
        i_min = np.argmin(x_diff)
        x_o[1 + i, :] = x[i_min, :]
        start = x[i_min, :]
        x = np.delete(x, i_min, axis=-2)

    return x_o


def remove_duplicates(q, eps=1e-5, verbose=0):
    assert q.ndim == 2
    
    dq = q[1:] - q[:-1]
    dqn = np.linalg.norm(dq, axis=-1)
    b = np.concatenate([[True], dqn > eps], axis=0)
    q = q[b]
    
    if verbose > 0:    
        printing.print_stats_bool(b=b, name="keep only unique waypoints")
        
    return q


# ----------------------------------------------------------------------------------------------------------------------
# Bee
def x2bee(x, n_wp=None):
    x_se = x[..., [0, -1], :]
    if n_wp is None:
        n_wp = x.shape[-2]

    bee = get_substeps(x_se, n=n_wp-1, include_start=True)
    return bee


def x2dbee(x, n_wp=None):
    return x - x2bee(x=x, n_wp=n_wp)


def x2sdbee(x, n_wp=None, eps=1e-4):
    dbee = x2dbee(x, n_wp=n_wp)

    s = np.linalg.norm(x[..., -1, :] - x[..., 0, :], axis=-1, keepdims=True)[..., np.newaxis] + eps
    sdbee = dbee / s
    return sdbee


def dbee2x(dbee, se):
    n_wp = dbee.shape[-2]
    bee = x2bee(x=se, n_wp=n_wp)
    x = bee + dbee
    return x


def sdbee2x(sdbee, se, eps=1e-4):
    n_wp = sdbee.shape[-2]
    assert np.all(sdbee[..., [0, -1], :] == 0)
    bee = x2bee(x=se, n_wp=n_wp)
    s = np.linalg.norm(se[..., -1, :] - se[..., 0, :], axis=-1, keepdims=True)[..., np.newaxis] + eps
    x = bee + sdbee * s
    return x


def position2velocity(x, timestep):
    v = np.zeros_like(x)
    v[1:] = (x[..., 1:, :] - x[..., :-1, :]) / timestep
    return v


def position2acceleration(x, timestep):
    v = position2velocity(x=x, timestep=timestep)
    a = position2velocity(x=v, timestep=timestep)
    return a


# --- Splines ----------------------------------------------------------------------------------------------------------
#
def to_spline(x, n_c=4, start_end_mode=None):

    n_wp, n_dof = x.shape[-2:]
    xx = np.linspace(0, 1, n_wp)
    if np.ndim(x) == 2:
        c = np.zeros((n_c, n_dof))
        for i_d in range(n_dof):
            c[..., i_d] = get_spline_coeffs(x=xx, y=x[:, i_d], n=n_c)

    elif np.ndim(x) == 3:
        n = x.shape[0]

        c = np.zeros((n, n_c, n_dof))
        for i_n in range(n):
            for i_d in range(n_dof):
                c[i_n, ..., i_d] = get_spline_coeffs(x=xx, y=x[i_n, :, i_d], n=n_c)
    else:
        raise ValueError

    if start_end_mode is None:
        pass
    elif start_end_mode == "c->0":
        c = c[..., 1:-1, :]
    elif start_end_mode == "x->0":
        pass
    else:
        raise ValueError(f"Unknown start_end_mode='{start_end_mode}'")

    return c


def get_spline_coeffs(x, y, n=None, s=None):
    if n is None:
        spl = UnivariateSpline(x=x, y=y, s=s)
        return spl.get_coeffs()

    else:
        s = len(x)
        for i in range(100):
            c = get_spline_coeffs(x=x, y=y, s=s)
            if len(c) == n:
                return c

            if len(c) < n:
                s *= 0.75
            else:
                s *= 1.5

        raise ValueError


def set_spline_coeffs(spl, coeffs):
    data = spl._data  # noqa
    k, n = data[5], data[7]
    data[9][:n - k - 1] = np.ravel(coeffs)
    spl._data = data


def from_spline(c, n_wp, start_end_mode=None):
    xx = np.linspace(0, 1, n_wp)
    spl = UnivariateSpline(x=xx, y=xx, )

    if start_end_mode == "c->0":
        z = np.zeros_like(c[..., :1, :])
        c = np.concatenate([z, c, z], axis=-2)

    n_c, n_dof = c.shape[-2:]

    if np.ndim(c) == 2:
        x = np.zeros((n_wp, n_dof))
        for i_d in range(n_dof):
            set_spline_coeffs(spl, coeffs=c[:, i_d])
            x[:, i_d] = spl(xx)

    elif np.ndim(c) == 3:
        n = c.shape[0]

        x = np.zeros((n, n_wp, n_dof))
        for i_n in range(n):
            for i_d in range(n_dof):
                spl = UnivariateSpline(x=xx, y=x[i_n, :, i_d])
                set_spline_coeffs(spl, coeffs=c[i_n, :, i_d])
                x[i_n, :, i_d] = spl(xx)

    else:
        raise ValueError

    if start_end_mode == "x->0":
        x[..., [0, -1], :] = 0

    return x


def fromto_spline2(x, n_c=4, x_mode="sdbee", start_end_mode="x->0"):
    n_wp = x.shape[-2]

    if x_mode == "dbee":
        dx = x2dbee(x=x)
    elif x_mode == "sdbee":
        dx = x2sdbee(x=x)
    elif x_mode is None:
        dx = x
    else:
        raise ValueError(f"Unknown x_mode='{x_mode}'")

    c = to_spline(dx, n_c=n_c, start_end_mode=start_end_mode)
    dx_new = from_spline(c=c, n_wp=n_wp, start_end_mode=start_end_mode)

    if x_mode == "dbee":
        x_spline = dbee2x(dbee=dx_new, se=x2se(x=x))
    elif x_mode == "sdbee":
        x_spline = sdbee2x(sdbee=dx_new, se=x2se(x=x))
    elif x_mode is None:
        x_spline = dx_new
    else:
        raise ValueError(f"Unknown x_mode='{x_mode}'")

    x_spline = get_path_adjusted(x=x_spline, )
    return x_spline


# ----------------------------------------------------------------------------------------------------------------------
# DERIVATIVES
def d_substeps__dx(n, order=0):
    """
    Get the dependence of substeps ' on the outer way points (x).
    The substeps are placed linear between the waypoints.
    To prevent counting points double one step includes only one of the two endpoints
    This gives a symmetric number of steps but ignores either the start or the end in
    the following calculations.

         x--'--'--'--x---'---'---'---x---'---'---'---x--'--'--'--x--'--'--'--x
    0:  {>         }{>            } {>            } {>         }{>         }
    1:     {         <} {            <} {            <}{         <}{         <}

    Ordering of the waypoints into a matrix:
    0:
    s00 s01 s02 s03 -> step 0 (with start)
    s10 s11 s12 s13
    s20 s21 s22 s23
    s30 s31 s32 s33
    s40 s41 s42 s43 -> step 4 (without end)

    1: (shifting by one, and reordering)
    s01 s02 s03 s10 -> step 0 (without start)
    s11 s12 s13 s20
    s21 s22 s23 s30
    s31 s32 s33 s40
    s41 s42 s43 s50 -> step 4 (with end)


    n          # way points
    n-1        # steps
    n_s        # intermediate points per step
    (n-1)*n_s  # substeps (total)

    n_s = 5
    jac0 = ([[1., 0.8, 0.6, 0.4, 0.2],  # following step  (including way point (1.))
             [0., 0.2, 0.4, 0.6, 0.8]]) # previous step  (excluding way point (1.))

    jac1 = ([[0.2, 0.4, 0.6, 0.8, 1.],  # previous step  (including way point (2.))
             [0.8, 0.6, 0.4, 0.2, 0.]])  # following step  (excluding way point (2.))

    """

    if order == 0:
        jac = (np.arange(n) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]
    else:
        jac = (np.arange(start=n - 1, stop=-1, step=-1) / n)[np.newaxis, :].repeat(2, axis=0)
        jac[0, :] = 1 - jac[0, :]

    return jac


def combine_d_substeps__dx(d_dxs, n):
    # Combine the jacobians of the sub-way-points (joints) to the jacobian for the optimization variables
    if n <= 1:
        return d_dxs

    if d_dxs.ndim == 3:
        n_samples, n_wp_ss, n_dof = d_dxs.shape
        d_dxs = d_dxs.reshape(n_samples, n_wp_ss//n, n, n_dof)
        ss_jac = d_substeps__dx(n=n, order=1)[np.newaxis, np.newaxis, ..., np.newaxis]
        d_dx = np.einsum("ijkl, ijkl -> ijl", d_dxs, ss_jac[:, :, 0, :, :])
        d_dx[:, :-1, :] += np.einsum("ijkl, ijkl -> ijl", d_dxs[:, 1:, :, :], ss_jac[:, :, 1, :, :])
        return d_dx
    else:
        raise ValueError(f"{d_dxs.ndim}")
