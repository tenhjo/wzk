"""Weighted Least Squares"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import least_squares

from wzk import mp2


def nl_wls(
    fun_delta: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    w_sqrt: np.ndarray | None = None,
    max_nfev: int = 100,
    log_level: int = 0,
) -> np.ndarray:
    """non-linear weighted least squares
    == min_x (y - f(beta, x))' W (y - f(beta, x))
    """

    if w_sqrt is None:
        fun_delta2 = fun_delta

    else:
        # scipy.least_squares does not accept a weighting -> apply it's square root on dy
        #   x' A x == y'y
        #   with y = matrix_sqrt(A) x
        # A_sqrt = wzk.math2.matrix_sqrt(A)

        def fun_delta2(x: np.ndarray) -> np.ndarray:
            dy = fun_delta(x)
            return w_sqrt @ dy

    # is way quicker without setting bounds
    res = least_squares(fun=fun_delta2, x0=x0, log_level=log_level, max_nfev=max_nfev)

    return res.x


def nl_wls_mp(
    fun_delta: Callable[[np.ndarray], np.ndarray],
    x0_list: np.ndarray,
    w_sqrt: np.ndarray | None = None,
    n_processes: int = 1,
    log_level: int = 0,
) -> np.ndarray:
    # least_squares does not work with multiprocessing -> use joblib

    if np.ndim(x0_list) == 1:  # if only one x0 is provided
        return nl_wls(fun_delta, x0_list, w_sqrt=w_sqrt, log_level=log_level)

    def fun_delta_mp(x0_list2: np.ndarray) -> np.ndarray:
        x_list2 = np.zeros_like(x0_list2)
        for i, x0 in enumerate(x0_list2):
            x_list2[i, :] = nl_wls(fun_delta, x0, w_sqrt=w_sqrt, log_level=log_level)
        return x_list2

    n_samples = len(x0_list)
    n_samples_pp, n_samples_pp_cs = mp2.get_n_samples_per_process(n_samples=n_samples, n_processes=n_processes)

    arg_list = [
        x0_list[n_samples_pp_cs[i_process] : n_samples_pp_cs[i_process + 1]] for i_process in range(n_processes)
    ]

    x_list = Parallel(n_jobs=n_processes)(delayed(fun_delta_mp)(arg) for arg in arg_list)
    x_list = np.reshape(x_list, np.shape(x0_list))
    return x_list


def wls(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """(linear) weighted least squares
    == min (y - Beta x)' W (y - Beta x)
    """

    if w is None:
        w = np.ones_like(x)

    xw = np.swapaxes(x, -1, -2) @ w
    xwx = xw @ x
    xwy = xw @ y
    beta = np.linalg.inv(xwx) @ xwy
    return beta


def wls_1d(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """
    linear weighted least squares - 1D

    == line interpolation
    == find the best fit for:
    beta_0 + x*beta_1 = y

    """
    if w is None:
        w = np.ones_like(x)

    w_sum = float(np.sum(w))
    xw_sum = np.sum(x * w) / w_sum
    yw_sum = np.sum(y * w) / w_sum

    beta_1 = np.sum(w * (x - xw_sum) * (y - yw_sum)) / float(np.sum(w * (x - xw_sum) ** 2))
    beta_0 = yw_sum - beta_1 * xw_sum

    return beta_0, beta_1


def __ih_combine_deltas_to_wls(delta: np.ndarray, weighting: np.ndarray | float | None) -> np.ndarray:
    if weighting is not None:
        if isinstance(weighting, (int, float)):
            weighting = np.eye(len(delta.ravel())) * weighting
        delta = weighting @ delta.ravel()
    return delta.ravel()


def combine_deltas_to_wls(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    weighting_a: np.ndarray | float | None = None,
    weighting_b: np.ndarray | float | None = None,
) -> np.ndarray:
    """"""
    #     min( a' A a + b(x)' B b(x) )
    # ==  min( c' C c)
    #     (a)' (A 0) (a)
    #     (b)' (0 B) (b)
    # ==   z'    C    z

    delta_a = __ih_combine_deltas_to_wls(delta=delta_a, weighting=weighting_a)
    delta_b = __ih_combine_deltas_to_wls(delta=delta_b, weighting=weighting_b)

    delta = np.hstack([delta_a, delta_b])
    return delta
