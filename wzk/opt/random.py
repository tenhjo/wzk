from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wzk import mp2
from wzk.logger import setup_logger

logger = setup_logger(__name__)


def random_ball_search(fun: Callable[[np.ndarray], float],
                       n_outer: int, n_inner: int,
                       x0: np.ndarray,
                       eps: float = 1e-9,
                       n_processes: int = 1,
                       log_level: int = 0) -> tuple[np.ndarray, float]:
    x0 = x0.ravel()
    n_x = len(x0)

    x_outer = np.zeros((n_outer+1, n_x))
    x_outer[0] = x0
    o_outer = np.zeros(n_outer+1)
    o_outer[0] = fun(x0)

    n_termination = 20
    radius = 0.1

    def fun_loop(x_list: np.ndarray) -> np.ndarray:
        o_list = np.zeros(len(x_list))
        for l, x in enumerate(x_list):
            o_list[l] = fun(x)
        return o_list

    for i in range(n_outer):

        delta = (np.random.random((n_inner, n_x)) - 0.5) * radius

        x_inner = x_outer[i]+delta

        o_inner = mp2.mp_wrapper(x_inner, fun=fun_loop, n_processes=n_processes)

        j = np.argmin(o_inner)
        if o_inner[j] < o_outer[i]:
            o_outer[i+1] = o_inner[j]
            x_outer[i+1] = x_inner[j]
            radius = np.linalg.norm(delta[j]) * 1.5
        else:
            o_outer[i+1] = o_outer[i]
            x_outer[i+1] = x_outer[i]
            radius = radius * 0.75

        if i > n_termination and np.mean(o_outer[i-n_termination:i] - o_outer[i+1]) < eps:
            o_outer[-1] = o_outer[i]
            x_outer[-1] = x_outer[i]
            break

        if log_level > 1:
            logger.debug("iteration: %d | objective: %.5g", i, o_outer[i+1])

    if log_level == 1:
        logger.debug("iteration: %d | objective: %.5g", n_outer, o_outer[-1])

    return x_outer[-1], o_outer[-1]
