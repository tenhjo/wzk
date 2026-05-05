from __future__ import annotations

from collections.abc import Callable

import numpy as np

from wzk import math2, mp2, mpl2, printing
from wzk.logger import setup_logger

logger = setup_logger(__name__)


def idx_times_all(idx: list[int] | np.ndarray, n: int) -> np.ndarray:
    idx = np.atleast_2d(idx)
    idx2 = idx.repeat(n, axis=0)
    idx2 = np.hstack((idx2, np.tile(np.arange(n), reps=idx.shape[0])[:, np.newaxis])).astype(int)

    return idx2


def greedy(
    n: int,
    k: int,
    fun: Callable[[np.ndarray], np.ndarray],
    i0: list[int] | np.ndarray | None = None,
    log_level: int = 0,
) -> tuple[np.ndarray, float]:
    """
    choose k elements out of a set with size n
    fun(idx_list) measures how good the current choice is
    """

    if i0 is not None:
        s = np.array(i0).tolist()
    else:
        s = []

    for i in range(k):
        printing.progress_bar(i=i, n=k, eta=True, log_level=log_level)
        idx_i = idx_times_all(idx=s, n=n)
        o = fun(idx_i)
        o[s] = np.inf
        s.append(np.argmin(o))

    # best = np.sort(best)
    s = np.array(s, dtype=int)
    o = fun(s[np.newaxis, :])[0]
    if log_level > 1:
        logger.debug("set: %r | objective: %s", s, o)
    return s, o


def detmax(
    fun: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | None = None,
    n: int = 100,
    k: int = 30,
    excursion: int = 10,
    method: str = "add->remove",
    max_loop: int = 3,
    log_level: int = 0,
) -> tuple[np.ndarray, float]:
    """
    method:  'add->remove'
             'remove->add'
    """

    improvement_threshold = 1e-2
    if x0 is None:
        x0 = greedy(n=n, k=k, fun=fun, log_level=log_level - 1)
        # x0 = math2.random_subset(n=n, k=k, m=1, dtype=np.int16)[0]

    def __add(x: np.ndarray, nn: int) -> tuple[np.ndarray, float]:
        x = idx_times_all(idx=x, n=nn)
        oo = fun(x)
        oo[x[0, :-1]] = np.inf
        idx_min = np.argmin(oo)
        # idx_min = np.random.choice(np.argsort(oo)[:nn//10])
        oo = oo[idx_min]
        x = x[idx_min]
        return x, oo

    def remove(x: np.ndarray, exc: int) -> tuple[np.ndarray, float]:
        oo = None
        for _ in range(1, exc + 1):
            x = np.repeat([x], repeats=len(x), axis=0)
            x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x) - 1)

            oo = fun(x)
            idx_min = np.argmin(oo)
            # idx_min = np.random.choice(np.argsort(oo)[:100//10])

            oo = oo[idx_min]
            x = x[idx_min]

        return np.sort(x), oo

    def add(x: np.ndarray, nn: int, exc: int) -> tuple[np.ndarray, float]:
        oo = None
        for _ in range(1, exc + 1):
            x, oo = __add(x=x, nn=nn)

        return np.sort(x), oo

    def addremove(x: np.ndarray, nn: int, exc: int) -> tuple[np.ndarray, float]:
        x = np.repeat([x], repeats=len(x), axis=0)
        x = x[np.logical_not(np.eye(len(x), dtype=bool))].reshape(len(x), len(x) - 1)

        x, oo = __add(x=x, nn=nn)
        return np.sort(x), oo

    o = np.inf
    for q in range(1, excursion + 1):
        for i in range(max_loop):  # noqa: B007
            o_old = o
            if method == "add->remove":
                x0, o = add(x=x0, nn=n, exc=q)
                x0, o = remove(x=x0, exc=q)
            elif method == "remove->add":
                x0, o = remove(x=x0, exc=q)
                x0, o = add(x=x0, nn=n, exc=q)
            elif method == "both":
                raise NotImplementedError()
            else:
                raise ValueError("Unknown method, see doc string for more information")

            if o_old - o < improvement_threshold:
                break

        if log_level >= 2:
            logger.debug("Depth: %d | Loop %d | Objective: %.4g | Configuration: %s", q, i + 1, o, x0)

    if log_level >= 1:
        logger.debug("Objective: %.4g | Configuration: %s", o, x0)
    return x0, o


def random(
    n: int,
    k: int,
    m: int,
    fun: Callable[[np.ndarray], np.ndarray],
    chunk: int = 1000,
    n_processes: int = 10,
    dtype: type = np.uint8,
    log_level: int = 0,
) -> tuple[np.ndarray, np.ndarray]:

    def fun2(_m: int) -> tuple[np.ndarray, np.ndarray]:
        _idx = math2.random_subset(n=n, k=k, m=_m, dtype=dtype)
        _o = fun(_idx)
        return _idx, _o

    idx, o = mp2.mp_wrapper(m, fun=fun2, n_processes=n_processes, max_chunk_size=chunk)

    if log_level > 1:
        fig, ax = mpl2.new_fig()
        ax.hist(o, bins=100)

    i_sorted = np.argsort(o)
    o = o[i_sorted]
    idx = idx[i_sorted].astype(int)

    return idx, o


def ga(
    n: int, k: int, m: int, fun: Callable[[np.ndarray], np.ndarray], log_level: int, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    from wzk.alg.ga_kofn import kofn

    best, ancestors = kofn(n=n, k=k, fitness_fun=fun, pop_size=m, log_level=log_level, **kwargs)

    logger.debug(repr(best))
    return best, ancestors
