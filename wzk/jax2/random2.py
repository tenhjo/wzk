from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import norm

from wzk import grid
from wzk import limits as limits2
from wzk.logger import setup_logger

from . import math2
from . import np2
from ._types import ArrayLike, ShapeLike, Scalar, float32, int32
from .basics import scalar2array
from .shape import shape_wrapper

logger = setup_logger(__name__)


def p_normal_skew(x: ArrayLike | Scalar,
                  loc: float = 0.0,
                  scale: float = 1.0,
                  a: float = 0.0) -> jax.Array:
    t = (np.asarray(x) - loc) / scale
    return jnp.asarray(2 * norm.pdf(t) * norm.cdf(a * t), dtype=float32)


def normal_skew_int(loc: float = 0.0,
                    scale: float = 1.0,
                    a: float = 0.0,
                    low: int | None = None,
                    high: int | None = None,
                    size: int | ShapeLike = 1) -> int | jax.Array:
    if low is None:
        low = int(loc - 10 * scale)
    if high is None:
        high = int(loc + 10 * scale + 1)

    p_max = float(np.asarray(p_normal_skew(x=loc, loc=loc, scale=scale, a=a)))
    n_samples = int(np.prod(size))
    samples = np.zeros(n_samples, dtype=int32)

    for i in range(n_samples):
        while True:
            x = np.random.randint(low=low, high=high)
            p = float(np.asarray(p_normal_skew(x=x, loc=loc, scale=scale, a=a)))
            if np.random.rand() <= p / p_max:
                samples[i] = x
                break

    if size == 1:
        return int(samples[0])
    return jnp.asarray(samples, dtype=int32)


def random_uniform_ndim(low: ArrayLike, high: ArrayLike, shape: ShapeLike | None = None) -> jax.Array:
    n_dim = np.shape(low)[0]
    out = np.random.uniform(low=low, high=high, size=shape_wrapper(shape) + (n_dim,))
    return jnp.asarray(out, dtype=float32)


def noise(shape: ShapeLike | None, scale: float, mode: str = "normal") -> jax.Array:
    shape = shape_wrapper(shape)

    if mode == "constant":  # Could argue that this is no noise.
        return jnp.full(shape=shape, fill_value=+scale, dtype=float32)
    if mode == "plusminus":
        out = np.where(np.random.random(shape) < 0.5, -scale, +scale)
        return jnp.asarray(out, dtype=float32)
    if mode == "uniform":
        out = np.random.uniform(low=-scale, high=+scale, size=shape)
        return jnp.asarray(out, dtype=float32)
    if mode == "normal":
        out = np.random.normal(loc=0.0, scale=scale, size=shape)
        return jnp.asarray(out, dtype=float32)
    raise ValueError(f"Unknown mode '{mode}'")


def get_n_in2(n_in: int,
              n_out: int,
              n_total: int,
              n_current: int,
              safety_factor: float = 1.01,
              max_factor: int = 128) -> int:
    if n_out == 0:
        n_in2 = n_in * 2
    else:
        n_in2 = (n_total - n_current) * n_in / n_out

    n_in2 = min(n_total * max_factor, n_in2)  # Otherwise it can grow up to 2**maxiter.
    n_in2 = max(int(np.ceil(safety_factor * n_in2)), 1)
    return n_in2


def fun2n(fun: Callable[[int], ArrayLike],
          n: int,
          max_iter: int = 100,
          max_factor: int = 128) -> jax.Array:
    """
    Wrapper to repeatedly call fun(n_i) and concatenate outputs until len(x) >= n.
    """
    x = np.asarray(fun(n))
    x_new = x

    n_in = n
    for i in range(max_iter):
        n_in = get_n_in2(n_in=n_in, n_out=len(x_new), n_total=n, n_current=len(x), max_factor=max_factor)

        x_new = np.asarray(fun(n_in))
        x = np.concatenate([x, x_new], axis=0)

        logger.debug("%s: total:%s | current:%s | new:%s/%s", i, n, len(x), len(x_new), n_in)

        if len(x) >= n:
            return jnp.asarray(x[:n])

    Warning(f"Maximum number of iterations reached! Only {len(x)} samples could be generated")
    return jnp.asarray(x)


def choose_from_sections(n_total: int,
                         n_sections: int,
                         n_choose_per_section: int | ArrayLike,
                         flatten: bool = True) -> jax.Array | list[np.ndarray]:
    n_i = np.array_split(np.arange(n_total), n_sections)

    n_choose_per_section = scalar2array(n_choose_per_section, shape=n_sections)
    i = [np.random.choice(arr, size=int(m)) for arr, m in zip(n_i, n_choose_per_section)]
    if flatten:
        return jnp.asarray(np.concatenate(i, axis=0), dtype=int32)
    return i


def choose_from_uniform_grid(x: ArrayLike, n: int) -> jax.Array:
    x = np.asarray(x)
    _, n_dim = x.shape

    limits = limits2.x2limits(x=x, axis=1)
    limits = limits2.make_limits_symmetrical(limits=limits)

    def fun(_s: float) -> int:
        _shape = (int(_s),) * n_dim
        _i = grid.x2i(x=x, limits=limits, shape=_shape)
        _u = np.unique(_i, axis=0)
        return len(_u) - n

    s = math2.bisection(f=fun, a=2, b=100, tol=0)
    shape = (int(np.ceil(s)),) * n_dim

    ix = grid.x2i(x=x, limits=limits, shape=shape)
    u, inv = np.unique(ix, axis=0, return_inverse=True)
    iu = np.random.choice(np.arange(len(u)), n, replace=False)

    i = [np.random.choice(np.nonzero(np.asarray(inv == j, dtype=bool))[0]) for j in iu]
    return jnp.asarray(np.array(i, dtype=int32), dtype=int32)


def block_shuffle(arr: ArrayLike | int, block_size: int, inside: bool = False) -> jax.Array:
    """
    Shuffle along the first dimension.
    If block_size > 1, keep block_size elements together and shuffle blocks.
    """
    if isinstance(arr, int):
        n = arr
        arr_np = np.arange(n)
    else:
        arr_np = np.asarray(arr).copy()
        n = arr_np.shape[0]

    if block_size == 1:
        np.random.shuffle(arr_np)
        return jnp.asarray(arr_np)

    assert block_size > 0
    assert isinstance(block_size, int)
    assert n % block_size == 0
    n_blocks = n // block_size

    if inside:
        idx = np.arange(n)
        for i in range(0, n, block_size):
            np.random.shuffle(idx[i:i + block_size])
        return jnp.asarray(arr_np[idx])

    idx_block = np.arange(n_blocks)
    np.random.shuffle(idx_block)
    idx_ele = np2.expand_block_indices(idx_block=idx_block, block_size=block_size, squeeze=True)
    return jnp.asarray(arr_np[idx_ele])
