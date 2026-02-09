from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import convolve2d

from wzk import ltd

from ._types import ArrayLike, ShapeLike, float32
from .shape import shape_wrapper
from .basics import scalar2array


def tile_offset(a: ArrayLike,
                reps: int | tuple[int, ...],
                offsets: ArrayLike | None = None) -> jax.Array:
    """Tile array and add per-tile offsets."""
    a = jnp.asarray(a)
    s = shape_wrapper(a.shape)
    b = jnp.tile(a, reps)

    if offsets is not None:
        r = np.array(b.shape) // np.array(a.shape)
        if np.size(offsets) == 1:
            o = scalar2array(offsets, shape=len(s))
        else:
            o = np.asarray(offsets)

        assert len(o) == len(s)
        offsets = [jnp.repeat(jnp.arange(rr), ss) * oo for ss, rr, oo in zip(s, r, o)]
        b = b + sum(jnp.meshgrid(*offsets, indexing="ij"))
    return b


def tile_2d(*,
            pattern: ArrayLike,
            v_in_row: int,
            v_to_next_row: tuple[int, int],
            offset: tuple[int, int] = (0, 0),
            shape: ShapeLike) -> jax.Array:
    nodes = np.zeros((shape[0] + v_to_next_row[0], shape[1] + v_in_row), dtype=float32)

    for ii, i in enumerate(range(0, nodes.shape[0], v_to_next_row[0])):
        nodes[i, range((ii * v_to_next_row[1]) % v_in_row, nodes.shape[1], v_in_row)] = 1

    img = convolve2d(nodes, pattern, mode="full")

    ll = (v_to_next_row[0] + offset[0],
          v_to_next_row[1] + offset[1])

    img = img[ll[0]:ll[0] + shape[0],
              ll[1]:ll[1] + shape[1]]
    return jnp.asarray(img)


def block_collage(*,
                  img_arr: ArrayLike,
                  inner_border: tuple[int, int] | None = None,
                  outer_border: tuple[int, int] | None = None,
                  fill_boarder: float | int = 0,
                  dtype=float32) -> jax.Array:
    img_arr = np.asarray(img_arr)
    assert img_arr.ndim == 4
    n_rows, n_cols, n_x, n_y = img_arr.shape

    bv_i, bh_i = ltd.tuple_extract(inner_border, default=(0, 0), mode="repeat")
    bv_o, bh_o = ltd.tuple_extract(outer_border, default=(0, 0), mode="repeat")

    img = np.full(shape=(n_x * n_rows + bv_i * (n_rows - 1) + 2 * bv_o,
                         n_y * n_cols + bh_i * (n_cols - 1) + 2 * bh_o),
                  fill_value=fill_boarder, dtype=dtype)

    for r in range(n_rows):
        for c in range(n_cols):
            img[bv_o + r * (n_y + bv_i):bv_o + (r + 1) * (n_y + bv_i) - bv_i,
                bh_o + c * (n_x + bh_i):bh_o + (c + 1) * (n_x + bh_i) - bh_i] = img_arr[r, c]

    return jnp.asarray(img)
