from __future__ import annotations

import numpy as np

from wzk import np2


def initialize_frames(
    shape: int | tuple[int, ...], n_dim: int, mode: str = "hm", dtype: type | None = None, order: str | None = None
) -> np.ndarray:
    """Beware of the n_dim+1 coming from homogenous transformation"""
    f = np.zeros((np2.shape_wrapper(shape) + (n_dim + 1, n_dim + 1)), dtype=dtype, order=order)
    if mode == "zero":
        pass
    elif mode == "eye":
        for i in range(f.shape[-1]):
            f[..., i, i] = 1
    elif mode == "hm":
        f[..., -1, -1] = 1
    else:
        raise ValueError(f"Unknown mode '{mode}'")
    return f


def fill_frames_trans(f: np.ndarray, trans: np.ndarray | None = None) -> None:
    if trans is not None:
        f[..., :-1, -1] = trans
