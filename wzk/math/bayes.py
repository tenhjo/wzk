from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def maximum_a_posteriori(
    delta_y: ArrayLike,
    delta_theta: ArrayLike,
    cov_y_inv: ArrayLike,
    cov_theta_inv: ArrayLike | None = None,
    sum_up: bool = True,
) -> np.ndarray:
    if cov_theta_inv is None:
        return dCd(delta_y, cov_y_inv, sum_up=sum_up)

    else:
        return dCd(delta_y, cov_y_inv, sum_up=sum_up) + dCd(delta_theta, cov_theta_inv, sum_up=sum_up)


def dCd(delta: ArrayLike, C: ArrayLike, sum_up: bool = True) -> np.ndarray:
    if np.ndim(delta) + 1 == C.ndim:
        cdc = delta[..., :, np.newaxis] * C @ delta[..., :, np.newaxis]

    elif np.ndim(delta) == C.ndim == 2:
        cdc = delta[:, :, np.newaxis] * C[np.newaxis, :, :] @ delta[:, :, np.newaxis]

    else:
        raise ValueError(f"unknown sizes {delta.shape} & {C.shape}")

    if sum_up:
        return np.sum(cdc)
    else:
        return cdc
