import numpy as np


def maximum_a_posteriori(delta_y, delta_theta, cov_y_inv, cov_theta_inv=None, sum_up=True):
    if cov_theta_inv is None:
        return dCd(delta_y, cov_y_inv, sum_up=sum_up)

    else:
        return dCd(delta_y, cov_y_inv, sum_up=sum_up) + dCd(delta_theta, cov_theta_inv, sum_up=sum_up)


def dCd(delta, C, sum_up=True):
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
