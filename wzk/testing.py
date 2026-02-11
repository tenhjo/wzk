
from wzk.logger import log_print
import numpy as np

from wzk.np2 import allclose


def compare_arrays(a, b, axis=None, log_level=0,
                   rtol=1.e-5, atol=1.e-8,
                   title=""):
    eps = 1e-9
    all_equal = allclose(a=a, b=b, axis=axis, rtol=rtol, atol=atol)

    if not np.all(all_equal) or log_level > 0:
        log_print(title)
        log_print("shape: ", a.shape)
        log_print(f"nan: a {int(np.isnan(a).any())} b {int(np.isnan(b).any())}")
        log_print("maximal difference:", np.abs(a - b).max())
        log_print("variance difference:", np.std(a-b))
        ratio = a[np.abs(b) > eps] / b[np.abs(b) > eps]
        if np.count_nonzero(ratio) == 0:
            ratio = np.zeros(1)
        log_print("mean ratio:", ratio.mean())
        if not np.all(all_equal) and log_level >= 2:
            log_print(all_equal.astype(int))

    return np.all(all_equal)
