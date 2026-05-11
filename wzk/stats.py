import numpy as np


def get_iqr(x):
    q1, q3 = np.quantile(x.ravel(), q=[0.25, 0.75])
    iqr = q3 - q1
    return iqr


def get_fences(x):
    q1, q3 = np.quantile(x.ravel(), q=[0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def get_outliers(x, mode="fences"):

    if mode == "fences":
        lower, upper = get_fences(x)
        outliers = np.logical_or(x < lower, x > upper)
        return outliers

    else:
        raise ValueError
