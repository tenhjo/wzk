import numpy as np


def combine_iterative_indices(n, idx_list):
    i = np.arange(n)
    for j in idx_list:
        i = np.delete(i, j)

    i2 = np.ones(n)
    i2[i] = 0
    i = np.nonzero(i2)
    return i
