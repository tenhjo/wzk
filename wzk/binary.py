import numpy as np


def binary_table(a, b):
    a, b = np.atleast_1d(a, b)
    assert np.all(a.shape == b.shape)

    not_a = np.logical_not(a)
    not_b = np.logical_not(b)

    x = np.zeros((3, 3), dtype=int)

    x[0, 0] = np.count_nonzero(np.logical_and(not_a, not_b))
    x[0, 1] = np.count_nonzero(np.logical_and(not_a, b))
    x[0, 2] = np.count_nonzero(not_a)

    x[1, 0] = np.count_nonzero(np.logical_and(a, not_b))
    x[1, 1] = np.count_nonzero(np.logical_and(a, b))
    x[1, 2] = np.count_nonzero(a)

    x[2, 0] = np.count_nonzero(not_b)
    x[2, 1] = np.count_nonzero(b)
    x[2, 2] = b.size

    return x


def logical_or(*args):
    b = np.logical_or(args[0], args[1])

    for a in args[2:]:
        b = np.logical_or(b, a)
    return b
