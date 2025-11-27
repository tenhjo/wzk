import numpy as np
from itertools import combinations, permutations


def get_tuples(n: np.ndarray | list | tuple | int, m: int):
    if isinstance(n, (list, tuple, np.ndarray)):
        x = np.array(n)
        n = len(x)
    else:
        assert isinstance(n, int)
        x = np.arange(n)

    pairs = np.array(list(combinations(np.arange(n), m)))
    pairs = x[pairs]
    return pairs


def __unique_permutations(iterable, r=None):
    previous = tuple()
    for p in permutations(sorted(iterable), r):
        if p > previous:
            previous = p
            yield p


def unique_permutations(iterable, r=None):
    return np.array(list(__unique_permutations(iterable, r)))
