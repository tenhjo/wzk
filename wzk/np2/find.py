import numpy as np

from .basics import rolling_window, args2arrays


def find_subarray(a, b):
    """
    Find b in a
    Return the index where the overlap begins.

    # a = np.array((2, 3, 4, 3, 5, 1))
    # b = np.array((3, 4, 3))

    # a = np.array((27, 3))
    # b = np.array((3,))
    # -> array([1])
    """
    a, b = np.atleast_1d(a, b)

    window = len(b)
    a_window = rolling_window(a=a, window=window)
    idx = np.nonzero(np.sum(a_window == b, axis=-1) == window)[0]
    return idx


def find_values(arr, values):
    res = np.zeros_like(arr, dtype=bool)
    for v in values:
        res[~res] = arr[~res] == v
    return res


def find_common_values(a, b):
    """if there are multiple elements with the same value, the first one is taken"""
    i_a = []
    i_b = []
    for i, aa in a:
        for j, bb in b:
            if np.allclose(aa, bb):
                i_a.append(i)
                i_b.append(j)
                break
    return np.array(i_a, dtype=int), np.array(i_b, dtype=int)


def find_array_occurrences(a, o):
    assert a.ndim == o.ndim
    assert a.shape[-1] == o.shape[-1]

    if a.ndim == 2:
        b = a[:, np.newaxis, :] == o[np.newaxis, :, :]
        b = np.sum(b, axis=-1) == o.shape[-1]
        i = np.array(np.nonzero(b)).T

    else:
        raise ValueError

    return i


def get_element_overlap(arr1, arr2=None, verbose=0):
    """
    arr1 is a 2D array (n, m)
    arr2 is a 2D array (l, k)
    along the first dimension are different samples and the second dimension are different features of one sample

    return a 2D int array (n, l) where each element o, j shows how many of the elements
    of arr1[o] are also present in arr2[j], without regard of specific position in the arrays
    """
    if arr2 is None:
        arr2 = arr1

    overlap = np.zeros((len(arr1), len(arr2)), dtype=int)
    for i, arr_i in enumerate(arr1):
        if verbose > 0:
            print(f"{i} / {len(arr1)}")
        for j, arr_j in enumerate(arr2):
            for k in arr_i:
                if k in arr_j:
                    overlap[i, j] += 1

    return overlap


def get_first_row_occurrence(bool_arr):
    """

    array([[ True,  True, False,  True,  True,  True],
           [False,  True, False,  True,  True, False],
           [False, False,  True, False, False, False],
           [False, False, False, False, False, False]])

    -> array([ 0,  1,  2, -1])
    """
    nz_i, nz_j = np.nonzero(bool_arr)
    u, idx = np.unique(nz_i, return_index=True)
    res = np.full(bool_arr.shape[0], fill_value=-1)
    res[u] = nz_j[idx]
    return res


def fill_interval_indices(interval_list, n):
    if isinstance(interval_list, np.ndarray):
        interval_list = interval_list.tolist()

    if np.size(interval_list) == 0:
        return np.array([[1, n]])

    if interval_list[0][0] != 0:
        interval_list.insert(0, [0, interval_list[0][0]])

    if interval_list[-1][1] != n:
        interval_list.insert(len(interval_list), [interval_list[-1][1], n])

    i = 1
    while i < len(interval_list):
        if interval_list[i - 1][1] != interval_list[i][0]:
            interval_list.insert(i, [interval_list[i - 1][1], interval_list[i][0]])
        i += 1

    return np.array(interval_list)


def get_interval_indices(bool_array, expand=False):
    """
    Get list of start and end indices, which indicate the sections of True values in the array.

    Array is converted to bool first

    [0, 0, 0, 0]  ->  [[]]
    [0, 0, 0, 1]  ->  [[3, 4]]
    [0, 1, 1, 0]  ->  [[1, 3]]
    [1, 0, 0, 0]  ->  [[0, 1]]
    [1, 0, 0, 1]  ->  [[0, 1], [3, 4]]
    [1, 1, 0, 1]  ->  [[0, 2], [3, 4]]
    [1, 1, 1, 1]  ->  [[0, 4]]

    """
    bool_array = bool_array.astype(bool)
    assert bool_array.ndim == 1
    interval_list = np.where(np.diff(bool_array) != 0)[0] + 1
    if bool_array[0]:
        interval_list = np.concatenate([[0], interval_list])
    if bool_array[-1]:
        interval_list = np.concatenate([interval_list, bool_array.shape])
    interval_list = interval_list.reshape(-1, 2)

    if expand:
        interval_list = [list(range(i0, i1)) for (i0, i1) in interval_list]

    return interval_list


def get_cropping_indices(pos, shape_small, shape_big, mode="lower_left"):
    """
    Adjust the boundaries to fit small array in a larger array.
    pos:  idx where the small image should be set in the bigger picture, option A
    mode:  mode how to position the smaller array in the larger:
                  "center": pos describes the center of the small array inside the big array (shape_small must be odd)
                  "lower_left":
                  "upper_right":
    shape_small:  Size of the small image (=2*sm-1) in (number of pixels in each dimension)
    shape_big:  Size of the large image in (number of pixels in each dimension)
    :return:
    """

    shape_small, shape_big = args2arrays(shape_small, shape_big)

    if mode == "center":
        assert np.all(np.array(shape_small) % 2 == 1), shape_small
        shape_small2 = (np.array(shape_small) - 1) // 2

        ll_big = pos - shape_small2
        ur_big = pos + shape_small2 + 1

    elif mode == "lower_left":
        ll_big = pos
        ur_big = pos + shape_small

    elif mode == "upper_right":
        ll_big = pos - shape_small
        ur_big = pos

    else:
        raise ValueError(f"Invalid position mode {mode}")

    ll_small = np.where(ll_big < 0,
                        -ll_big,
                        0)
    ur_small = np.where(shape_big - ur_big < 0,
                        shape_small + (shape_big - ur_big),
                        shape_small)

    ll_big = np.where(ll_big < 0, 0, ll_big)
    ur_big = np.where(shape_big - ur_big < 0, shape_big, ur_big)

    return ll_big, ur_big, ll_small, ur_small


def find_closest(x, y):
    d = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    i_x = np.argmin(d, axis=0)
    i_y = np.argmin(d, axis=1)
    return i_x, i_y


def find_consecutives(x, n):
    if n == 1:
        return np.arange(len(x))
    assert n > 1
    return np.nonzero(np.convolve(np.abs(np.diff(x)), v=np.ones(n - 1), mode="valid") == 0)[0]


def find_largest_consecutives(x):
    i2 = np.nonzero(np.convolve(np.abs(np.diff(x)), v=np.ones(2 - 1), mode="valid") == 0)[0]
    i2 -= np.arange(len(i2))
    _, c2 = np.unique(i2, return_counts=True)
    if c2.size == 0:
        n = 1
    else:
        n = c2.max() + 1

    return n, find_consecutives(x, n=n)


def find_block_shuffled_order(a, b, block_size, threshold, verbose=1):
    n = len(a)
    m = len(b)
    assert n == m
    assert n % block_size == 0

    nn = n // block_size
    idx = np.empty(nn)

    for i in range(nn):
        for j in range(nn):
            d = (a[i * block_size:(i + 1) * block_size] -
                 b[j * block_size:(j + 1) * block_size])

            d = np.abs(d).max()
            if d < threshold:
                idx[i] = j
                if verbose > 0:
                    print(i, j, d)

    return idx


# uses
def align_shapes(a, b):
    """
    a = np.array((2, 3, 4, 3, 5, 1))
    b = np.array((3, 4, 3))
    -> array([-1, 1, 1, 1, -1, -1])
    """
    idx = find_subarray(a=a, b=b).item()
    aligned_shape = np.full(shape=len(a), fill_value=-1, dtype=int)
    aligned_shape[idx:idx + len(b)] = 1
    return aligned_shape
