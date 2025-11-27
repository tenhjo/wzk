from typing import Literal

import numpy as np
from wzk import dtypes2

from . import basics
from . import find
from . import reshape
from .range import slicen

np.core.arrayprint._line_width = 80  # noqa


class DummyArray:
    """Allows indexing but always returns the same 'dummy' value"""

    def __init__(self, arr, shape):
        self.arr = arr
        self.shape = shape

    def __assert_int(self, item, i):
        assert item in range(-self.shape[i], self.shape[i])

    def __assert_slice(self, item, i):
        pass

    def __assert_ellipsis(self, item, i):
        pass

    def __getitem__(self, item):
        if isinstance(item, int):
            self.__assert_int(item=item, i=0)
        elif isinstance(item, slice):
            self.__assert_slice(item=item, i=0)
        elif isinstance(item, type(...)):
            self.__assert_ellipsis(item=item, i=0)

        else:
            assert len(item) == len(self.shape), f"Incompatible index {item} for array with shape {self.shape}"
            for i, item_i in enumerate(item):
                if isinstance(item_i, int):
                    self.__assert_int(item=item_i, i=i)
                elif isinstance(item_i, slice):
                    self.__assert_slice(item=item_i, i=i)
                elif isinstance(item, type(...)):
                    self.__assert_ellipsis(item=item, i=i)
                else:
                    raise ValueError

        return self.arr


def initialize_array(shape: tuple | list | np.ndarray,
                     mode: str = "zeros",
                     dtype=None,
                     order: Literal["C", "F"] = "C"):
    if mode == "zeros":
        return np.zeros(shape, dtype=dtype, order=order)
    elif mode == "ones":
        return np.ones(shape, dtype=dtype, order=order)
    elif mode == "empty":
        return np.empty(shape, dtype=dtype, order=order)
    elif mode == "random":
        return np.random.random(shape).astype(dtype=dtype, order=order)
    else:
        raise ValueError(f"Unknown initialization method {mode}")


# Checks
def np_isinstance(o, c):
    """
    # Like isinstance if o is not np.ndarray
    np_isinstance(('this', 'that'), tuple)  # True
    np_isinstance(4.4, int)                 # False
    np_isinstance(4.4, float)               # True

    # else
    np_isinstance(np.ones(4, dtype=int), int)    # True
    np_isinstance(np.ones(4, dtype=int), float)  # False
    np_isinstance(np.full((4, 4), 'bert'), str)  # True
    """

    if isinstance(o, np.ndarray):
        c = (dtypes2.c2np[cc] for cc in c) if isinstance(c, tuple) else dtypes2.c2np[c]
        return isinstance(o.flat[0], c)

    else:
        return isinstance(o, c)


def delete_args(*args, i, axis=None):
    return tuple(np.delete(a, obj=i, axis=axis) for a in args)


# Combine
def interleave(arrays, axis=0, out=None):
    """
    https://stackoverflow.com/questions/5347065/interweaving-two-numpy-arrays
    """

    shape = list(np.asanyarray(arrays[0]).shape)
    if axis < 0:
        axis += len(shape)
    assert 0 <= axis < len(shape), "'axis' is out of bounds"
    if out is not None:
        out = out.reshape(shape[:axis + 1] + [len(arrays)] + shape[axis + 1:])
    shape[axis] = -1
    return np.stack(arrays, axis=axis + 1, out=out).reshape(shape)


# Functions
def digitize_group(x, bins, right=False):
    """
    https://stackoverflow.com/a/26888164/7570817
    Similar to scipy.stats.binned_statistic but just return the indices corresponding to each bin.
    Same signature as numpy.digitize()
    """
    from scipy.sparse import csr_matrix  # TODO

    idx_x = np.digitize(x=x, bins=bins, right=right)
    n, m = len(x), len(bins) + 1
    s = csr_matrix((np.arange(n), [idx_x, np.arange(n)]), shape=(m, n))
    return [group for group in np.split(s.data, s.indptr[1:-1])]


def sort_args(idx, *args):
    return [a[idx] for a in args]


def make_odd(arr):
    mo = (np.array(arr.shape) + 1) % 2
    arr_new = np.zeros(np.array(arr.shape) + mo, dtype=bool)
    arr_new[slicen(end=arr.shape)] = arr
    return arr_new


def convolve_2d(img, kernel):
    s = np.array(img.shape)
    ks = np.array(kernel.shape)
    assert np.all(ks % 2 == 1)

    ks2 = ks // 2
    out = np.zeros(s, float)
    for i0 in range(ks2[0], s[0] - ks2[0]):
        for i1 in range(ks2[1], s[1] - ks2[1]):
            out[i0, i1] = np.sum(img[i0-ks2[0]:i0+ks2[0]+1, i1-ks2[1]:i1+ks2[1]+1] * kernel)

    return out


def add_small2big(idx, small, big, mode_crop="center", mode_add="add"):
    """
    Insert a small array into a bigger array at the position 'idx'
    Assumption: all dimension of the small_img are odd, and idx indicates the center of the image,
    if this is not the case, there are zeros added at the end of each dimension to make the image shape odd
    """

    idx = reshape.flatten_without_last(idx)
    n_samples, n_dim = idx.shape
    ll_big, ur_big, ll_small, ur_small = find.get_cropping_indices(pos=idx, mode=mode_crop,
                                                                   shape_small=small.shape[-n_dim:],
                                                                   shape_big=big.shape)

    if small.ndim > n_dim:
        for ll_b, ur_b, ll_s, ur_s, s in zip(ll_big, ur_big, ll_small, ur_small, small):
            big[slicen(ll_b, ur_b)] += s[slicen(ll_s, ur_s)]
    else:
        for ll_b, ur_b, ll_s, ur_s in zip(ll_big, ur_big, ll_small, ur_small):

            try:
                if mode_add == "add":
                    big[slicen(ll_b, ur_b)] += small[slicen(ll_s, ur_s)]
                elif mode_add == "replace":
                    big[slicen(ll_b, ur_b)] = small[slicen(ll_s, ur_s)]
            except ValueError:
                print("idx", idx)
                print("big ll-ur", ll_b, ur_b)  # TODO sometimes this fails, check
                print("big slice shape", big[slicen(ll_b, ur_b)].shape)
                print("small ll-ur", ll_s, ur_s)
                print("small slice shape", small[slicen(ll_s, ur_s)].shape)


def get_exclusion_mask(a, exclude_values):
    exclude_values = np.atleast_1d(exclude_values)
    bool_a = np.ones_like(a, dtype=bool)
    for v in exclude_values:
        bool_a[a == v] = False

    return bool_a


def matmul(a, b, axes_a=(-2, -1), axes_b=(-2, -1)):
    if axes_a == (-2, -1) and axes_b == (-2, -1):
        return a @ b

    if axes_a == (-3, -2) and axes_b == (-2, -1) and np.ndim(a) == np.ndim(b) + 1:
        return np.concatenate([(a[..., i] @ b)[..., np.newaxis]
                               for i in range(a.shape[-1])], axis=-1)
    elif axes_a == (-2, -1) and axes_b == (-3, -2) and np.ndim(b) == np.ndim(a) + 1:
        return np.concatenate([(a @ b[..., i])[..., np.newaxis]
                               for i in range(b.shape[-1])], axis=-1)
    else:
        raise NotImplementedError


def matsort(mat, order_j=None):
    """
    mat = np.where(np.random.random((500, 500)) > 0.01, 1, 0)
    ax[0].imshow(mat, cmap='gray_r')
    ax[1].imshow(matsort(mat)[0], cmap='gray_r')
    """

    def idx2interval(idx):
        idx = np.sort(idx)
        interval = np.zeros((len(idx) - 1, 2), dtype=int)
        interval[:, 0] = idx[:-1]
        interval[:, 1] = idx[1:]
        return interval

    n, m = mat.shape

    if order_j is None:
        order_j = np.argsort(np.sum(mat, axis=0))[::-1]

    order_i = np.argsort(mat[:, order_j[0]])[::-1]

    interval_idx = np.zeros(2, dtype=int)
    interval_idx[1] = n
    for i in range(0, m - 1):
        interval_idx = np.unique(np.hstack([interval_idx,
                                            find.get_interval_indices(mat[order_i, order_j[i]]).ravel()]))

        for j, il in enumerate(idx2interval(idx=interval_idx)):
            slice_j = slice(il[0], il[1])
            if j % 2 == 0:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])[::-1]]
            else:
                order_i[slice_j] = order_i[slice_j][np.argsort(mat[order_i[slice_j], order_j[i + 1]])]

    return mat[order_i, :][:, order_j], order_i, order_j


def idx2boolmat(idx, n=100):
    """
    The last dimension of idx contains the indices. n-1 is the maximal possible index
    Returns matrix with shape 'np.shape(idx)[:-1] + (n, )'

    """

    s = np.shape(idx)[:-1]

    mat = np.zeros(s + (n,), dtype=bool)

    for i, idx_i in enumerate(idx.reshape(-1, idx.shape[-1])):
        print(i, (np.unravel_index(i, shape=s)))
        mat[np.unravel_index(i, shape=s)][idx_i] = True
    return mat


def construct_array(shape, val, idx, init_mode="zeros", dtype=None,
                    axis=None, insert_mode=None):
    a = initialize_array(shape=shape, mode=init_mode, dtype=dtype)
    basics.insert(a=a, val=val, idx=idx, axis=axis, mode=insert_mode)
    return a


# Blocklists
def block_view(a, shape, aslist=False, require_aligned_blocks=True):
    """
    Return a 2N-D view of the given N-D array, rearranged so each ND block (tile)
    of the original array is indexed by its block address using the first N
    indexes of the output array.
    Note: This function is nearly identical to ``skimage.util.view_as_blocks()``, except:
          - "imperfect" block shapes are permitted (via require_aligned_blocks=False)
          - only contiguous arrays are accepted.  (This function will NOT silently copy your array.)
            As a result, the return value is *always* a view of the input.
    Args:
        a: The ND array
        shape: The tile shape
        aslist: If True, return all blocks as a list of ND blocks
                instead of a 2D array indexed by ND block coordinate.
        require_aligned_blocks: If True, check to make sure no data is "left over"
                                in each row/column/etc. of the output view.
                                That is, the blockshape must divide evenly into the full array shape.
                                If False, "leftover" items that cannot be made into complete blocks
                                will be discarded from the output view.
    Here's a 2D example (this function also works for ND):
    # >>> arr = np.arange(1,21).reshape(4,5)
    # >>> print(arr)
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    # >>> view = blockwise_view(arr, (2,2), require_aligned_blocks=False)
    # >>> print(view)
    [[[[ 1  2]
       [ 6  7]]
    <BLANKLINE>
      [[ 3  4]
       [ 8  9]]]
    <BLANKLINE>
    <BLANKLINE>
     [[[11 12]
       [16 17]]
    <BLANKLINE>
      [[13 14]
       [18 19]]]]

    Inspired by the 2D example shown here: https://stackoverflow.com/a/8070716/162094
    """
    assert a.flags["C_CONTIGUOUS"], "This function relies on the memory layout of the array."
    shape = tuple(shape)
    outershape = tuple(np.array(a.shape) // shape)
    view_shape = outershape + shape

    if require_aligned_blocks:
        assert np.all(np.mod(a.shape, shape) == 0), (
            "blockshape {} must divide evenly into array shape {}".format(shape, a.shape))

    # inner strides: strides within each block (same as original array)
    intra_block_strides = a.strides

    # outer strides: strides from one block to another
    inter_block_strides = tuple(a.strides * np.array(shape))

    # This is where the magic happens.
    # Generate a view with our new strides (outer+inner).
    view = np.lib.stride_tricks.as_strided(a, shape=view_shape, strides=(inter_block_strides + intra_block_strides))

    if aslist:
        return list(map(view.__getitem__, np.ndindex(outershape)))
    return view


def expand_block_indices(idx_block, block_size, squeeze=True):
    """
    Expand the indices to get an index for each element

           block_size: |   1  |     2    |       3      |         4        |           5          |
    block_idx:          --------------------------------------------------------------------------|
              0        |   0  |   0,  1  |   0,  1,  2  |   0,  1,  2,  3  |   0,  1,  2,  3,  4  |
              1        |   1  |   2,  3  |   3,  4,  5  |   4,  5,  6,  7  |   5,  6,  7,  8,  9  |
              2        |   2  |   4,  5  |   6,  7,  8  |   8,  9, 10, 11  |  10, 11, 12, 13, 14  |
              3        |   3  |   6,  7  |   9, 10, 11  |  12, 13, 14, 15  |  15, 16, 17, 18, 19  |
              4        |   4  |   8,  9  |  12, 13, 14  |  16, 17, 18, 19  |  20, 21, 22, 23, 24  |
              5        |   5  |  10, 11  |  15, 16, 17  |  20, 21, 22, 23  |  25, 26, 27, 28, 29  |
              6        |   6  |  12, 13  |  18, 19, 20  |  24, 25, 26, 27  |  30, 31, 32, 33, 34  |
              7        |   7  |  14, 15  |  21, 22, 23  |  28, 29, 30, 31  |  35, 36, 37, 38, 39  |
              8        |   8  |  16, 17  |  24, 25, 26  |  32, 33, 34, 35  |  40, 41, 42, 43, 44  |
              9        |   9  |  18, 19  |  27, 28, 29  |  36, 37, 38, 39  |  45, 46, 47, 48, 49  |
              10       |  10  |  20, 21  |  30, 31, 32  |  40, 41, 42, 43  |  50, 51, 52, 53, 54  |
    """

    idx_block = np.atleast_1d(idx_block)
    if np.size(idx_block) == 1:
        return np.arange(block_size * int(idx_block), block_size * (int(idx_block) + 1))
    else:
        idx2 = np.array([expand_block_indices(i, block_size=block_size, squeeze=squeeze) for i in idx_block])
        if squeeze:
            return idx2.flatten()
        else:
            return idx2


def replace(arr, r_dict, copy=True, dtype=None):
    if copy:
        arr2 = arr.copy()
        if dtype is not None:
            arr2 = arr2.astype(dtype=dtype)
        for key in r_dict:
            arr2[arr == key] = r_dict[key]
        return arr2
    else:
        for key in r_dict:
            arr[arr == key] = r_dict[key]
        return None


def replace_tail_roll(a, b):
    """
    Replace the last elements of the array with the new array and roll the new ones to the start
    So that a repeated call of this function cycles through the array
    replace_tail_roll(a=[ 1,  2,  3,  4,  5,  6, 7, 8], b=[77, 88])   -->   [77, 88,  1,  2,  3,  4,  5,  6]
    replace_tail_roll(a=[77, 88,  1,  2,  3,  4, 5, 6], b=[55, 66])   -->   [55, 66, 77, 88,  1,  2,  3,  4]
    replace_tail_roll(a=[55, 66, 77, 88,  1,  2, 3, 4], b=[33, 44])   -->   [33, 44, 55, 66, 77, 88,  1,  2]
    replace_tail_roll(a=[33, 44, 55, 66, 77, 88, 1, 2], b=[11, 22])   -->   [11, 22, 33, 44, 55, 66, 77, 88]
    """
    n_a, n_b = np.shape(a)[0], np.shape(b)[0]
    assert n_a > n_b

    a[-n_b:] = b
    return np.roll(a, n_b, axis=0)


def replace_tail_roll_list(arr_list, arr_new_list):
    assert len(arr_list) == len(arr_new_list)
    return (replace_tail_roll(a=arr, b=arr_new) for (arr, arr_new) in zip(arr_list, arr_new_list))


def diag_wrapper(x, n=None):
    x = np.asarray(x)
    if n is None:
        n = x.shape[0]

    if np.all(x.shape == (n, n)):
        return x
    else:
        d = np.eye(n)
        d[range(n), range(n)] = x

    return d


def create_constant_diagonal(n, m, v, k):
    diag = np.eye(N=n, M=m, k=k) * v[0]
    for i in range(1, len(v)):
        diag += np.eye(N=n, M=m, k=k + i) * v[i]
    return diag


def banded_matrix(v_list, k0):
    m = np.diag(v_list[0], k=k0)
    for i, v in enumerate(v_list[1:], start=1):
        m += np.diag(v, k=k0 + i)

    return m


def get_stats(x, axis=None, return_array=False):
    """
    order :size, mean, std, median, min, max

    """

    stats = {"size": int(np.size(x, axis=axis)),
             "mean": np.mean(x, axis=axis),
             "std": np.std(x, axis=axis),
             "median": np.median(x, axis=axis),
             "min": np.min(x, axis=axis),
             "max": np.max(x, axis=axis)}

    if return_array:
        return np.array([stats["size"], stats["mean"], stats["std"], stats["median"], stats["min"], stats["max"]])

    return stats


def verbose_reject_x(title, x, b):
    if b.size == 0:
        mean = 0
    else:
        mean = b.mean()
    print(f"{title}: {b.sum()}/{b.size} ~ {np.round(mean * 100, 3)}%")
    return x[b].copy()


def get_points_inbetween(x, extrapolate=False):
    assert x.ndim == 1

    delta = x[1:] - x[:-1]
    x_new = np.zeros(np.size(x) + 1)
    x_new[1:-1] = x[:-1] + delta / 2
    if extrapolate:
        x_new[0] = x_new[1] - delta[0]
        x_new[-1] = x_new[-2] + delta[-1]
        return x_new
    else:
        return x_new[1:-1]
