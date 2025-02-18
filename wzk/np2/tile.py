import numpy as np
from scipy.signal import convolve2d

from wzk import ltd

from .shape import shape_wrapper
from .basics import scalar2array


def tile_offset(a, reps, offsets=None):
    """
    first tile array a via reps
    than offset each repetition according to an increasing offset
    """

    s = shape_wrapper(a.shape)

    b = np.tile(a, reps)

    if offsets is not None:
        r = np.array(b.shape) // np.array(a.shape)
        if np.size(offsets) == 1:
            o = scalar2array(offsets, shape=len(s))
        else:
            o = offsets

        assert len(o) == len(s)
        offsets = [np.repeat(np.arange(rr), ss) * oo for ss, rr, oo in zip(s, r, o)]
        b += sum(np.meshgrid(*offsets, indexing='ij') + (0,))  # noqa
    return b


def tile_2d(*, pattern, v_in_row, v_to_next_row, offset=(0, 0),
            shape):
    """

    Examples:

    # Point A
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 4
    v_to_next_row = (1, 2)

    # [[1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]
    #  [0 0 1 0 0 0 1 0 0 0 1]
    #  [1 0 0 0 1 0 0 0 1 0 0]]

    # Point B
    pattern = np.ones((1, 1))
    shape = (11, 11)
    offset = (0, 0)
    v_in_row = 5
    v_to_next_row = (1, 2)

    # [[1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]
    #  [0 0 1 0 0 0 0 1 0 0 0]
    #  [0 0 0 0 1 0 0 0 0 1 0]
    #  [0 1 0 0 0 0 1 0 0 0 0]
    #  [0 0 0 1 0 0 0 0 1 0 0]
    #  [1 0 0 0 0 1 0 0 0 0 1]]

    # Dumbbell
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    pattern[1, 0] = 0
    shape = (16, 16)
    offset = (0, 0)
    v_in_row = 7
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 0 1 0 0 0 0 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 0 1 0 0 0 0 1 0]
    #  [0 0 0 1 0 1 0 0 0 0 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 0 1 0 0 0 0 1 0 1]
    #  [0 0 1 0 1 0 0 0 0 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 0 1 0 0 0 0 1 0 1 0]
    #  [0 1 0 1 0 0 0 0 1 0 1 0 0 0 0 1]]

    # Triangle
    pattern = np.ones((2, 2))
    pattern[0, 1] = 0
    shape = (10, 10)
    offset = (0, 0)
    v_in_row = 8
    v_to_next_row = (1, 3)

    # [[1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]
    #  [0 0 0 1 1 0 1 0 0 0]
    #  [0 1 0 0 0 0 1 1 0 1]
    #  [0 1 1 0 1 0 0 0 0 1]
    #  [0 0 0 0 1 1 0 1 0 0]
    #  [1 0 1 0 0 0 0 1 1 0]
    #  [0 0 1 1 0 1 0 0 0 0]
    #  [1 0 0 0 0 1 1 0 1 0]
    #  [1 1 0 1 0 0 0 0 1 1]]

    # Cross
    pattern = np.zeros((3, 3))
    pattern[1, :] = 1
    pattern[:, 1] = 1
    shape = (13, 13)
    offset = (1, 1)
    v_in_row = 4
    v_to_next_row = (3, 2)

    # [[1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [0 1 1 1 0 1 1 1 0 1 1 1 0]
    #  [0 0 1 0 0 0 1 0 0 0 1 0 0]
    #  [1 0 0 0 1 0 0 0 1 0 0 0 1]
    #  [1 1 0 1 1 1 0 1 1 1 0 1 1]]
    """
    nodes = np.zeros((shape[0]+v_to_next_row[0], shape[1]+v_in_row))

    for ii, i in enumerate(range(0, nodes.shape[0], v_to_next_row[0])):
        nodes[i, range((ii*v_to_next_row[1]) % v_in_row, nodes.shape[1], v_in_row)] = 1

    img = convolve2d(nodes, pattern, mode="full")

    ll = (v_to_next_row[0] + offset[0],
          v_to_next_row[1] + offset[1])

    return img[ll[0]:ll[0]+shape[0],
               ll[1]:ll[1]+shape[1]]


def block_collage(*, img_arr, inner_border=None, outer_border=None, fill_boarder=0, dtype=float):

    assert img_arr.ndim == 4
    n_rows, n_cols, n_x, n_y = img_arr.shape

    bv_i, bh_i = ltd.tuple_extract(inner_border, default=(0, 0), mode="repeat")
    bv_o, bh_o = ltd.tuple_extract(outer_border, default=(0, 0), mode="repeat")

    img = np.full(shape=(n_x * n_rows + bv_i * (n_rows - 1) + 2*bv_o,
                         n_y * n_cols + bh_i * (n_cols - 1) + 2*bh_o), fill_value=fill_boarder, dtype=dtype)

    for r in range(n_rows):
        for c in range(n_cols):
            img[bv_o + r * (n_y + bv_i):bv_o + (r + 1) * (n_y + bv_i) - bv_i,
                bh_o + c * (n_x + bh_i):bh_o + (c + 1) * (n_x + bh_i) - bh_i] = img_arr[r, c]

    return img
