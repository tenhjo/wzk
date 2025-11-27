import numpy as np

from wzk import np2


def limits2size(limits):
    return limits[:, 1] - limits[:, 0]


def limits2center(limits):
    return limits[:, 0] + limits2size(limits) / 2


def limits2voxel_size(shape, limits, unify=True):
    voxel_size = limits2size(limits) / np.array(shape)
    if unify:
        voxel_size = np2.unify(x=voxel_size)
    return voxel_size


def __mode2offset(voxel_size, mode="c"):
    """Modes
        'c': center
        'b': boundary

    """
    if mode == "c":
        return voxel_size / 2
    elif mode == "b":
        return 0
    else:
        raise NotImplementedError(f"Unknown mode: '{mode}'")


def x2i(x, limits, shape):
    """
    Get the indices of the grid cell at the coordinates 'x' in a grid with symmetric cells.
    Always use mode='boundary'
    """

    if x is None:
        return None
    voxel_size = limits2voxel_size(shape=shape, limits=limits, unify=False)
    lower_left = limits[:, 0]

    return np.asarray((x - lower_left) / voxel_size, dtype=int)


def i2x(i, limits, shape, mode="c"):
    """
    get the coordinates of the grid at the index "i" in a grid with symmetric cells.
    borders: 0 | 2 | 4 | 6 | 8 | 10
    centers: | 1 | 3 | 5 | 7 | 9 |
    """

    if i is None:
        return None
    voxel_size = limits2voxel_size(shape=shape, limits=limits, unify=False)
    lower_left = limits[:, 0]

    offset = __mode2offset(voxel_size=voxel_size, mode=mode)
    return np.asarray(lower_left + offset + i * voxel_size, dtype=float)


def create_grid(limits, shape, mode="c", flatten=False):
    n_dim = len(limits)
    # voxel_size = limits2voxel_size(shape=shape, limits=limits)

    ll = i2x(i=np.zeros(n_dim), limits=limits, mode=mode, shape=shape)
    ur = i2x(i=np.array(shape)-1, limits=limits, mode=mode, shape=shape)

    x = np.array(np.meshgrid(*[np.linspace(start=ll[i], stop=ur[i], num=shape[i]) for i in range(n_dim)],
                             indexing="ij"))

    x = np.moveaxis(x, 0, -1)
    if flatten:
        return x.reshape((np.prod(shape), n_dim))
    return x
