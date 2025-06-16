import numpy as np

from scipy.signal import convolve
from scipy import ndimage
from skimage import measure
from skimage.morphology import flood_fill

from wzk import geometry, np2, printing, trajectory, grid, spatial


__eps = 1e-9


def __closest_boundary_rel_idx(half_side):
    idx_rel = np.arange(start=-half_side + 1, stop=half_side + 2, step=1)
    idx_rel[half_side:] = idx_rel[half_side - 1:-1]
    return idx_rel


def closest_grid_boundary(*, x, half_side, limits, shape, idx=None):
    """
    Given the coordinates of a point 'x', the radius of a sphere and dimensions of a grid.
    Calculate for that coordinate the closest boundary to the point 'x' for a cell defined by the relative idx j.
    4 cases:

           ----- ----- ----- ----- -----
          |     |     |     |     |     |
          |     |     |     |     |     |
          |    o|    o|  j  |o    |o    |
           ----- ----- ----- ----- -----
          |     |ooooo|ooooo|ooooo|     |
          |     |ooooo|ooooo|ooooo|     |
          |    o|ooooo|ooooo|ooooo|o    |
           ----- ----- ----- ----- -----
          |     |ooooo|     |ooooo|     |
          |    j|ooooo|  x  |ooooo|j    |
          |     |ooooo|     |ooooo|     |
           ----- ----- ----- ----- -----
          |    o|ooooo|ooooo|ooooo|o    |
          |     |ooooo|ooooo|ooooo|     |
          |     |ooooo|ooooo|ooooo|     |
           ----- ----- ----- ----- -----
          |    o|    o|  j  |o    |o    |
          |     |     |     |     |     |
          |     |     |     |     |     |
           ----- ----- ----- ----- -----


    The four cases at the intersection of cross with the grid
        __j__
        j_x_j
        __j__
    """

    if idx is None:
        idx = grid.x2i(x=x, limits=limits, shape=shape)

    idx = np2.flatten_without_last(x=idx)
    rel_idx = __closest_boundary_rel_idx(half_side=half_side)

    idx = idx[:, np.newaxis, :] + rel_idx[np.newaxis, :, np.newaxis]
    x_closest = grid.i2x(i=idx, limits=limits, shape=shape, mode="b")

    x_closest[:, half_side, :] = x

    return x_closest


def __get_centers(voxel_size, n_dim):
    limits0 = np.zeros((n_dim, 2))
    limits0[:, 0] = __eps
    limits0[:, 1] = voxel_size - __eps

    if n_dim == 2:
        v, e = geometry.rectangle(limits=limits0)

    elif n_dim == 3:
        v, e, f = geometry.cube(limits=limits0)

    else:
        raise ValueError

    return v


# Helper
# ----------------------------------------------------------------------------------------------------------------------
def __compare_dist_against_radius(x_a, x_b, r):
    dist = x_a - x_b
    dist = (dist**2).sum(axis=-1)
    return dist < r ** 2 - 5 * __eps


def get_max_occupied_cells(length, voxel_size):
    return np.asarray(np.ceil(length / voxel_size), dtype=int) + 1


def get_outer_edge(img):
    n_dim = np.ndim(img)
    kernel = np.ones((3,)*n_dim)
    edge_img = convolve(img, kernel, mode="same", method="direct") > 0
    return np.logical_xor(edge_img, img)


def get_sphere_stencil(r: float, voxel_size: float, n_dim: int = 2) -> (np.ndarray, np.ndarray):
    half_side = get_max_occupied_cells(length=r, voxel_size=voxel_size) - 1

    if half_side == 0:
        return np.ones((1,) * n_dim, dtype=bool), np.zeros((1,) * n_dim, dtype=bool)

    x_center = __get_centers(voxel_size=voxel_size, n_dim=n_dim)

    limits = np.zeros((n_dim, 2))
    limits[:, 1] = voxel_size
    shape = np.ones(n_dim)
    x_closest = closest_grid_boundary(x=x_center, half_side=half_side, limits=limits, shape=shape)

    img = np.zeros((len(x_center),) + (2 * half_side + 1,) * n_dim, dtype=bool)
    for i in range(len(x_center)):
        x_closest_i = np.array(np.meshgrid(*x_closest[i].T, indexing="ij")).T
        img[i, ...] = __compare_dist_against_radius(x_a=x_center[i], x_b=x_closest_i, r=r)

    inner = img.sum(axis=0) == img.shape[0]
    outer = get_outer_edge(inner)

    return inner, outer


def get_stencil_list(r, n,
                     voxel_size, n_dim):
    if np.size(r) > 1:
        assert np.size(r) == n
        r_unique, stencil_idx = np.unique(r, return_inverse=True)
        stencil_list = [get_sphere_stencil(r=r_, voxel_size=voxel_size, n_dim=n_dim) for r_ in r_unique]
    else:
        stencil_list = [get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)]
        stencil_idx = np.zeros(n, dtype=int)
        r_unique = np.array([r])

    return r_unique, stencil_list, stencil_idx


def create_stencil_dict(voxel_size, n_dim):
    stencil_dict = dict()
    n = int(5*(1//voxel_size))
    for i, r in enumerate(np.linspace(voxel_size/10, 2, num=n)):
        printing.progress_bar(i=i, n=n, prefix="create_stencil_dict")
        d = int((r // voxel_size) * 2 + 3)
        if d not in stencil_dict.keys():
            stencil = np.logical_or(*get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim))
            assert d == stencil.shape[0]
            stencil_dict[d] = stencil
    return stencil_dict


def bimg2surf(img, limits, level=None):
    lower_left = limits[:, 0]
    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits)
    if img.sum() == 0:
        verts = np.zeros((3, 3))
        faces = np.zeros((1, 3), dtype=int)
        faces[:] = np.arange(3)

    else:
        verts, faces, _, _ = measure.marching_cubes(img, level=level, spacing=(voxel_size,) * img.ndim,)
        verts = verts + lower_left

    return verts, faces


def mesh2bimg(p, shape, limits, f=None):
    img = np.zeros(shape, dtype=int)

    voxel_size = grid.limits2voxel_size(shape=shape, limits=limits)
    if img.ndim == 2:
        p2 = np.concatenate([p, p[:1]], axis=0)
        p2 = trajectory.get_substeps_adjusted(x=p2, n=2 * len(p) * max(shape))
        i2 = grid.x2i(x=p2, limits=limits, shape=shape)
        img[np.clip(i2[:, 0], a_min=0, a_max=img.shape[0]-1),
            np.clip(i2[:, 1], a_min=0, a_max=img.shape[1]-1)] = 1

    elif img.ndim == 3:
        if f is None:
            ch = geometry.ConvexHull(p)
            p = ch.points
            f = ch.simplices  # noqa
        p2 = geometry.discretize_triangle_mesh(p=p, f=f, voxel_size=voxel_size)
        i2 = grid.x2i(x=p2, limits=limits, shape=shape)
        img[np.clip(i2[:, 0], a_min=0, a_max=img.shape[0]-1),
            np.clip(i2[:, 1], a_min=0, a_max=img.shape[1]-1),
            np.clip(i2[:, 2], a_min=0, a_max=img.shape[2]-1)] = 1

    else:
        raise ValueError

    img = flood_fill(img, seed_point=(0,) * img.ndim, connectivity=1, new_value=2)
    img = np.array(img != 2)

    return img


def spheres2bimg(x, r, shape, limits,
                 stencil_dict=None):
    x = np.atleast_2d(x)
    n, n_dim = x.shape
    assert len(shape) == n_dim

    r = np2.scalar2array(r, shape=n)
    img = np.zeros(shape, dtype=bool)
    voxel_size = grid.limits2voxel_size(shape=shape, limits=limits)

    for i in range(n):
        j = grid.x2i(x[i], limits=limits, shape=shape)
        d = int((r[i] // voxel_size) * 2 + 3)
        if stencil_dict:
            stencil = stencil_dict[d]
        else:
            stencil = np.logical_or(*get_sphere_stencil(r=r[i], voxel_size=voxel_size, n_dim=n_dim))
        np2.add_small2big(idx=j, small=stencil, big=img)

    return img


def add_boxes_img(img, box_list, limits):
    for x in box_list:
        x = geometry.cube(limits=x)[0]
        img_x = mesh2bimg(p=x, limits=limits, shape=img.shape)
        img[:] = np.logical_or(img, img_x)


# Sampling 
# ----------------------------------------------------------------------------------------------------------------------
def sample_bimg_i(img, n, replace=True):
    i = np.array(np.nonzero(img)).T
    j = np.random.choice(a=np.arange(len(i)), size=n, replace=replace)
    return i[j]


def sample_bimg_x(img, limits, n, cell_noise=True, replace=True):
    """
    Sample a set of points from a binary image.
    Use 'sample_bimg_i' to sample the indices and
    then use noise to sample inside each cell for full coverage of the coordinate space.
    """

    i = sample_bimg_i(img=img, n=n, replace=replace)
    x = grid.i2x(i=i, limits=limits, shape=img.shape, mode="c")
    if cell_noise:
        voxel_size2 = grid.limits2voxel_size(shape=img.shape, limits=limits) / 2
        cell_noise = np.random.uniform(low=-voxel_size2, high=+voxel_size2, size=(n, img.ndim))
        x += cell_noise
    return x


def sample_spheres_bimg_x(x, r, shape, limits, n,):
    img = spheres2bimg(x=x, r=r, shape=shape, limits=limits)
    x = sample_bimg_x(img=img, limits=limits, n=n, replace=True)
    return x


def rotate_bimg_3d(bimg: np.ndarray, dcm) -> np.ndarray:
    # TODO IS WAY QUICKER
    idx = np.array(np.nonzero(bimg)).T
    idx -= np.array(bimg.shape)
    idx = (dcm[np.newaxis, :, :] @ idx[:, :, np.newaxis])[:, :, 0]

    idx -= idx.min(axis=0)
    idx = idx.astype(int)
    bimg = np.zeros(idx.max(axis=0)+1)
    bimg[idx[:, 0], idx[:, 1], idx[:, 2]] = 1
    return bimg


def rotate(bimg, angle, axes):
    __eps2 = 1e-6
    angle = np.rad2deg(angle)
    threshold = 0.1

    if np.allclose(angle, 0):
        return bimg

    if np.allclose(angle % 90, 0) or np.allclose(angle % 90, 90):
        bimg = np.rot90(bimg, k=(angle + __eps2) // 90, axes=axes)

    else:
        bimg = bimg.astype(np.float32)
        bimg = ndimage.rotate(bimg, angle=angle, order=0, axes=axes)
        bimg = np.array(bimg > threshold)

    return bimg.astype(bool)


def rotate_bimg_3d_old(bimg: np.ndarray, dcm) -> np.ndarray:
    # TODO replace with pytorch / cupy or own binary rotate function
    assert bimg.ndim == 3

    assert np.all(np.shape(dcm) == (3, 3))
    euler = spatial.dcm2euler(dcm=dcm, seq="zxz")  # extrinsic rotations

    bimg = rotate(bimg=bimg, angle=euler[0], axes=(0, 1))  # z
    bimg = rotate(bimg=bimg, angle=euler[1], axes=(1, 2))  # x
    bimg = rotate(bimg=bimg, angle=euler[2], axes=(0, 1))  # z

    return bimg.astype(bool)


def crop_bimg_to_fit(bimg: np.ndarray, pad: int = 1):
    idx = np.array(np.nonzero(bimg))
    i_min = np.min(idx, axis=1)
    i_min = np.maximum(np.zeros(3), i_min - pad).astype(int)

    i_max = np.max(idx, axis=1)
    i_max = np.minimum(np.array(bimg.shape), i_max + pad + 1).astype(int)
    print("a", bimg.shape)
    bimg = bimg[np2.slicen(i_min, i_max)]
    print("b", bimg.shape)
    return bimg
