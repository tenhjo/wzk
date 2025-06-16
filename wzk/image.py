import zlib
import numpy as np
from skimage.io import imread, imsave  # noqa

from wzk import np2, math2
from wzk.bimage import sample_bimg_i


def imread_bw(file, threshold):

    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    img = imread(file)
    img = rgb2gray(img)
    obstacle_image = img < threshold  # 60
    return obstacle_image


def combine_shape_n_dim(shape, n_dim=None):
    if np.size(shape) == 1:
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (shape,)
        shape *= n_dim

    else:
        shape = tuple(shape)

    return shape


def image_array_shape(shape, n_samples=None, n_dim=None, n_channels=None):
    """
    Helper to set the shape for an image array.
    n_samples=100,  shape=64,          n_dim=2,    n_channels=None  ->  (100, 64, 64)
    n_samples=100,  shape=64,          n_dim=3,    n_channels=2     ->  (100, 64, 64, 64, 2)
    n_samples=None, n_voxel=(10, 11, 12), n_dim=None, n_channels=None  ->  (10, 11, 12)
    """

    shape = combine_shape_n_dim(shape=shape, n_dim=n_dim)

    if n_samples is not None:
        shape = (n_samples,) + shape
    if n_channels is not None:
        shape = shape + (n_channels,)

    return shape


def initialize_image_array(shape: tuple, n_dim: int = None, n_samples: int = None, n_channels: int = None,
                           dtype=None, initialization: str = "zeros") -> np.ndarray:
    shape = image_array_shape(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels)
    return np2.initialize_array(shape=shape, mode=initialization, dtype=dtype)


def reshape_img(img, n_dim=2, sample_dim=True, channel_dim=True,
                n_samples=None,  # either
                n_channels=None):  # or. infer the other

    shape = img.shape[1]  # Can go wrong for 1D

    if n_samples is None and n_channels is None:
        n_channels = 1
        n_samples = 1

    elif n_samples is None:
        n_samples = img.size // (shape ** n_dim) // n_channels
    elif n_channels is None:
        n_channels = img.size // (shape ** n_dim) // n_samples

    if n_samples == 1 and not sample_dim:
        n_samples = None

    if n_channels == 1 and not channel_dim:
        n_channels = None

    return img.reshape(image_array_shape(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels))


def concatenate_images(*imgs, axis=-1):
    """
    could add parameters which allow stacking horizontal, vertical, channel_wise, and sample wise
    """
    n_dim = np.max([i.ndim for i in imgs])
    if axis == +1:
        imgs = [i if i.ndim == n_dim else i[np.newaxis, ...] for i in imgs]
    elif axis == -1:
        imgs = [i if i.ndim == n_dim else i[..., np.newaxis] for i in imgs]
    else:
        raise NotImplementedError

    return np.concatenate(imgs, axis=axis)


def add_padding(img, padding, value):
    shape = np.array(img.shape)
    padding_arr = np.zeros((img.ndim, 2), dtype=int)

    if isinstance(padding, str):
        padding_arr[:, 0] = math2.make_even_odd(shape, mode=padding, rounding=+1) - shape

    elif isinstance(padding, int):
        padding_arr[:] = padding

    elif isinstance(padding, (np.ndarray, tuple, list)):
        if padding.ndim == 2:
            padding_arr[:] = padding
        elif len(padding) == len(shape):
            padding_arr[:] = np.expand_dims(padding, axis=1)

    else:
        raise ValueError

    new_shape = shape + padding_arr.sum(axis=1)
    new_img = np.full(shape=new_shape, fill_value=value, dtype=img.dtype)
    new_img[np2.slicen(padding_arr[:, 0], padding_arr[:, 0]+shape)] = img
    return new_img


def pooling(mat, kernel, method="max", pad=False):
    """
    Non-overlapping pooling on 2D or 3D Measurements.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel shape in (ky, kx).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has shape
           n//f, n being <mat> shape, f being kernel shape.
           if pad output has shape ceil(n/f).

    Return <result>: pooled matrix.
    """

    print("# TODO write general function with reshaping -> much faster")
    m, n = mat.shape[:2]
    ky, kx = kernel

    def _ceil(x, y):
        return int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, ky)
        nx = _ceil(n, kx)
        size = (ny * ky, nx * kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        ny = m // ky
        nx = n // kx
        mat_pad = mat[:ny * ky, :nx * kx, ...]

    new_shape = (ny, ky, nx, kx) + mat.shape[2:]

    if method == "max":
        result = np.nanmax(mat_pad.reshape(new_shape), axis=(1, 3))
    else:
        result = np.nanmean(mat_pad.reshape(new_shape), axis=(1, 3))

    return result


def check_overlap(a, b, return_arr=False):
    """
    Boolean indicating if the two arrays have an overlap.
    Sum over the axis that do not match.
    """

    a, b = (a, b) if a.n_dim > b.n_dim else (b, a)

    aligned_shape = np2.align_shapes(a=a, b=b)
    summation_axis = np.arange(a.n_dim)[aligned_shape == -1]
    a = a.sum(axis=summation_axis)

    if return_arr:
        return np.logical_and(a, b)
    else:
        return np.logical_and(a, b).any()



def sample_from_img(img, range, n, replace=False):   # noqa
    bimg = np.logical_and(range[0] < img, img < range[1])
    return sample_bimg_i(img=bimg, n=n, replace=replace)


# Image Compression <-> Decompression
def img2compressed(img, n_dim: int, level: int = 9):
    """
    Compress the given image with the zlib routine to a binary string.
    Level of compression can be adjusted. A timing with respect to different compression levels for decompression showed
    no difference, so the highest level is default, this corresponds to the largest compression.
    For compression, it is slightly slower but this happens just once and not during keras training, so the smaller
    needed memory was favoured.

    Alternative:
    <-> use numpy sparse for the world images, especially in 3d  -> zlib is more effective and more general
    """

    shape = img.shape[:-n_dim]
    if n_dim == -1 or shape == ():
        return zlib.compress(img.tobytes(), level=level)

    else:
        img_cmp = np.empty(shape, dtype=object)
        for idx in np.ndindex(*shape):
            img_cmp[idx] = zlib.compress(img[idx, ...].tobytes(), level=level)
        return img_cmp


def compressed2img(img_cmp, shape, n_dim=None, n_channels=None, dtype=None):
    """
    Decompress the binary string back to an array of given shape
    """

    shape2 = image_array_shape(shape=shape, n_dim=n_dim, n_channels=n_channels)
    if np.shape(img_cmp):
        n_samples = np.size(img_cmp)
        img_arr = initialize_image_array(shape=shape, n_dim=n_dim, n_samples=n_samples, n_channels=n_channels,
                                         dtype=dtype)
        for i in range(n_samples):
            img_arr[i, ...] = np.frombuffer(bytearray(zlib.decompress(img_cmp[i])), dtype=dtype).reshape(shape2)
        return img_arr

    else:
        return np.frombuffer(bytearray(zlib.decompress(img_cmp)), dtype=dtype).reshape(shape2)
