import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

from wzk.mpl2.bimage_boundaries import get_combined_edges

from wzk import grid


def plot_img_patch(img, limits,
                   ax=None, **kwargs):
    """
    Plot an image as a Collection of square Rectangle Patches.
    Draw all True / nonzero pixels.
    """

    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits)

    ij = np.array(np.nonzero(img)).T
    xy = grid.i2x(i=ij, limits=limits, shape=img.shape, mode="b")

    pc = PatchCollection([Rectangle((x, y), width=voxel_size, height=voxel_size,
                                    fill=False, snap=True) for (x, y) in xy], **kwargs)
    ax.add_collection(pc)
    return pc


def plot_img_outlines(img, limits,
                      ax, **kwargs):
    """
    Plot the image by drawing the outlines of the areas where the values are True.
    """

    combined_edges = get_combined_edges(bimg=img)
    combined_edges = [grid.i2x(i=ce, shape=img.shape, limits=limits, mode="b")
                      for ce in combined_edges]

    lc = LineCollection(combined_edges, **kwargs)
    ax.add_collection(lc)
    return lc


def __img_none_limits(limits=None, img=None):
    if limits is None and img is not None:
        limits = np.zeros((img.ndim, 2))
        limits[:, 1] = img.shape

    return limits


def plot_img_patch_w_outlines(ax, img, limits,
                              color=None, edgecolor="k", hatchcolor="k", facecolor="None",
                              hatch="xx",
                              lw=2, alpha_outline=1, alpha_patch=1.,
                              **kwargs):
    if img is None:
        return None

    if color is not None:
        facecolor = color
        edgecolor = color

    limits = __img_none_limits(limits=limits, img=img)

    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])

    plot_img_outlines(img=img, limits=limits, ax=ax, color=edgecolor, ls="-", lw=lw, alpha=alpha_outline,
                      **kwargs)
    plot_img_patch(img=img, limits=limits, ax=ax, lw=0, hatch=hatch, facecolor=facecolor, edgecolor=hatchcolor,
                   alpha=alpha_patch, **kwargs)
    return None


def initialize_pixel_grid(img, limits, ax, **kwargs):

    ij = np.array(list(np.ndindex(img.shape)))
    xy = grid.i2x(i=ij, shape=img.shape, limits=limits, mode="b")
    voxel_size = grid.limits2voxel_size(shape=img.shape, limits=limits)
    pixel_grid = PatchCollection([Rectangle(xy=(x, y), width=voxel_size,
                                            height=voxel_size, snap=True)
                                  for (x, y) in xy], **kwargs)
    ax.add_collection(pixel_grid)

    return pixel_grid


def set_pixel_grid(bimg, pixel_grid, **kwargs):
    val_none_dict = {"color": "None",
                     "edgecolor": "None",
                     "facecolor": "None",
                     "linewidth": 0,
                     "linestyle": "None"}

    def value_wrapper(k):
        v = kwargs[k]

        if isinstance(v, (tuple, list)) and len(v) == 2:
            v, v_other = v

        else:
            v_other = val_none_dict[k]

        def ravel(val):
            if np.size(val) == np.size(bimg):
                val = np.ravel(val)
            return val

        v = ravel(v)
        v_other = ravel(v_other)

        return np.where(np.ravel(bimg), v, v_other).tolist()

    for kw in kwargs:
        set_fun = getattr(pixel_grid, f"set_{kw}")
        set_fun(value_wrapper(kw))


def switch_img_values(bimg, i, j, value=None):
    i = [i] if isinstance(i, int) else i
    j = [j] if isinstance(j, int) else j

    if value is None:
        mean_obstacle_occurrence = np.mean(bimg[i, j])
        value = not np.round(mean_obstacle_occurrence)
    bimg = bimg.copy()
    bimg[i, j] = value

    return bimg
