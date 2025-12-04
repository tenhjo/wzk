import numpy as np
from matplotlib import patches, transforms, pyplot, path

from wzk import geometry


class RelativeFancyArrow(patches.FancyArrow):
    def __init__(self, x, y, dx, dy, width=0.2, length_includes_head=True, head_width=0.3, head_length=0.4,
                 shape="full", overhang=0, head_starts_at_zero=False, **kwargs):
        length = np.hypot(dx, dy)
        super().__init__(x, y, dx, dy, width=width*length, length_includes_head=length_includes_head*length,
                         head_width=head_width*length, head_length=head_length*length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyArrowX2(patches.FancyArrow):
    def __init__(self, xy0, xy1, offset0=0.0, offset1=0.0,
                 width=0.2, head_width=0.3, head_length=0.4, overhang=0,
                 shape="full",  length_includes_head=True, head_starts_at_zero=False, **kwargs):

        xy0, xy1 = np.array(xy0), np.array(xy1)
        dxy = xy1 - xy0
        dxy /= np.linalg.norm(dxy, keepdims=True)

        xy0 += offset0 * dxy

        dxy = xy1 - xy0
        dxy -= dxy / np.linalg.norm(dxy) * offset1

        super().__init__(*xy0, *dxy, width=width, length_includes_head=length_includes_head,
                         head_width=head_width, head_length=head_length,
                         shape=shape, overhang=overhang, head_starts_at_zero=head_starts_at_zero, **kwargs)


class FancyBbox(patches.FancyBboxPatch):
    def __init__(self, xy: np.ndarray = np.zeros(2), width=1., height=1., boxstyle="Round", pad=0.3, corner_size=None,
                 **kwargs):
        if boxstyle in ["Roundtooth", "Sawtooth"]:
            bs = patches.BoxStyle(boxstyle, pad=pad, tooth_size=corner_size)
        elif boxstyle in ["Round", "Round4"]:
            bs = patches.BoxStyle(boxstyle, pad=pad, rounding_size=corner_size)
        else:
            bs = patches.BoxStyle(boxstyle, pad=pad)

        xy_pad = np.array(xy) + pad
        super().__init__(xy=(xy_pad[0], xy_pad[1]), width=width - 2*pad, height=height - 2*pad, boxstyle=bs, **kwargs)


class RoundedPolygon(patches.PathPatch):
    def __init__(self, xy, pad, **kwargs):
        p = path.Path(*self.round(xy=xy, pad=pad))
        super().__init__(path=p, **kwargs)

    @staticmethod
    def round(xy, pad):
        verts = None
        n = len(xy)
        for i in range(0, n):

            x0, x1, x2 = np.atleast_1d(xy[i - 1], xy[i], xy[(i + 1) % n])

            d01, d12 = x1 - x0, x2 - x1
            d01, d12 = d01 / np.linalg.norm(d01), d12 / np.linalg.norm(d12)

            x00 = x0 + pad * d01
            x01 = x1 - pad * d01
            x10 = x1 + pad * d12
            # x11 = x2 - pad * d12

            if i == 0:
                verts = [x00, x01, x1, x10]
            else:
                verts += [x01, x1, x10]

        codes = [path.Path.MOVETO] + n*[path.Path.LINETO, path.Path.CURVE3, path.Path.CURVE3]

        return np.atleast_1d(verts, codes)


class CurlyBrace(patches.PathPatch):
    """
    Create a matplotlib patch corresponding to a curly brace | i.e., these things: { / }

    Adopted from
    https://github.com/bensondaled/curly_brace/blob/master/curly_brace_patch.py
    """

    def __init__(self, p, x0, x1,
                 curliness=1/np.e, **kwargs):

        kwargs["edgecolor"] = kwargs.pop("color", "black")
        kwargs["facecolor"] = "none"

        p, x0, x1, = np.atleast_1d(p, x0, x1)

        pp = geometry.projection_point_line(p=p, x0=x0, x1=x1)
        d0pp = x0 - pp
        d1pp = x1 - pp

        if np.linalg.norm(d0pp) < np.linalg.norm(d1pp):
            curliness_v = d0pp * curliness
        else:
            curliness_v = d1pp * curliness

        self.verts = np.stack([x0,  #
                               p + d0pp,
                               pp + curliness_v,
                               p,  #
                               pp - curliness_v,
                               p + d1pp,
                               x1])  #

        codes = [path.Path.MOVETO] + 6 * [path.Path.CURVE4]
        super().__init__(patches.Path(self.verts, codes), **kwargs)  # noqa


# Transformations
def do_aff_trafo(patch, theta, xy=None, por=(0, 0)):
    if xy is not None:
        patch.set_xy(xy=xy)
    patch.set_transform(get_aff_trafo(theta=theta, por=por, patch=patch))


def get_aff_trafo(xy0=None, xy1=None, theta=0, por=(0, 0), ax=None, patch=None):
    """

    xy0: current position of the object, if not provided patch.get_xy() is used
    xy1: desired position of the object
    theta: rotation in degrees
    por: point of rotation relative to the object coordinates
    ax:
    patch:
    :return:
    """

    if xy0 is None:
        if patch is None:
            xy0 = (0, 0)
        else:
            xy0 = patch.get_xy()

    if xy1 is None:
        xy1 = xy0

    if ax is None:
        if patch is None:
            ax = pyplot.gca()
        else:
            ax = patch.axes

    return (transforms.Affine2D().translate(-xy0[0]-por[0], -xy0[1]-por[1])
                                 .rotate_deg_around(0, 0, theta)
                                 .translate(xy1[0], xy1[1]) + ax.transData)


def make_every_box_fancy(ax,
                         shrink_x=0.0, shrink_y=0.0, pad=0.1):
    # Rounded bar plots (https://stackoverflow.com/questions/58425392/bar-chart-with-rounded-corners-in-matplotlib)
    new_patches = []
    for patch in reversed(ax.patches):
        try:
            bb = patch.get_bbox()
        except AttributeError:
            continue

        if abs(bb.height) > 0.5:
            p_bbox = FancyBbox(xy=np.array((bb.xmin+shrink_x/2, bb.ymin+shrink_y/2)),
                               width=abs(bb.width)-shrink_x,
                               height=abs(bb.height)-shrink_y,
                               pad=pad,
                               corner_size=None, color=patch.get_facecolor(),
                               alpha=patch.get_alpha())
            new_patches.append(p_bbox)

        patch.remove()

    for patch in new_patches:
        ax.add_patch(patch)
