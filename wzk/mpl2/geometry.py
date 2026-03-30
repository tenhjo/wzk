import numpy as np
from matplotlib import pyplot as plt, patches, colors as mcolors

from wzk.math import geometry as _geometry
from wzk.mpl2 import Patches2, axes, plotting


def draw_arc(xy, radius, theta0=0., theta1=2 * np.pi, n=0.01, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    x = _geometry.get_arc(xy=xy, radius=radius, theta0=theta0, theta1=theta1, n=n)
    h = ax.plot(*x.T, **kwargs)[0]

    return x, h


def fill_circle_intersection(xy0, r0, xy1, r1, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    int01 = _geometry.circle_circle_intersection(xy0=xy0, r0=r0, xy1=xy1, r1=r1)
    if int01 is None:
        return None
    else:
        int0, int1 = int01

    d00 = int0 - xy0
    d10 = int1 - xy0
    d01 = int0 - xy1
    d11 = int1 - xy1
    aa00 = np.arctan2(d00[1], d00[0])
    aa01 = np.arctan2(d10[1], d10[0])
    aa10 = np.arctan2(d01[1], d01[0])
    aa11 = np.arctan2(d11[1], d11[0])

    arc0, _ = draw_arc(xy=xy0, radius=r0, theta0=aa00, theta1=aa01, alpha=0)
    arc1, _ = draw_arc(xy=xy1, radius=r1, theta0=aa11, theta1=aa10, alpha=0)

    if np.allclose(arc0[0], arc1[0]):
        pp = np.concatenate([arc0, arc1[-2:0:-1]], axis=0)
    else:
        assert np.allclose(arc0[0], arc1[-1])
        pp = np.concatenate([arc0, arc1[1:-1]], axis=0)

    poly = patches.Polygon(pp, **kwargs)
    ax.add_patch(poly)

    return ((int0, aa00, aa01), (int1, aa10, aa11)), poly


def draw_rays(xy, radius0, radius1, theta0=0., theta1=None, n=1, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    theta0, theta1 = _geometry.theta_wrapper(theta0=theta0, theta1=theta1)

    h = np.zeros(n, dtype=object)
    for i in range(n):
        a_i = theta0 + (theta1 - theta0) * i/(n-1)

        h[i] = ax.plot([xy[0] + np.cos(a_i)*radius0, xy[0] + np.cos(a_i)*radius1],
                       [xy[1] + np.sin(a_i)*radius0, xy[1] + np.sin(a_i)*radius1],
                       **kwargs)[0]
    return h


def plot_coordinate_frames(ax=None, x=None, dcm=None, f=None, scale=1.0,
                           color="k", mode="quiver", marker=None,
                           h=None, **kwargs):
    """
    Assume matrix is a homogeneous matrix

    Note: the columns of the frame are the vectors x, y, z in the base coordinate frame
    """
    if x is None and dcm is None and f is None:
        raise ValueError
    if f is not None:
        x, dcm = f[..., :-1, -1], f[..., :-1, :-1]
    elif x is None:
        x = np.zeros((1, 3))
    elif dcm is None:
        dcm = np.eye(3)[np.newaxis, :, :]

    x = np.asarray(x)
    dcm = np.asarray(dcm)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if dcm.ndim == 2:
        dcm = dcm[np.newaxis, :, :]

    n_dim = min(x.shape[-1], dcm.shape[-1])
    x = x[..., :n_dim]
    dcm = dcm[..., :n_dim, :n_dim]

    x = x.reshape((-1, n_dim))
    dcm = dcm.reshape((-1, n_dim, n_dim))
    n_samples = max(x.shape[0], dcm.shape[0])

    if n_samples > 1:
        if h is None:
            h = [None] * n_samples
        else:
            assert len(h) == n_samples
        return [plot_coordinate_frames(ax=ax, x=x[i], dcm=dcm[i], color=color, mode=mode, marker=marker, h=h[i],
                                       **kwargs)
                for i in range(n_samples)]

    x = x[0]
    dcm = dcm[0] * scale

    if isinstance(color, str):
        # Support shorthand such as "bb" / "ry" for per-axis colors.
        if mcolors.is_color_like(color):
            color = [color]
        else:
            color = list(color)
    elif not isinstance(color, list):
        color = list(color)
    if len(color) < n_dim:
        color *= n_dim

    if h is not None:
        n_dim_h = len(h)
        x = x[:n_dim_h]
        dcm = dcm[:n_dim_h, :n_dim_h]
        for i, hh in enumerate(h):
            ax_h = ax if ax is not None else getattr(hh, "axes", None)
            if dcm.shape[0] == 3 and ax_h is not None:
                try:
                    hh.remove()
                except ValueError:
                    pass
                h[i] = ax_h.quiver(x[0], x[1], x[2], dcm[0, i], dcm[1, i], dcm[2, i], color=color[i], **kwargs)
            else:
                plotting.quiver(ax=ax, xy=x, uv=dcm[:, i], color=color[i], h=hh)
        return h

    h = []
    if mode == "quiver":
        for i in range(n_dim):
            if n_dim == 3:
                h.append(ax.quiver(x[0], x[1], x[2], dcm[0, i], dcm[1, i], dcm[2, i], color=color[i], **kwargs))
            else:
                h.append(plotting.quiver(ax=ax, xy=x, uv=dcm[:, i], color=color[i], **kwargs))

    elif mode == "fancy":
        for i in range(n_dim):
            h.append(patches.FancyArrow(x[0], x[1], dcm[0, i], dcm[1, i], color=color[i], **kwargs))
            ax.add_patch(h[-1])

    elif mode == "relative_fancy":
        for i in range(n_dim):
            h.append(Patches2.RelativeFancyArrow(x[0], x[1], dcm[0, i], dcm[1, i], color=color[i],  **kwargs))
            ax.add_patch(h[-1])
    else:
        raise ValueError

    if marker is not None:
        markersize = axes.size_units2points(size=2*kwargs["fig_width_inch"]*np.linalg.norm(dcm, axis=0).mean(), ax=ax)
        ax.plot(*x, marker=marker, markersize=markersize, color=color[-1], alpha=0.5)

    return h


# Combination of the building blocks
def eye_pov(xy, angle, radius, arc, n_rays=3,
            ax=None, solid_capstyle="round", **kwargs):

    if ax is None:
        ax = plt.gca()

    cornea_factor = 0.9
    cornea_factor = radius * cornea_factor

    pupil_factor = 0.2
    pupil_x = 1
    pupil_radius = radius * pupil_factor
    pupil_xy = np.array([xy[0] + np.cos(angle) * radius * pupil_x,
                         xy[1] + np.sin(angle) * radius * pupil_x])

    rays_radius0 = radius * 0.95
    rays_radius1 = radius * 1.15
    rays_section = 40 / 100

    h_edges = draw_rays(ax=ax, xy=xy, radius0=radius*0.0, radius1=radius,
                        theta0=angle - arc / 2, theta1=angle + arc / 2, n=2,
                        solid_capstyle=solid_capstyle, **kwargs)

    h_rays = draw_rays(ax=ax, xy=xy, radius0=rays_radius0, radius1=rays_radius1,
                       theta0=angle - arc/2 * rays_section, theta1=angle + arc/2 * rays_section, n=n_rays,
                       solid_capstyle=solid_capstyle, **kwargs)

    h_arc = draw_arc(ax=ax, xy=xy, radius=cornea_factor, theta0=angle - arc / 2, theta1=angle + arc / 2,
                     solid_capstyle=solid_capstyle, **kwargs)[1]

    *_, h_pupil = fill_circle_intersection(xy0=xy, r0=cornea_factor, xy1=pupil_xy, r1=pupil_radius, ax=ax, **kwargs)

    return h_edges, h_rays, h_arc, h_pupil


def plot_box(ax, limits=None, **kwargs):
    if limits is None:
        limits = np.array([[0, 1],
                           [0, 1]])

    x = _geometry.box(limits)

    ax.plot(*x.T, **kwargs)
