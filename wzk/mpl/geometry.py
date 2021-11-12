import numpy as np
from matplotlib import pyplot as plt, patches, collections

from wzk.numpy2 import scalar2array
from wzk.geometry import circle_circle_intersection, theta_wrapper, angle_resolution_wrapper
from wzk.mpl import Patches2, size_units2points, plotting


def draw_circles(ax: plt.Axes, x, y, r, **kwargs):
    # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale
    r = scalar2array(r, shape=np.size(x))
    circles = [plt.Circle((xi, yi), radius=ri, linewidth=0, **kwargs) for xi, yi, ri in zip(x, y, r)]
    c = collections.PatchCollection(circles, **kwargs)
    ax.add_collection(c)


def draw_arc(xy, radius, theta0=0., theta1=2 * np.pi, n=0.01, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    theta0, theta1 = theta_wrapper(theta0=theta0, theta1=theta1)
    n = angle_resolution_wrapper(n, angle=theta1 - theta0)

    theta = np.linspace(start=theta0, stop=theta1, num=n)
    x = xy[0] + np.cos(theta) * radius
    y = xy[1] + np.sin(theta) * radius
    h = ax.plot(x, y, **kwargs)[0]

    return np.array([x, y]).T, h


def fill_circle_intersection(xy0, r0, xy1, r1, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    int01 = circle_circle_intersection(xy0=xy0, r0=r0, xy1=xy1, r1=r1)
    if int01 is None:
        return
    else:
        int0, int1 = int01
    aa00 = np.arctan2(*(int0 - xy0)[::-1])
    aa01 = np.arctan2(*(int1 - xy0)[::-1])
    aa10 = np.arctan2(*(int0 - xy1)[::-1])
    aa11 = np.arctan2(*(int1 - xy1)[::-1])

    arc0, _ = draw_arc(xy=xy0, radius=r0, theta0=aa00, theta1=aa01, alpha=0)
    arc1, _ = draw_arc(xy=xy1, radius=r1, theta0=aa11, theta1=aa10, alpha=0)

    if np.allclose(arc0[0], arc1[0]):
        pp = np.concatenate([arc0, arc1[-2:0:-1]], axis=0)
    else:
        assert np.allclose(arc0[0], arc1[-1])
        pp = np.concatenate([arc0, arc1[1:-1]], axis=0)

    poly = patches.Polygon(pp, **kwargs)
    ax.add_patch(poly)

    return (int0, aa00, aa01), (int1, aa10, aa11), poly


def draw_rays(xy, radius0, radius1, theta0=0., theta1=None, n=1, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()

    theta0, theta1 = theta_wrapper(theta0=theta0, theta1=theta1)

    h = np.zeros(n, dtype=object)
    for i in range(n):
        a_i = theta0 + (theta1 - theta0) * i/(n-1)

        h[i] = ax.plot([xy[0] + np.cos(a_i)*radius0, xy[0] + np.cos(a_i)*radius1],
                       [xy[1] + np.sin(a_i)*radius0, xy[1] + np.sin(a_i)*radius1],
                       **kwargs)[0]
    return h


def plot_coordinate_frame(ax=None, x=None, dcm=None,
                          color='k', mode='quiver', marker=None,
                          h=None, **kwargs):
    """
    Assume matrix is a homogeneous matrix

    Note: the columns of the frame are the vectors x, y, z in the base coordinate frame
    """

    if x is None and dcm is None:
        raise ValueError
    elif x is None:
        x = np.zeros(3)
    elif dcm is None:
        dcm = np.eye(3)

    n_dim = min(len(x), len(dcm))
    x = x[:n_dim]

    if not isinstance(color, list):
        color = [color]
    if len(color) < n_dim:
        color *= n_dim

    if h is not None:
        for i, hh in enumerate(h):
            plotting.quiver(ax=ax, xy=x, uv=dcm[:, i], color=color[i], h=hh)
            # h[i].set_segments([np.array([x,
            #                              x + dcm[:, i]])])
        return h

    h = []
    if mode == 'quiver' or n_dim == 3:
        for i in range(n_dim):
            # h.append(ax.quiver(*x, *dcm[:, i], color=color[i], **kwargs))
            h.append(plotting.quiver(ax=ax,  xy=x, uv=dcm[:, i], color=color[i], **kwargs))

    elif mode == 'fancy':
        for i in range(n_dim):
            h.append(patches.FancyArrow(x[0], x[1], dcm[0, i], dcm[1, i], color=color[i], **kwargs))
            ax.add_patch(h[-1])

    elif mode == 'relative_fancy':
        for i in range(n_dim):
            h.append(Patches2.RelativeFancyArrow(x[0], x[1], dcm[0, i], dcm[1, i], color=color[i],  **kwargs))
            ax.add_patch(h[-1])
    else:
        raise ValueError

    if marker is not None:
        ax.plot(*x, marker=marker, color=color[-1],
                markersize=size_units2points(size=2*kwargs['fig_width_inch']*np.linalg.norm(dcm, axis=0).mean(), ax=ax),
                alpha=0.5)

    return h


# Combination of the building blocks
def eye_pov(xy, angle, radius, arc, n_rays=3,
            ax=None, solid_capstyle='round', **kwargs):

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
