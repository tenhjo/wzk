import datetime
import numpy as np
from itertools import combinations
from scipy.stats import linregress
from matplotlib import collections, patches

from wzk.mpl.figure import plt, new_fig, subplot_grid
from wzk.mpl.colors import arr2rgba
from wzk.mpl.axes import limits4axes, limits2extent, set_ax_limits
from wzk.mpl.legend import rectangle_legend

from wzk.math2 import binomial
from wzk.numpy2 import scalar2array, add_safety_limits
from wzk.dicts_lists_tuples import tuple_extract, atleast_tuple


def imshow(ax: plt.Axes, img: np.ndarray, h=None,
           cmap=None,
           limits: np.ndarray = None, origin: str = 'lower', axis_order: str = 'ij->yx',
           mask: np.ndarray = None, vmin: float = None, vmax: float = None, **kwargs):
    """

    ## origin: upper
    # axis_order: ij
    (i0 ,j0), (i0 ,j1), (i0 ,j2), ...
    (i1, j0), (i1, j1), (i1, j2), ...
    (i2, j0), (i2, j1), (i2, j2), ...
    ...     , ...     ,  ...    , ...

    # axis_order: ji
     (i0 ,j0), (i1 ,j0), (i2 ,j0), ...
     (i0, j1), (i1, j1), (i2, j1), ...
     (i0, j2), (i1, j2), (i2, j2), ...
     ...     , ...     ,  ...    , ...

    ## origin: lower
    # axis_order: ij
    ...     , ...     ,  ...    , ...
    (i2, j0), (i2, j1), (i2, j2), ...
    (i1, j0), (i1, j1), (i1, j2), ...
    (i0 ,j0), (i0 ,j1), (i0 ,j2), ...

    # axis_order: ji
     ...     , ...     ,  ...    , ...
     (i0, j2), (i1, j2), (i2, j2), ...
     (i0, j1), (i1, j1), (i2, j1), ...
     (i0 ,j0), (i1 ,j0), (i2 ,j0), ...
    """

    assert img.ndim == 2
    assert origin in ('lower', 'upper')
    assert axis_order in ('ij->yx', 'ij->xy')
    if limits is None:
        limits = img.shape

    if h is not None:
        if cmap is None:
            cmap = h.cmap
        img = arr2rgba(img=img, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask, axis_order=axis_order)
        h.set_data(img)
        return h

    extent = limits2extent(limits=limits, origin=origin, axis_order=axis_order)

    img = arr2rgba(img=img, cmap=cmap, vmin=vmin, vmax=vmax, mask=mask, axis_order=axis_order)
    if 'label' in kwargs:
        kwargs2 = kwargs.copy()
        kwargs2['color'] = cmap
        rectangle_legend(ax=ax, xy=limits[:, 0]-1, **kwargs2)

    return ax.imshow(img, extent=extent, origin=origin, **kwargs)


def plot_projections_2d(x, dim_labels=None, ax=None, limits=None, aspect='auto', **kwargs):
    n = x.shape[-1]

    n_comb = binomial(n, 2)
    if ax is None:
        ax = subplot_grid(n=n_comb, squeeze=False, aspect=aspect)
    else:
        ax = np.atleast_2d(ax)

    if dim_labels is None:
        dim_labels = [str(i) for i in range(n)]
    assert len(dim_labels) == n, f"{dim_labels} | {n}"

    comb = combinations(np.arange(n), 2)
    for i, c in enumerate(comb):
        i = np.unravel_index(i, shape=ax.shape)
        ax[i].plot(*x[..., c].T, **kwargs)
        ax[i].set_xlabel(dim_labels[c[0]])
        ax[i].set_ylabel(dim_labels[c[1]])
        if limits is not None:
            set_ax_limits(ax=ax[i], limits=limits[c, :])


def color_plot_connected(y, color_s, x=None, connect_jumps=True, ax=None, **kwargs):
    """
    Parameter
    ---------
    ax: matplotlib.axes
    y: array_like
        y-Measurements
    color_s: array_like
        Same length as y, colors for each measurement's point.
    x:
        x-measurements
    connect_jumps: bool
        If True, the border between two different colors is drawn 50 / 50 with both neighboring colors
    **d:
        Additional keyword arguments for plt.plot()
    """

    if ax is None:
        ax = plt.gca()

    n = len(y)
    if x is None:
        x = range(n)

    i = 0
    h = []
    while i < n:
        cur_col = color_s[i]
        j = i + 1
        while j < n and color_s[j] == cur_col:
            j += 1

        h.append(ax.plot(x[i:j], y[i:j], c=cur_col, **kwargs)[0])

        if connect_jumps:
            h.append(line_2colored(ax=ax, x=x[j - 1:j + 1], y=y[j - 1:j + 1], colors=color_s[j - 1:j + 1], **kwargs))
        i = j

    return h


def line_2colored(x, y, colors, ax=None, **kwargs):
    """
    Plot a line with 2 colors.
    Parameter
    ---------
    ax: matplotlib.axes
    x:
        x-Measurements, 2 points
    y:
        y-Measurements, 2 points
    colors:
        2 colors. First is for the first half of the line and the second color for the second part of the line
    **d:
        Additional keyword-arguments for plt.plot()
    """

    if ax is None:
        ax = plt.gca()

    if type(x[0]) is datetime.date or type(x[0]) is datetime.datetime:
        xh = datetime.datetime(year=x[0].year, month=x[0].month, day=x[0].day)
        xh += (x[1] - x[0]) / 2
    else:
        xh = np.mean(x)
    yh = np.mean(y)

    h = [ax.plot([x[0], xh], [y[0], yh], c=colors[0], **kwargs)[0],
         ax.plot([xh, x[1]], [yh, y[1]], c=colors[1], **kwargs)[0]]
    return h


def color_plot(x, y, color_s, plot_fcn, **kwargs):
    """
    Plot a line with an individual color for each point.
    x: Data for x-axis
    y: Data for y-axis
    color_s: array of colors with the same length as x and y respectively. If now enough colors are given,
             use just the first (only) one given
    plot_fcn: Matplotlib function, which should be used for plotting -> use ax.fun() to ensure that the right
              axis is used
    kwargs: Additional d for matplotlib.pyplot.plot()
    """

    h = []
    for i in range(len(x)):
        c = color_s[i % len(color_s)]
        h.append(plot_fcn([x[i]], [y[i]], color=c, **kwargs))

    return h


def draw_lines_between(*, x1=None, x2=None, y1, y2, ax=None, **kwargs):
    # https://stackoverflow.com/questions/59976046/connect-points-with-horizontal-lines
    ax = ax or plt.gca()
    x1 = x1 if x1 is not None else np.arange(len(y1))
    x2 = x2 if x2 is not None else x1

    cl = collections.LineCollection(np.stack((np.c_[x1, x2], np.c_[y1, y2]), axis=2), **kwargs)
    ax.add_collection(cl)
    return cl


def get_hist(ax):
    """
    https://stackoverflow.com/questions/33888973/get-values-from-matplotlib-axessubplot#33895378
    """

    x1 = None
    n, bins = [], []
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        n.append(y1 - y0)
        bins.append(x0)  # left edge of each bin
    bins.append(x1)  # also get right edge of last bin

    return n, bins


def error_area(x, y, y_std,
               ax,
               kwargs, kwargs_std):
    ax.plot(x, y, **kwargs)
    ax.fill_between(x, y-y_std, y+y_std, **kwargs_std)


def quiver(xy, uv,
           ax=None, h=None,
           **kwargs):

    if h is None:
        h = ax.quiver(xy[..., 0].ravel(), xy[..., 1].ravel(), uv[..., 0].ravel(), uv[..., 1].ravel(),
                      angles='xy', scale=1, units='xy', scale_units='xy',
                      **kwargs)
    else:
        h.set_offsets(xy)
        h.set_UVC(*uv.T)

    return h


# Grid
def create_grid(ll: (float, float), ur: (float, float), n: (int, int), pad: (float, float)):
    ll, ur, n, pad = np.atleast_1d(ll, ur, n, pad)

    w = ur - ll
    s = (w - pad*(n-1))/n

    x = ll[0] + np.arange(n[0])*(s[0]+pad[0])
    y = ll[1] + np.arange(n[1])*(s[1]+pad[1])
    return (x, y), s


def grid_lines(ax, start, step, limits, **kwargs):
    mins, maxs = limits4axes(limits=limits, n_dim=2)
    start = tuple_extract(t=start, default=(0, 0), mode='repeat')
    step = tuple_extract(t=step, default=(0, 0), mode='repeat')

    ax.hlines(y=np.arange(start=start[0], stop=maxs[1], step=step[0]), xmin=mins[0], xmax=maxs[0], **kwargs)
    ax.vlines(x=np.arange(start=start[1], stop=maxs[0], step=step[1]), ymin=mins[1], ymax=maxs[1], **kwargs)


def grid_lines_data(ax, x, limits: str = 'ax', **kwargs):
    if limits == 'ax':
        limits = np.array([ax.get_xlim(), ax.get_ylim()])
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())
    elif limits == 'data':
        limits = np.array([[x[:, 0].min(), x[:, 0].max()],
                           [x[:, 1].min(), x[:, 1].max()]])
    else:
        raise ValueError

    ax.vlines(x[..., 0].ravel(), ymin=limits[1, 0], ymax=limits[1, 1], **kwargs)
    ax.hlines(x[..., 1].ravel(), xmin=limits[0, 0], xmax=limits[0, 1], **kwargs)


def update_vlines(h, x, ymin: float = None, ymax: float = None):
    seg_old = h.get_segments()
    if ymin is None:
        ymin = seg_old[0][0, 1]
    if ymax is None:
        ymax = seg_old[0][1, 1]

    seg_new = [np.array([[xx, ymin],
                         [xx, ymax]]) for xx in x]

    h.set_segments(seg_new)


def update_hlines(h, y, xmin: float = None, xmax: float = None):
    seg_old = h.get_segments()
    if xmin is None:
        xmin = seg_old[0][0, 0]
    if xmax is None:
        xmax = seg_old[0][1, 0]

    seg_new = [np.array([[xmin, yy],
                         [xmax, yy]]) for yy in y]

    h.set_segments(seg_new)


def hist_vlines(x, name, bins=100,
                hl_idx=None, hl_color=None, hl_name=None,
                lower_perc=None, upper_perc=None):

    if lower_perc is not None:
        _range = (np.percentile(x, lower_perc), np.percentile(x, upper_perc))
    else:
        _range = None
    fig, ax = new_fig(title=f'Histogram: {name}', scale=2)
    hist = ax.hist(x, bins=bins, range=_range)

    perc_i = []
    if hl_idx is not None:
        hl_idx, hl_color, hl_name = scalar2array(hl_idx, hl_color, hl_name, shape=np.size(hl_idx))
        for i, c, n in zip(hl_idx, hl_color, hl_name):
            perc_i.append(np.sum(x[i] > x))
            label = None if n is None else f"{n} | {perc_i[-1]} / {len(x)}"
            ax.vlines(x[i], ymin=0, ymax=len(x), lw=4, color=c, label=label)

    ax.set_ylim(0, hist[0].max() * 1.02)

    if lower_perc is not None:
        ax.set_xlim(np.percentile(x, lower_perc), np.percentile(x, upper_perc))

    if hl_name is not None:
        ax.legend()
    return ax, perc_i


#
def correlation_plot(a, b, name_a, name_b,
                     regression_line=True,
                     lower_perc=0, upper_perc=100,
                     labels=None, colors=None, markers='o', markersizes=None, alphas=None, zorders=None,
                     ax=None, verbose=1, **kwargs):

    if ax is None:
        fig, ax = new_fig(width=10, title=f"Correlation: {name_a} | {name_b}")

    a, b = atleast_tuple(a, b, convert=False)

    a_all = np.concatenate(a)
    b_all = np.concatenate(b)

    limits = ((np.percentile(a_all, lower_perc), np.percentile(a_all, upper_perc)),
              (np.percentile(b_all, lower_perc), np.percentile(b_all, upper_perc)))

    limits = add_safety_limits(limits=limits, factor=0.01)

    if regression_line:
        s, i, r, p, _ = linregress(a_all, b_all)
        x_reg = np.linspace(limits[0, 0], limits[0, 1], 3)
        y_reg = x_reg * s + i
        a += (x_reg,)
        b += (y_reg,)
        if verbose > 0:
            print("slope: {:.4} | correlation: {:.4} | p {:.4}".format(s, r, p))
    else:
        r = None

    labels, colors, markers, markersizes, alphas, zorders = \
        scalar2array(labels, colors, markers, markersizes, alphas, zorders, shape=len(a))

    for i, (aa, bb, la, co, ma, ms, al, zo) in enumerate(zip(a, b,
                                                             labels, colors, markers, markersizes, alphas, zorders)):

        if regression_line and i+1 == len(a):
            la = None if la is None else la.format(r)
            ax.plot(aa, bb, color=co, label=la, ls=ma, lw=ms, alpha=al, zorder=zo)
        else:
            ax.plot(aa, bb, ls='', marker=ma, color=co, label=la, markersize=ms, alpha=al, zorder=zo, **kwargs)

    set_ax_limits(ax=ax, limits=limits)

    if labels is not None:
        ax.legend()

    ax.set_xlabel(name_a)
    ax.set_ylabel(name_b)

    return ax


def plot_circles(x, r,
                 ax=None, h=None,
                 color=None, alpha=None,
                 **kwargs):
    # https://stackoverflow.com/questions/48172928/scale-matplotlib-pyplot-axes-scatter-markersize-by-x-scale
    x = np.reshape(x, (-1, 2))
    r = scalar2array(r, shape=len(x))

    if h is None:
        h = []
        for x_i, r_i in zip(x, r):
            c = patches.Circle(xy=x_i, radius=r_i, alpha=alpha, color=color, **kwargs)
            ax.add_patch(c)
            h.append(c)
        return h

    else:
        for h_i, x_i in zip(h, x):
            h_i.set_center(x_i)

            if alpha is not None:
                h_i.set_alpha(alpha)

            if color is not None:
                h_i.set_color(color)
        return h
