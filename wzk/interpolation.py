
from wzk.logger import log_print
import numpy as np

from scipy.interpolate import CubicSpline, PPoly, splev, splrep
from scipy.signal import savgol_filter
from scipy.linalg import solve_banded
from scipy.stats import norm

from wzk.trajectory import get_substeps_adjusted, get_substeps
from wzk import new_fig


def get_cubic_spline(x, y, mode="i2"):
    m = get_tangents(x=x, y=y, mode=mode)
    c = get_coefficients(p=y, m=m, x=x)
    return PPoly(c[::-1, :], x)


def get_tangents(x, y, mode="i1"):
    n = len(x)

    h = np.diff(x)
    g = np.diff(y)
    s = g/h

    la = h[1:] / (h[:-1] + h[1:])
    mu = 1 - la

    # A x = b
    # ab = np.empty((3, n))  # banded matrix a[0, :] upper diag, a[1, :] diag, a[2, :] lower diag
    b = np.empty(n)

    if mode == "i0":
        h3 = h**3
        la3n = h3[+1:] / (h3[:-1] + h3[1:])
        la3p = h3[:-1] / (h3[:-1] + h3[1:])

        a_in = np.concatenate([[0, -3], -3*la3n])
        a_ii = np.full(n, 4)
        a_ip = np.concatenate([-3*la3p, [-3, 0]])
        ab = np.vstack([a_in, a_ii, a_ip])

        b[0] = s[0]
        b[1:-1] = la3p * s[:-1] + la3n * s[1:]
        b[-1] = s[-1]

    elif mode == "i1":
        a_in = np.concatenate([[0, -1], -la])
        a_ii = np.full(n, 4)
        a_ip = np.concatenate([-mu, [-1, 0]])
        ab = np.vstack([a_in, a_ii, a_ip])

        b[0] = 3 * s[0]
        b[1:-1] = 3 * g[1:] / (h[:-1] + h[1:])
        b[-1] = 3 * s[-1]

    elif mode == "i2":
        a_in = np.concatenate([[0, 1], mu])
        a_ii = np.full(n, 2)
        a_ip = np.concatenate([la, [1, 0]])
        ab = np.vstack([a_in, a_ii, a_ip])

        b[0] = 3 * s[0]
        b[1:-1] = 3 * (la * s[:-1] + mu * s[1:])
        b[-1] = 3 * s[-1]
    else:
        raise ValueError

    # log_print('Ab - own')
    # log_print(ab)
    # log_print(b)
    m = solve_banded((1, 1), ab, b)
    # log_print(matrix)

    return m


def get_coefficients(p, m, x=None):

    p0 = p[:-1]
    p1 = p[1:]
    m0 = m[:-1]
    m1 = m[1:]

    c = np.empty((4, len(p)-1))

    # Assume unit interval
    if x is None:
        c[0] = p0
        c[1] = m0
        c[2] = -3*p0 + 3*p1 - 2*m0 - m1
        c[3] = +2*p0 - 2*p1 + 1*m0 + m1

    else:
        h = np.diff(x)

        c[0] = p0
        c[1] = m0
        c[2] = (-3*p0 + 3*p1 - (2*m0 + m1)*h) / h**2
        c[3] = (+2*p0 - 2*p1 + 1*(m0 + m1)*h) / h**3

    # In scipy - equivalent
    # g = np.diff(p)
    # slope = g/h
    # t = (m0 + m1 - 2 * slope) / h
    # c[0] = p0
    # c[1] = m0
    # c[2] = (slope - m0) / h - t
    # c[3] = t / h

    return c


def scale_coefficients(c, x):
    """unit interval -> value interval"""

    d = np.diff(x)
    d1 = 1 / d
    d2 = d1 * d1
    d3 = d2 * d1

    c[1] *= d1
    c[2] *= d2
    c[3] *= d3

    return c


# def plot_tangents(ax, x, y, t, length=1, **kwargs):
#     l2 = length/2
#     for xx, tt in zip(x, t):
#         ax.plot([x-l2, x+l2], [y-l2*t, y+l2*t], **kwargs)


def test_interpolation_paper():
    n = 1000
    # Paper
    x = np.array([0.5, 3.5, 6, 8.5, 11., 14., 17., 20.])
    y = np.array([93., 104., 120., 98., 86., 102., 81., 90.])
    plot_3_splines(x=x, y=y, n=n, title="Blood-Glucose Level")

    x = np.array([0., 3., 5., 6., 8., 11.])
    y = np.array([0., 1., 2., 4., 5., 6.])
    plot_3_splines(x=x, y=y, n=n, title="Monotone Example")

    x = np.array([0., 0.3, 1., 1.8, 3., 4.2, 5., 5.7, 6.])
    y = np.sqrt(9 - (x - 3)**2)
    plot_3_splines(x=x, y=y, n=n, title="Circle")

    x = np.array([0.1, 0.2, 0.6, 1.0, 1.2, 1.4])
    y = 1/x**2
    plot_3_splines(x=x, y=y, n=n, title="1/x2")

    #


def test_interpolation():
    n = 2000
    x = np.linspace(0, 9, 5)
    y = np.sin(x)
    plot_3_splines(x=x, y=y, n=n, title="Sinus")

    n0 = 5
    x = np.arange(n0)
    y = np.random.random(n0)
    plot_3_splines(x=x, y=y, n=n, title="Random")

    n0 = 1000
    n05 = 1000
    x = np.arange(n0)
    y = np.random.random(n0)
    x, y = get_substeps_adjusted(x=np.vstack((x, y)).T[np.newaxis], n=n05)[0].T
    plot_3_splines(x=x, y=y, n=n, title="Random, Dense")


def test_interpolation2():
    n = 100
    n05 = 100
    x = np.arange(5)
    y = np.array([2, 3., 4., 5., 2.])
    plot_3_splines(x=x, y=y, n=n, title="Kink")

    x, y = get_substeps_adjusted(x=np.vstack((x, y)).T, n=n05).T
    plot_3_splines(x=x, y=y, n=n, title="Kink")

    x = np.linspace(-1, +1, 10001)
    y = np.abs(x)
    plot_3_splines(x=x, y=y, n=n, title="|x|")

    # x = np.array([1, 4, 20])
    # x = np.array([1, 4, 7])
    x = np.array([1, 4, 8])
    y = np.array([1, 10, 3])
    plot_3_splines(x=x, y=y, n=n, title="varying x")


def plot_3_splines(x, y, n=1000, title=""):
    cs_i0 = get_cubic_spline(x, y, mode="i0")
    cs_i1 = get_cubic_spline(x, y, mode="i1")
    cs_i2 = get_cubic_spline(x, y, mode="i2")

    x2 = np.linspace(x[0], x[-1], num=n)
    y_i0 = cs_i0(x2)
    y_i1 = cs_i1(x2)
    y_i2 = cs_i2(x2)

    fig, ax = new_fig(title=title)
    ax.plot(x2, y_i0, marker="o", markersize=2, c="orange", label="i0")
    ax.plot(x2, y_i1, marker="o", markersize=2, c="red", label="i1")
    ax.plot(x2, y_i2, marker="o", markersize=2, c="magenta", label="i2")
    ax.plot(x, y, marker="x", markersize=5, ls="", c="k")
    ax.legend()

    fig, ax = new_fig(title=title + " Derv")
    ax.plot(x2[1:], np.diff(y_i0), marker="o", markersize=2, c="orange", label="i0")
    ax.plot(x2[1:], np.diff(y_i1), marker="o", markersize=2, c="red", label="i1")
    ax.plot(x2[1:], np.diff(y_i2), marker="o", markersize=2, c="magenta", label="i2")
    ax.legend()


def test_i2_natural():
    def test(x, y):
        cs_i2 = get_cubic_spline(x, y, mode="i2")
        cs_natural = CubicSpline(x, y, bc_type="natural")
        return np.allclose(cs_i2.c, cs_natural.c)

    n = 100
    b = np.zeros(n)
    for i in range(n):
        xx = np.sort(np.random.random(10))
        yy = np.random.random(10)
        b[i] = test(x=xx, y=yy)

    log_print(b.mean())


def cumsum_diff(x0, x_diff):
    return np.cumsum(np.concatenate([x0[np.newaxis], x_diff]))


def savgol_error():
    # window_length = 21
    n_old = 20
    n_new = 2000
    x = np.linspace(0, 10, n_old)
    y = np.random.random(n_old) * 2
    xy = np.vstack((x, y)).T
    xy_fine = get_substeps_adjusted(x=xy, n=n_new)
    for window_length in [5, 7, 9, 11, 13, 15, 17, 21]:
        log_print(window_length)
        y_fine_savgol = savgol_filter(x=xy_fine[:, 1], window_length=window_length, polyorder=3, deriv=0)
        y_fine_savgol2 = savgol_filter(x=y_fine_savgol, window_length=window_length, polyorder=3, deriv=0)

        diff = np.abs(xy_fine[:, 1] - y_fine_savgol)
        diff2 = np.abs(xy_fine[:, 1] - y_fine_savgol2)
        log_print(np.max(diff), np.max(diff2))

        # fig, ax = new_fig()
        # ax.plot(y_fine_savgol, c='r', marker='o', markersize=5)
        # ax.plot(y_fine_savgol2, c='b', marker='o', markersize=5)

        # d1_savgol = np.diff(y_fine_savgol, 1)
        # d1_savgol2 = np.diff(y_fine_savgol2, 1)

        # d2_savgol = np.diff(y_fine_savgol, 2)
        # d2_savgol2 = np.diff(y_fine_savgol2, 2)
        # log_print(np.sum(np.abs(d2_savgol)), np.max(np.abs(d2_savgol)))
        # log_print(np.sum(np.abs(d2_savgol2)), np.max(np.abs(d2_savgol2)))

        # fig, ax = new_fig()
        # ax.plot(d1_savgol, c='r', marker='o', markersize=5)
        # ax.plot(d1_savgol2, c='b', marker='o', markersize=5)


def dummy1():
    n_old = 20
    n_new = 1000
    x = np.linspace(0, 10, n_old)
    y = np.random.random(n_old) * 5
    # y = np.sort(y)
    xy = np.vstack((x, y)).T
    xy_fine = get_substeps_adjusted(x=xy, n=n_new)
    xy_fine2 = get_substeps_adjusted(x=xy, n=n_new*2)

    # fig, ax = new_fig()
    # ax.plot(*xy.T, alpha=0.5, marker='x')
    # ax.plot(*xy_fine.T, alpha=0.5, marker='o')
    y_fine_filter = savgol_filter(x=xy_fine2[:, 1], window_length=7, polyorder=3, deriv=0)
    y_fine_spline = splev(xy_fine2[:, 0], splrep(xy_fine[:, 0], y=xy_fine[:, 1], s=0.001, k=3), der=0)
    y_fine_spline_i0 = get_cubic_spline(x=xy_fine[:, 0], y=xy_fine[:, 1], mode="i0")(xy_fine2[:, 0])
    # y_fine_spline_i1 = get_cubic_spline(x=xy_fine[:, 0], y=xy_fine[:, 1], mode='i1')(xy_fine2[:, 0])

    fig, ax = new_fig()
    ax.plot(*xy_fine.T, c="k", marker="o", markersize=5)
    ax.plot(xy_fine2[:, 0], y_fine_filter, c="r", marker="o", markersize=5)
    ax.plot(xy_fine2[:, 0], y_fine_spline, c="b", marker="o", markersize=5)
    # ax.plot(xy_fine2[:, 0], y_fine_spline, c='matrix', marker='s', markersize=5)

    fig, ax = new_fig()
    ax.plot(np.diff(y_fine_filter), c="r", marker="o", markersize=5)
    ax.plot(np.diff(y_fine_spline), c="b", marker="o", markersize=5)

    fig, ax = new_fig()

    d2_filter = np.diff(y_fine_filter, 2)
    d2_spline = np.diff(y_fine_spline, 2)
    d2_spline_i0 = np.diff(y_fine_spline_i0, 2)
    # d2_spline_i1 = np.diff(y_fine_spline_i1, 2)
    log_print(np.sum(np.abs(d2_filter)), np.max(np.abs(d2_filter)))
    log_print(np.sum(np.abs(d2_spline)), np.max(np.abs(d2_spline)))
    log_print(np.sum(np.abs(d2_spline_i0)), np.max(np.abs(d2_spline_i0)))
    ax.plot(d2_filter, c="r", marker="o", markersize=5)
    ax.plot(d2_spline, c="b", marker="o", markersize=5)
    # ax.plot(d2_spline_i0, c='matrix', marker='o', markersize=5)
    # ax.plot(d2_spline_i1, c='orange', marker='o', markersize=5)
    # ax.plot(np.diff(y_fine_spline), c='matrix', marker='s', markersize=5)


def dummy2():
    n_old = 20
    n_new = 1000
    n_fraction = 1
    x = np.linspace(0, 10, n_old)
    y = np.random.random(n_old) * 2
    # y = np.sort(y)
    xy = np.vstack((x, y)).T
    xy_fine = get_substeps_adjusted(x=xy, n=n_new//n_fraction)

    # fig, ax = new_fig()
    # ax.plot(*xy.T, alpha=0.5, marker='x')
    # ax.plot(*xy_fine.T, alpha=0.5, marker='o')

    y_fine_diff = np.diff(xy_fine[:, 1])
    y_fine_diff_fine = get_substeps(y_fine_diff[:, np.newaxis], n=n_fraction)[:, 0]
    x_fine_fine = get_substeps(xy_fine, n=n_fraction)[:, 0]

    idx = np.nonzero(np.abs(np.diff(y_fine_diff)) > 1e-5)[0]
    n = 5
    kernel = np.zeros(2*n)
    kernel[:n] = 1 / np.linspace(10, np.sqrt(2.25), num=n) ** 2
    kernel[n:] = 1 - kernel[:n][::-1]
    y_fine_diff2 = y_fine_diff.copy()
    for i in idx:
        d = y_fine_diff2[i+1] - y_fine_diff2[i]
        y_fine_diff2[i-n+1:i+n+1] = y_fine_diff2[i] + d * kernel

    fig, ax = new_fig()
    ax.plot(y_fine_diff)
    ax.plot(y_fine_diff2)

    # fig, ax = new_fig()
    # ax.plot(xy_fine[1:, 0], np.linalg.norm(np.diff(xy_fine, axis=0), axis=-1), alpha=0.5, marker='s')

    # y_smooth = smooth_vel(xy_fine[:, 1].copy(), kernel_size=11, iterations=101, alpha=0.5)
    # y_fine_diff_smooth = smooth_vel(y_fine_diff.copy(), kernel_size=21, iterations=3, alpha=0.1)
    # y_fine_diff_fine_smooth = smooth_vel(y_fine_diff_fine.copy(), kernel_size=21, iterations=3, alpha=0.1)

    # y_smooth_diff = np.diff(y_smooth)
    fig, ax = new_fig()
    # ax.plot(y_smooth_diff)
    # ax.plot(y_fine_diff)
    ax.plot(xy_fine[1:, 0], y_fine_diff, marker="o", c="r", alpha=0.5, markersize=5)
    ax.plot(x_fine_fine[n_fraction:], y_fine_diff_fine, marker="s", c="b", alpha=0.5, markersize=2)

    # ax.plot(y_fine_diff_fine_smooth)

    # y_fine_diff_fine_smooth_cs = np.cumsum(np.concatenate([y[:1],
    #                                                        np.repeat(y_fine_diff_fine_smooth[0], 4),
    #                                                        y_fine_diff_fine_smooth / 5 ]))
    # y_fine_diff_fine_cs = np.cumsum(np.concatenate([y[:1],
    #                                                 np.repeat(y_fine_diff_fine[0], n_fraction-1) / n_fraction,
    #                                                 y_fine_diff_fine / n_fraction]))

    y_fine_diff_cs = np.cumsum(np.concatenate([y[:1], y_fine_diff]))
    y_fine_diff2_cs = np.cumsum(np.concatenate([y[:1], y_fine_diff2]))
    # y_fine_diff_fine_smooth_cs = np.cumsum(y_fine_diff_fine_smooth / 5) + y[0]
    # y_fine_diff_fine_cs = np.cumsum(y_fine_diff_fine * 0.20181634712411706) + y[0]

    fig, ax = new_fig()
    ax.plot(*xy_fine.T, marker="o", markersize=1.5, c="b", alpha=0.5)
    # ax.plot(x_fine_fine, y_fine_diff_fine_smooth_cs, marker='o', markersize=1.5, c='r', alpha=0.5)
    # ax.plot(x_fine_fine, y_fine_diff_fine_cs, marker='o', markersize=1.5, c='matrix', alpha=0.5)
    # ax.plot(x_fine_fine, y_fine_diff_fine2_cs, marker='s', markersize=1.5, c='orange', alpha=0.5)
    ax.plot(xy_fine[:, 0], y_fine_diff_cs, marker="o", markersize=1.5, c="matrix", alpha=0.5)
    ax.plot(xy_fine[:, 0], y_fine_diff2_cs, marker="s", markersize=1.5, c="orange", alpha=0.5)
    # log_print(len(xy_fine) / len(x_fine_fine))


def dummy3():

    n_old = 21
    # n_new = 101
    x_old = np.linspace(0, 1, num=n_old)
    # x_new = np.linspace(0, 1, num=n_new)
    y_old = np.random.random(n_old)

    fig, ax = new_fig()
    ax.plot(x_old, y_old, alpha=0.5, marker="x")

    y_diff = np.diff(y_old)
    # xx = np.vstack((x_old, y_old)).T
    y_diff_fine = get_substeps(y_diff[:, np.newaxis], n=100)[:, 0]
    # x_fine = get_substeps(x_old[1:][:, np.newaxis], n=100)[:, 0]

    y_diff_fine_smooth = smooth_vel(y_diff_fine.copy(), kernel_size=11, iterations=1, alpha=1.)

    fig, ax = new_fig()
    ax.plot(y_diff_fine, alpha=0.5, marker="x")
    ax.plot(y_diff_fine_smooth, alpha=0.3, marker="s", c="k")

    # yy_cs = cumsum_diff(x_old[0], y_diff)
    yy_cs_fine = cumsum_diff(x_old[0], y_diff_fine)
    yy_cs_smooth = cumsum_diff(x_old[0], y_diff_fine_smooth)
    #
    fig, ax = new_fig()
    yy_cs = get_substeps(x=y_old[1:, np.newaxis], n=100)[:, 0]

    ax.plot(yy_cs * 80, alpha=0.5, marker="o")
    ax.plot(yy_cs_fine, alpha=0.3, marker="x")
    ax.plot(yy_cs_smooth, alpha=0.3, marker="s", c="k")

    fig, ax = new_fig()
    ax.plot(x_old, y_old, alpha=0.5)

    fig, ax = new_fig()
    ax.plot(x_old, y_old, c="k", alpha=0.5)


def smooth_vel_test():
    n_wp = 10
    n_wp_new = 13
    x = 1 + np.random.random(n_wp) * 0.2
    xp = np.linspace(start=0, stop=1, num=n_wp)
    # s = np.cumsum(np.random.random(n_wp_new))
    # s /= s[-1]
    s = np.linspace(0, 1, n_wp_new)

    x_cs = np.cumsum(x)
    x_cs_smooth = np.interp(x=s, xp=xp, fp=x_cs)
    x_cs_smooth = np.concatenate([x_cs_smooth[:1], np.diff(x_cs_smooth)])

    x_smooth = np.interp(x=s, xp=xp, fp=x)

    fig, ax = new_fig()
    # ax.plot(s)
    ax.plot(x_cs_smooth, label="cumsum")
    ax.plot(x_smooth, label="normal")
    ax.legend()


def smooth_vel(v, kernel_size=9, iterations=1, alpha=1.):
    # kernel = np.array([1/2, 0, 1/2])
    # kernel = np.array([1/6, 1/3, 0, 1/3, 1/6])
    # kernel = np.array([1/12, 1/8, 1/8, 1/6, 0, 1/6, 1/8, 1/8, 1/12])

    n = len(v)
    k2 = kernel_size // 2

    nrm = norm()
    x = np.linspace(-3, +3, kernel_size)
    kernel = nrm.pdf(x)
    kernel[k2] = 0
    kernel /= np.sum(kernel)
    kernel[k2] = -1

    # alpha = 1
    for _ in range(iterations):
        for i in range(k2, n-k2):
            v_window = v[i-k2:i+k2+1]
            mean_i = np.mean(v_window)
            diff = v[i] - mean_i
            v_window += kernel * alpha * diff

    return v
