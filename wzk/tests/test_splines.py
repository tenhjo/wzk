
from wzk.logger import log_print
import numpy as np

from wzk import trajectory, splines, mpl2


def test_basis_function():

    n_points = 10
    p = np.random.random((n_points, 3))
    nurbs = splines.NURBS(p=p, degree=0)

    u = np.linspace(0, 1, 100)
    for degree in range(5):
        nurbs.degree = degree
        nurbs.set_knotvector(k=np.linspace(0, 1, n_points+degree+1))
        fig, ax = mpl2.new_fig()
        ax.set_aspect(1)
        for i in range(10):
            n_in = nurbs.n_in(u=u, i=i, n=degree)
            ax.plot(u, n_in+i*0.001,  label=f"{degree} - {i}")

        ax.legend()


def test_random():
    n = 3
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = splines.NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    fig, ax = mpl2.new_fig()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(*x.T, color="black", marker="o", markersize=3)
    ax.plot(*p.T, color="blue", marker="o")


def test_random_jac():
    n = 5
    p = np.random.random((n, 2))
    u = np.linspace(0, 1, 20)

    nurbs = splines.NURBS(p=p, degree=3)
    x = nurbs.evaluate(u)

    def length_jac(xx):
        steps = xx[..., 1:, :] - xx[..., :-1, :]
        return steps[..., :-1, :] - steps[..., +1:, :]  # do_dx

    fig, ax = mpl2.new_fig()
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    hx = ax.plot(*x.T, color="black", marker="o", markersize=3)[0]
    hp = ax.plot(*p.T, color="blue", marker="o")[0]

    dx_dp = nurbs.evaluate_jac(u)[..., 1:-1, :]  # does not depend on p
    for i in range(100):
        do_dx = length_jac(x)
        do_dp = (do_dx[:, np.newaxis, :] * dx_dp[:, :, np.newaxis]).sum(axis=0)
        nurbs.p[1:-1] -= do_dp[1:-1]
        x = nurbs.evaluate(u)

        hx.set_data(*x.T)
        hp.set_data(*nurbs.p.T)
        mpl2.plt.pause(0.1)
        # input ('press key for next gradient step')


def test_unit_circle():
    k = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4], dtype=float)
    p = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0]])

    sq22 = np.sqrt(2)/2
    w = np.array([1, sq22, 1, sq22, 1, sq22, 1, sq22, 1])
    nurbs = splines.NURBS(p=p, k=k, w=w, degree=2)
    u = np.linspace(0, 1, 90)
    x = nurbs.evaluate(u=u)
    j = nurbs.evaluate_jac(u=u)
    log_print("x", x)
    log_print("j", j)
    fig, ax = mpl2.new_fig()
    ax.set_aspect(1)
    ax.plot(*x.T, color="black", marker="o")
    ax.plot(*p.T, color="blue", marker="o", markersize=3)


def test_gui():
    n = 5
    p = np.random.random((n, 2))
    u = np.linspace(0., 1, 20)

    nurbs = splines.NURBS(p=p, degree=3)
    x_spline = nurbs.evaluate(u)

    uv = trajectory.to_spline(x_spline, n_c=5)
    mpl2.close_all()
    fig, ax = mpl2.new_fig(aspect="equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    dcl = mpl2.DraggableCircleList(ax=ax, xy=p, radius=0.02, color="r")

    h_spline = ax.plot(*x_spline.T, color="k", marker="o", lw=3)[0]
    h_base = ax.plot(*p.T, color="r", marker="o", lw=2)[0]
    h_uv = ax.plot(*uv.T, color="b", marker="o", lw=2)[0]

    def update(*args):  # noqa
        x_base = dcl.get_xy()

        nurbs.p = x_base
        _x_spline = nurbs.evaluate(u)
        _uv = trajectory.to_spline(_x_spline, n_c=5)

        h_base.set_xdata(x_base[:, 0])
        h_base.set_ydata(x_base[:, 1])
        h_spline.set_xdata(_x_spline[:, 0])
        h_spline.set_ydata(_x_spline[:, 1])
        h_uv.set_xdata(_uv[:, 0])
        h_uv.set_ydata(_uv[:, 1])

    dcl.set_callback_drag(update)


if __name__ == "__main__":
    # pass
    # pass
    # test_random_jac()
    # test_basis_function()
    # test_unit_circle()
    # test_random()
    test_gui()
