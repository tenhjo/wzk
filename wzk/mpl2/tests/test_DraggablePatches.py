from unittest import TestCase

import numpy as np

from wzk.mpl2 import new_fig, close_all
from wzk.mpl2.DraggablePatches import (DraggableCircle, DraggableRectangle, DraggableEllipse,
                                       DraggableEllipseList, DraggableRectangleList,
                                       DraggableFrame)


verbose = 1


class TestDraggableCircle(TestCase):

    def test_circles(self):
        fig, ax = new_fig(aspect="equal", title="Circles")

        c1 = DraggableCircle(xy=(.2, .4), radius=.1, ax=ax, color="y", alpha=0.5, hatch="///")
        c2 = DraggableCircle(xy=(.2, .4), radius=.1, ax=ax, color="r")

        c1.set_limits(limits=np.array(((0.1, 0.2),
                                       (0.1, 0.2))))
        c2.set_color("g")
        c2.set_center(xy=(0.4, 0.4))  # noqa
        c2.set_radius(radius=0.02)  # noqa

        if verbose == 0:
            close_all()
        self.assertTrue(True)


class TestDraggableEllipse(TestCase):
    def test_ellipses2spheres(self):
        fig, ax = new_fig(aspect="auto", title="Ellipses -> Spheres")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 200)
        DraggableEllipse(ax=ax, xy=(25, 25), width=10, height=-1, color="b")
        DraggableEllipse(ax=ax, xy=(50, 50), width=-1, height=50, color="r")

        if verbose == 0:
            close_all()
        self.assertTrue(True)


class TestDraggablePatchList(TestCase):

    def test_ellipses(self):
        fig, ax = new_fig(aspect="equal", title="Ellipses + Rectangles")
        DraggableEllipse(ax=ax, xy=(0.5, 0.6), width=0.1, height=0.2, vary_xy=(False, True), color="k")
        DraggableRectangle(ax=ax, xy=(0.6, 0.5), width=0.2, height=0.1, vary_xy=(True, False), color="k")

        DraggableEllipseList(ax=ax, xy=np.random.uniform(low=0.1, high=0.4, size=(3, 2)),
                             width=0.06, height=0.09, alpha=0.7, color="b")
        DraggableRectangleList(ax=ax, xy=np.random.uniform(low=0.6, high=0.9, size=(3, 2)),
                               width=0.06, height=0.09, alpha=0.7, color="r")

        if verbose == 0:
            close_all()

        self.assertTrue(True)


class TestDraggableFrame(TestCase):

    def test_frame(self):
        fig, ax = new_fig(aspect="equal")
        df = DraggableFrame(ax=ax, xy=np.zeros(2), scale=0.4, color="red")
        df.update_x(None)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        def callback33(dgr):
            f = dgr.get_frame()  # noqa

        df.add_callback_drag(callback33)
        df.update_x(None)
        df.dp_list[0].on_release(1)
        if verbose == 0:
            close_all()
        self.assertTrue(True)


if __name__ == "__main__":
    test = TestDraggableCircle()
    test.test_circles()

    test = TestDraggableEllipse()
    test.test_ellipses2spheres()

    test = TestDraggablePatchList()
    test.test_ellipses()

    test = TestDraggableFrame()
    test.test_frame()
