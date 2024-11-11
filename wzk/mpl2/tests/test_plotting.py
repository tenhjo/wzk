from unittest import TestCase

import numpy as np
from wzk.mpl2 import plotting


class Test(TestCase):

    def test_plot_projections_2d(self):
        n_dof = 4
        n = 100
        x = np.random.normal(size=(n, n_dof))
        limits = np.array([[-4, +4]] * n_dof)
        plotting.plot_projections_2d(x=x, dim_labels="xyzuvw", limits=limits, aspect=1)

        n_dof = 3
        n = 100
        x = np.random.normal(size=(n, n_dof))
        limits = np.array([[-4, +4]] * n_dof)
        fig, ax = plotting.new_fig(n_rows=n_dof, n_cols=1, aspect=1)
        plotting.plot_projections_2d(ax=ax, x=x, limits=limits, aspect=1)

        self.assertTrue(True)

    def test_imshow(self):
        arr = np.arange(45).reshape(5, 9)
        mask = arr % 2 == 0
        limits = np.array([[0, 5],
                           [0, 9]])

        arr2 = arr.copy()
        arr2[mask] = 0
        print(arr2)

        fig, ax = plotting.new_fig(title="upper, ij")
        plotting.imshow(ax=ax, img=arr, limits=limits, cmap="Blues", mask=mask, origin="upper", axis_order="ij->xy")

        fig, ax = plotting.new_fig(title="upper, ji")
        plotting.imshow(ax=ax, img=arr, limits=limits, cmap="Blues", mask=mask, origin="upper", axis_order="ij->yx")

        fig, ax = plotting.new_fig(title="lower, ij")
        plotting.imshow(ax=ax, img=arr, limits=limits, cmap="Blues", mask=mask, origin="lower", axis_order="ij->xy")

        fig, ax = plotting.new_fig(title="lower, ji")
        plotting.imshow(ax=ax, img=arr, limits=limits, cmap="Blues", mask=mask, origin="lower", axis_order="ij->yx")

        fig, ax = plotting.new_fig(title="lower, ji")
        h = plotting.imshow(ax=ax, img=arr, limits=limits, cmap="Blues", mask=mask, origin="lower", axis_order="ij->yx")
        plotting.imshow(h=h, img=arr, mask=arr % 2 == 1, cmap="Reds", axis_order="ij->yx")

        fig, ax = plotting.new_fig(aspect=1)
        arr = np.arange(42).reshape(6, 7)
        plotting.imshow(ax=ax, img=arr, limits=None, cmap="Blues", mask=arr % 2 == 0, vmin=0, vmax=100)

        self.assertTrue(True)

    def test_grid_lines(self):
        fig, ax = plotting.new_fig()

        limits = np.array([[0, 4],
                           [0, 5]])
        plotting.set_ax_limits(ax=ax, limits=limits, n_dim=2)
        plotting.grid_lines(ax=ax, start=0.5, step=(0.2, 0.5), limits=limits, color="b", ls=":")

        self.assertTrue(True)
