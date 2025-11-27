from unittest import TestCase
import numpy as np

from wzk import trajectory, testing


class Test(TestCase):
    def test_get_substeps(self):

        # A
        x = np.array([0, 2, 5]).reshape(-1, 1)
        # A1
        n = 2
        x_ss = trajectory.get_substeps(x=x, n=n)
        x_true = np.array([0, 1, 2, 3.5, 5]).reshape(-1, 1)
        self.assertTrue(testing.compare_arrays(x_ss, x_true))
        # A2
        n = 4
        x_true = np.array([0, 0.5, 1, 1.5, 2, 2.75, 3.5, 4.25, 5]).reshape(-1, 1)
        x_ss = trajectory.get_substeps(x=x, n=n)
        self.assertTrue(testing.compare_arrays(x_ss, x_true))

        # B
        x = np.array([[0, 1],
                      [1, 1],
                      [2, 2]])
        # B1
        n = 2
        x_ss = trajectory.get_substeps(x=x, n=n)
        x_true = np.array([[0, 1],
                           [0.5, 1],
                           [1, 1],
                           [1.5, 1.5],
                           [2, 2]])
        self.assertTrue(testing.compare_arrays(x_ss, x_true))
        # B2
        x_ss = trajectory.get_substeps(x=x, n=0)
        self.assertTrue(testing.compare_arrays(x_ss, x))
        # B3
        x_ss = trajectory.get_substeps(x=x, n=1)
        self.assertTrue(testing.compare_arrays(x_ss, x))

    def test_inner2full(self):

        def __assert(full, shape):
            self.assertTrue(np.all(full[..., 0, :] == 1))
            self.assertTrue(np.all(full[..., -1, :] == 2))
            self.assertTrue(full.shape == shape)

        inner = np.full((10, 5, 3), 0)
        start = np.full((1, 1, 3), 1)
        end = np.full((1, 1, 3), 2)
        __assert(full=trajectory.inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

        inner = np.full((10, 5, 3), 0)
        start = np.full((1, 3), 1)
        end = np.full((1, 3), 2)
        __assert(full=trajectory.inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

        inner = np.full((10, 5, 3), 0)
        start = np.full(3, 1)
        end = np.full(3, 2)
        __assert(full=trajectory.inner2full(inner=inner, start=start, end=end),
                 shape=(10, 7, 3))

    def test_get_substeps_adjusted(self):
        x = np.array([0, 2, 10, 19])[:, np.newaxis]
        x2 = trajectory.get_substeps_adjusted(x=x, n=20)
        true = np.arange(20)[:, np.newaxis]
        self.assertTrue(testing.compare_arrays(x2, true))

        x = np.deg2rad(np.array([[-150], [+150], [90], [-20]]))
        x2 = trajectory.get_substeps_adjusted(x=x, n=12, is_periodic=[True])
        x2 = np.rad2deg(x2)
        true = np.array([-150, -170, 170.,
                         150, 130, 110,
                         90, 68, 46, 24, 2,
                         -20])[:, np.newaxis]
        self.assertTrue(testing.compare_arrays(x2, true))

    def test_get_path_adjusted(self):
        n0 = 3
        n1 = 300
        n_dof = 2
        x0 = np.random.random((n0, n_dof))
        x1 = trajectory.get_path_adjusted(x=x0, n=n1)

        from wzk.mpl2 import new_fig

        fig, ax = new_fig(aspect="equal")
        ax.plot(*x0.T, marker="o")
        ax.plot(*x1.T, marker="s")
