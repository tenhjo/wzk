from unittest import TestCase

import numpy as np

from wzk import random2


class Test(TestCase):
    def test_fun2n(self):

        def fun(n):
            r = 0.5
            w = 0.05

            x = np.random.uniform(low=-1, high=+1, size=(n, 2))
            d = np.linalg.norm(x, axis=-1)
            b = np.logical_and(r-w < d, d < r+w)
            return x[b]

        nn = 1000
        y = random2.fun2n(fun, n=nn, log_level=1)

        self.assertTrue(len(y) == nn)

        # fig, ax = mpl2.new_fig(aspect="equal")
        # ax.plot(*y.T, 'ro')
