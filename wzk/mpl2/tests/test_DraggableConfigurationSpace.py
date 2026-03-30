
from wzk.logger import log_print
from unittest import TestCase

import numpy as np

from wzk.mpl2.DraggableConfigurationSpace import DraggableConfigSpace


verbose = 1


class Test(TestCase):

    def test_DraggableConfigSpace(self):
        x = np.random.random((20, 3))
        limits = np.zeros((x.shape[-1], 2))
        limits[:, 1] = 1
        dcs = DraggableConfigSpace(x=x, limits=limits, circle_ratio=1 / 4, color="k")

        log_print(dcs.get_x().shape)

        dcs.set_x(x=np.zeros((20, 3)))
        dcs.set_x(x=np.ones((20, 3)))


if __name__ == "__main__":
    test = Test()
    test.test_DraggableConfigSpace()
