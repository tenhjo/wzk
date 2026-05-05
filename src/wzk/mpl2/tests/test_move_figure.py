from unittest import TestCase

from wzk.mpl2 import move_figure
from wzk.mpl2.backend import plt


class Test(TestCase):
    def test_move_fig(self):
        fig, ax = plt.subplots()
        move_figure.move_fig(fig=fig, position="top right")

        fig, ax = plt.subplots()
        move_figure.move_fig(fig=fig, position="top left")

        fig, ax = plt.subplots()
        move_figure.move_fig(fig=fig, position="bottom right")

        fig, ax = plt.subplots()
        move_figure.move_fig(fig=fig, position="bottom left")
        plt.draw()
        plt.close("all")

        for i in range(1, 10):
            fig, ax = plt.subplots()
            move_figure.move_fig(fig=fig, position=(3, 3, i))

        plt.draw()
        plt.close("all")

        self.assertTrue(True)
