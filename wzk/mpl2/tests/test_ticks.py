from unittest import TestCase

from wzk.mpl2 import new_fig, plt, ticks


class Test(TestCase):
    def test_set_ticks_position(self):
        fig, ax = new_fig()
        positions = ["left", "top", "right", "bottom",
                     "all", "none",
                     "default", "inverse",
                     ("bottom", "right"), ("top", "left"),
                     ("bottom", "top"), ("left", "right"),
                     ("left", "bottom", "right"),
                     ("top", "bottom", "right"),
                     ("top", "left", "right"),
                     ("top", "left", "bottom")]

        for p in positions:
            ticks.set_ticks_position(ax=ax, position=p)
            ticks.set_labels_position(ax=ax, position=p)
            ax.cla()
            ax.text(0.5, 0.5, "".join(p))
            plt.draw()
        self.assertTrue(True)

    def test_remove_ticks(self):
        fig, ax = new_fig()

        plt.draw()
        ticks.remove_ticks(ax=ax, v=[0, 0.4], axis="both")
        self.assertTrue(True)

    def test_add_ticks(self):
        fig, ax = new_fig()

        ticks.set_ticks_position(ax=ax, position="all")
        ticks.set_labels_position(ax=ax, position="all")
        plt.draw()

        ticks.add_ticks(ax=ax, ticks=0.55, labels="wow", axis="x")
        ticks.add_ticks(ax=ax, ticks=(0.3, 0.5), labels=("wtf", "ftw"), axis="y")

    def test_change_tick_appearance(self):

        fig, ax = new_fig()
        ticks.set_ticks_position(ax=ax, position="all")
        ticks.set_labels_position(ax=ax, position="all")

        plt.draw()
        ticks.change_tick_appearance(ax, position="bottom", v=3, size=40, color="red")
        ticks.change_tick_appearance(ax, position="top", v=0.4, size=30, color="blue")

        plt.draw()
        ticks.change_tick_appearance(ax, position="left", v=2, size=40, color="green")
        ticks.change_tick_appearance(ax, position="right", v=0.6, size=30, color="magenta")

        ax.text(0.5, 0.5, "Each Axis should have one larger tick in different colors", ha="center", weight="bold")
        ax.annotate("bottom", (0.6, 0), (0.6, 0.4), arrowprops=dict(arrowstyle="->"))
        ax.annotate("top", (0.4, 1), (0.4, 0.6), arrowprops=dict(arrowstyle="->"))
        ax.annotate("left", (0, 0.4), (0.4, 0.4), arrowprops=dict(arrowstyle="->"))
        ax.annotate("right", (1, 0.6), (0.6, 0.6), arrowprops=dict(arrowstyle="->"))
        self.assertTrue(True)

    def test_transform_tick_labels(self):
        fig, ax = new_fig()
        ticks.transform_tick_labels(ax=ax, xt=0, yt=-0.1, axis="x")
        ticks.transform_tick_labels(ax=ax, xt=0.4, yt=0, rotation=45, axis="y")

        fig, ax = new_fig()
        ticks.transform_tick_labels(ax=ax, xt=0, yt=0, rotation=90, axis="both", ha="center", va="center")

        self.assertTrue(True)

    def test_elongate_ticks_and_labels(self):

        fig, ax = new_fig()
        ticks.elongate_ticks_and_labels(ax, newline=(10, [1, 2]), labels=None, axis="x")
        ticks.elongate_ticks_and_labels(ax, newline=(10, [1, 2]), labels=None, axis="y")
        ticks.elongate_ticks_and_labels(ax, newline=(10, [1, 2]), labels=["a", "b", "c", "d", "e", "f"], axis="x")


if __name__ == "__main__":
    test = Test()
    # test.test_set_ticks_position()
    # test.test_remove_ticks()
    # test.test_remove_ticks()
    test.test_change_tick_appearance()
    # test.test_elongate_ticks_and_labels()
