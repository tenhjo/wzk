from unittest import TestCase

import numpy as np
from wzk.mpl2.figure import new_fig, plt
from wzk.mpl2 import Patches2
from wzk.mpl2 import size_units2points


class Test(TestCase):

    def test_transform(self):
        height = 0.1
        width = 1
        fig, ax = new_fig()
        ax.set_aspect("equal")
        ax.set_xlim([-width * 1.2, width * 1.2])
        ax.set_ylim([-width * 1.2, width * 1.2])

        rec = plt.Rectangle((0, 0), width=width, height=height, alpha=0.8,
                            transform=Patches2.get_aff_trafo(xy1=(0.5, 0.5), theta=90, por=(0.5, 0.05), ax=None))
        ax.add_patch(rec)

        Patches2.do_aff_trafo(patch=rec, xy=(0, 0), por=(0, 0.1), theta=90)
        plt.pause(1)
        Patches2.do_aff_trafo(patch=rec, xy=(0, 1), por=(0.5, 0.1), theta=45)
        plt.pause(1)
        Patches2.do_aff_trafo(patch=rec, xy=(0, 0), por=(0.5, 0.05), theta=135)
        self.assertTrue(True)

    def test_RelativeArrowPatch(self):

        fig, ax = new_fig(aspect="equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        arrow1 = Patches2.RelativeFancyArrow(0.5, 0.5, 0.0, 0.5, width=1, head_length=0.5, head_width=2)
        arrow2 = Patches2.RelativeFancyArrow(0.5, 0.5, 0.5, 0.0, width=1, head_length=0.5, head_width=2)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)

        ax.plot(0.5, 0.5, marker="o", markersize=size_units2points(size=0.5, ax=ax), alpha=0.5, zorder=100)
        self.assertTrue(True)

    def test_AbsoluteFancyBboxPatch(self):
        fig, ax = new_fig(aspect="equal")
        ax.add_patch(Patches2.FancyBbox(xy=(0.1, 0.1), boxstyle='Round4',  # noqa
                                        height=0.5, width=0.5, pad=0.1, corner_size=0))

    def test_RoundedPolygon(self):
        xy = np.array([(0, 0), (0.25, 0), (0.5, -0.25), (0.75, 0),
                       (1, 0), (1, 0.25), (1.25, 0.5), (1, 0.75),
                       (1, 1), (0.75, 1), (0.5, 1.25), (0.25, 1),
                       (0, 1), (0, 0.75), (-0.25, 0.5), (0, 0.25)])
        rp = Patches2.RoundedPolygon(xy=xy, pad=0.1, facecolor="red", edgecolor="magenta", lw=3)

        fig, ax = new_fig(aspect="equal")

        ax.add_patch(rp)
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)

    def test_CurlyBrace(self):
        fig, ax = new_fig()
        ax.axis("off")

        for i, (w, h, c) in enumerate(zip(
                np.linspace(.1, .18, 4),
                np.linspace(.95, .5, 4),
                np.linspace(.1, .5, 4))):
            x = i * .1
            lw = 3 * i + 1
            # noinspection PyUnresolvedReferences
            col = plt.cm.plasma(i / 8)
            brace = Patches2.CurlyBrace(p=(x + 0.05, h / 2), x0=(x, 0.1), x1=(x, 0.1 + h), lw=lw,
                                        curliness=c, color=col)
            ax.add_artist(brace)

        fig, ax = new_fig()
        ax.axis("off")
        ax.set_aspect(1)
        for i in range(10):
            x = np.random.random((3, 2))
            x = np.sort(x, axis=0)
            brace = Patches2.CurlyBrace(x[1], x[0], x[2])
            ax.add_artist(brace)

        fig, ax = new_fig()
        ax.set_aspect(1)
        brace = Patches2.CurlyBrace(p=(2.5, 1.5), x0=(1, 1), x1=(4, 5), color="b")
        ax.add_artist(brace)
        ax.plot(*brace.verts.T, ls="", marker="o", alpha=0.5)
        ax.plot(*brace.verts[[0, 1, 5, 6, 0]].T)

        text = ["a", "ac0", "ac1", "c", "cb0", "cb1", "b"]
        for i, v in enumerate(brace.verts):
            ax.annotate(xy=v, text=str(i) + "\n" + text[i], ha="center", va="center", zorder=100)

        self.assertTrue(True)


if __name__ == "__main__":
    test = Test()
    test.test_transform()
    test.test_RelativeArrowPatch()
    test.test_AbsoluteFancyBboxPatch()
    test.test_RoundedPolygon()
    test.test_CurlyBrace()
