from unittest import TestCase

import numpy as np
from wzk.mpl2 import legend, new_fig


class Test(TestCase):

    def test_annotate_arrow(self):
        fig, ax = new_fig()
        from_xy = np.vstack([np.ones(5), np.arange(5)]).T / 5
        to_xy = np.vstack([np.arange(5) + 0.1, np.arange(1, 6)]).T / 5
        legend.annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0, color="r")
        legend.annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0.04, color="k")
        legend.annotate_arrow(ax=ax, xy0=from_xy, xy1=to_xy, offset=0.08, color="b")
        legend.annotate_arrow(ax=ax, xy0=(0.2, 0.2), xy1=(0.4, 0.4), offset=0.08, color="k")

        self.assertTrue(True)

    def test_make_legend_arrow_wrapper(self):
        fig, ax = new_fig(aspect="equal")
        arrow_a = ax.arrow(0, 0, np.cos(0.3), np.sin(0.3), head_width=0.1, color="r", label="A")
        arrow_b = ax.arrow(0, 0, np.cos(0.5), np.sin(0.5), head_width=0.05, color="b", label="B")
        arrow_c = ax.arrow(0, 0, np.cos(0.9), np.sin(0.9), head_width=0.05, color="m", label="C")
        ax.legend([arrow_a, arrow_b, arrow_c], ["My label - A", "Dummy - B", "Another One - C"],
                  handlelength=1, borderpad=1.2, labelspacing=1.2,
                  handler_map={legend.patches.FancyArrow: legend.make_legend_arrow_wrapper(theta=0.9,
                                                                                           label2theta_dict={"A": 0.3,
                                                                                                             "B": 0.5})}
                  )

        self.assertTrue(True)
