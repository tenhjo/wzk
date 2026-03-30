import numpy as np

from wzk import limits as limits2
from wzk import math2
from wzk.mpl2 import DraggableEllipseList, new_fig


def draggable_configurations(x: np.ndarray, limits, circle_ratio=1 / 3, **kwargs):
    x = np.squeeze(x)
    n_wp, n_dof = x.shape

    if n_dof > 5:
        n_cols, n_rows = math2.get_mean_divisor_pair(n=n_dof)
    else:
        n_cols, n_rows = 1, n_dof

    fig, axes = new_fig(n_rows=n_rows, n_cols=n_cols, share_x=True)
    fig.subplots_adjust(hspace=0.0, wspace=0.2)

    axes.flatten()[-1].set_xlim([-1, n_wp])
    axes.flatten()[-1].set_xticks(np.arange(n_wp))
    axes.flatten()[-1].set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(n_wp)])

    for ax, limits_i in zip(axes.flatten(), limits, strict=False):
        limits_i_larger = limits2.add_safety_limits(limits=limits_i, factor=0.05)

        y_ticks = np.linspace(limits_i[0], limits_i[1], 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{v:.2f}" for v in y_ticks])
        ax.set_ylim(limits_i_larger)

    x_temp = np.arange(n_wp)

    h_lines = [ax.plot(x_temp, x_i, **kwargs)[0] for ax, x_i in zip(axes.flatten(), x.T, strict=False)]

    def update_wrapper(i):
        def __update():
            y_i = dgel_list[i].get_xy()[:, 1]
            h_lines[i].set_ydata(y_i)

        return __update

    dgel_list = [
        DraggableEllipseList(
            ax=ax,
            vary_xy=(False, True),
            xy=np.vstack([x_temp, x_i]).T,
            width=circle_ratio,
            height=-1,
            callback=update_wrapper(i),
            **kwargs,
        )
        for i, (ax, x_i, limits_i) in enumerate(zip(axes.flatten(), x.T, limits, strict=False))
    ]

    return fig, axes, dgel_list, h_lines


class DraggableConfigSpace:
    def __init__(self, x: np.ndarray, limits, circle_ratio=1 / 3, **kwargs):
        self.x = np.squeeze(x)
        self.n_wp, self.n_dof = self.x.shape

        self.fig, self.axes, self.dgel_list, self.h_lines = draggable_configurations(
            x=self.x, limits=limits, circle_ratio=circle_ratio, **kwargs
        )

    def set_callback(self, callback):
        for dgel in self.dgel_list:
            dgel.set_callback_drag(callback=callback)

    def add_callback(self, callback):
        for dgel in self.dgel_list:
            dgel.add_callback_drag(callback=callback)

    def get_x(self):
        return np.array([dgel.get_xy()[:, 1] for dgel in self.dgel_list]).T

    def set_x(self, x):
        self.x = x
        for dgel, h_line_i, x_i in zip(self.dgel_list, self.h_lines, self.x.T, strict=False):
            dgel.set_xy(y=x_i)
            h_line_i.set_ydata(x_i)
