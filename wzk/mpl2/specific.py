import numpy as np

from wzk.mpl2 import new_fig


def plot_loc_rot_difference(d_loc, d_rot, x, xlabel, title=None):
    fig, ax1 = new_fig(title=title)
    ax2 = ax1.twinx()
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Error TCP [Meter]")
    ax2.set_ylabel("Error TCP [Degree]")
    ax1.plot(x, d_loc, marker="s", color="blue", label="location")
    ax2.plot(x, np.rad2deg(d_rot), marker="o", color="red", label="orientation")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
