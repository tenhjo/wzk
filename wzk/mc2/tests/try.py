import numpy as np

from wzk import mc2


def try_plot_bimg():
    vis = mc2.Visualizer()
    limits = np.zeros((3, 2))
    limits[:, 1] = 1
    limits += 0.33
    img = np.ones((5, 5, 5), dtype=bool)
    mc2.plot_cube(vis=vis, limits=limits, mode="lines", color="grey")
    mc2.plot_bimg(vis=vis, img=img, limits=limits, mode="voxel", color="grey")
