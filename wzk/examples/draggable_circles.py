import numpy as np
from wzk import mpl2, bimage

n = 4
shape = (64, 64)
x = np.random.random((n, 2))
r = np.random.uniform(low=0.1, high=0.2, size=n)
limits = np.array([[0., 1.],
                   [0., 1.]])

fig, ax = mpl2.new_fig(aspect="equal")
dcl = mpl2.DraggableCircleList(ax=ax, xy=x, radius=r, alpha=0.5, color="red")
img = bimage.spheres2bimg(x=x, r=r, shape=shape, limits=limits)

p = dcl.get_xy()
h_x = ax.plot(*x.T, marker="o", color="blue")[0]
h_img = mpl2.imshow(img=img, mask=~img, limits=limits, ax=ax, cmap="binary", zorder=-1)


def callback(*args):  # noqa
    x_cur = dcl.get_xy()
    img_cur = bimage.spheres2bimg(x=x_cur, r=r, shape=shape, limits=limits)

    h_x.set_data(x_cur.T)
    mpl2.imshow(h=h_img, img=img_cur, mask=~img_cur, limits=limits, ax=ax, cmap="binary", zorder=-1)


dcl.set_callback_drag(callback=callback)
