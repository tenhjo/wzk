import numpy as np


from wzk import bimage


def test_get_sphere_stencil():
    from wzk import new_fig
    r = 10
    voxel_size = 0.3
    n_dim = 2

    inner, outer = bimage.get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    fig, ax = new_fig()
    ax.imshow(inner)

    n_dim = 3
    inner, outer = bimage.get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    fig, ax = new_fig()
    ax.imshow(inner.sum(axis=-1), cmap="gray_r")


def test_mesh2bimg():
    p = np.random.random((10, 2))
    limits = np.array([[0, 1],
                       [0, 1]])
    img = bimage.mesh2bimg(p=p, shape=(64, 64), limits=limits)

    from wzk.mpl2 import new_fig, imshow

    fig, ax = new_fig()
    ax.plot(*p.T, ls="", marker="o")
    imshow(ax=ax, img=img, mask=~img, limits=limits)


def test_spheres2bimg():
    n = 10
    shape = (256, 256, 256)

    limits = np.array([[-1, 2],
                       [-1, 2],
                       [-1, 2]])
    x = np.random.random((n, 3))
    r = np.random.uniform(low=0.1, high=0.2, size=n)
    img = bimage.spheres2bimg(x=x, r=r, shape=shape, limits=limits)

    from wzk import mc2
    p = mc2.Visualizer()
    mc2.plot_bimg(vis=p, img=img, limits=limits, h=None)


if __name__ == "__main__":
    test_spheres2bimg()
    # test_get_sphere_stencil()
    # test_mesh2bimg()
