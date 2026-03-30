
import numpy as np


from wzk import bimage


def test_get_sphere_stencil():
    r = 10
    voxel_size = 0.3
    n_dim = 2

    inner, outer = bimage.get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    assert inner.dtype == bool
    assert outer.dtype == bool
    assert inner.shape == outer.shape
    assert inner.ndim == n_dim
    assert inner.any()
    assert outer.any()

    n_dim = 3
    inner, outer = bimage.get_sphere_stencil(r=r, voxel_size=voxel_size, n_dim=n_dim)
    assert inner.dtype == bool
    assert outer.dtype == bool
    assert inner.shape == outer.shape
    assert inner.ndim == n_dim
    assert inner.any()
    assert outer.any()


def test_mesh2bimg():
    p = np.random.random((10, 2))
    limits = np.array([[0, 1],
                       [0, 1]])
    img = bimage.mesh2bimg(p=p, shape=(64, 64), limits=limits)
    assert img.dtype == bool
    assert img.shape == (64, 64)
    assert img.any()


def test_spheres2bimg():
    n = 10
    shape = (256, 256, 256)

    limits = np.array([[-1, 2],
                       [-1, 2],
                       [-1, 2]])
    x = np.random.random((n, 3))
    r = np.random.uniform(low=0.1, high=0.2, size=n)
    img = bimage.spheres2bimg(x=x, r=r, shape=shape, limits=limits)
    assert img.dtype == bool
    assert img.shape == shape
    assert img.any()


def test_create_stencil_dict():
    stencil_dict = bimage.create_stencil_dict(voxel_size=0.2, n_dim=2)
    assert len(stencil_dict) > 0
    for d, stencil in stencil_dict.items():
        assert isinstance(d, int)
        assert stencil.dtype == bool
        assert stencil.ndim == 2
        assert stencil.shape == (d, d)
        assert stencil.any()



if __name__ == "__main__":
    test_spheres2bimg()
    # test_get_sphere_stencil()
    # test_mesh2bimg()
