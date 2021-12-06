import numpy as np
import wzk.cpppy.MinSphere.MinSphere as MinSphere


def min_sphere(x, r):
    n, d = np.shape(x)
    res = np.zeros(d + 1, dtype='f4', order='c')
    if d < 2 or 4 < d:
        raise ValueError
    if d == 2:
        MinSphere.min_sphere2(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 3:
        MinSphere.min_sphere3(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    elif d == 4:
        MinSphere.min_sphere3(x=x.astype(dtype='f4', order='c'), r=r.astype(dtype='f4', order='c'), n=n, res=res)
    else:
        raise ValueError

    return res[:-1], res[-1]


def test_min_sphere():
    from wzk.mpl import new_fig, plot_circles
    n = 10
    d = 2
    r = np.zeros(n)
    x = np.random.random((n, d))
    x0, r0 = min_sphere(x=x, r=r)

    fig, ax = new_fig(aspect=1)
    ax.plot(*x.T, marker='o', ls='', color='red')
    ax.plot(*x0, marker='x', ls='', color='blue')
    plot_circles(ax=ax, x=x0, r=r0, alpha=0.5, color='blue')


if __name__ == '__main__':
    test_min_sphere()