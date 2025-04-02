import math
import numpy as np
from scipy.spatial import ConvexHull

from wzk import printing, np2, math2, random2


def get_ortho_star_2d(x):
    assert x.shape[-1] == 2
    x4 = np.zeros(x.shape[:-1] + (4, 2))
    x4[..., 0, :] = x.copy()

    x4[..., 1, 0] = -x[..., 1]
    x4[..., 1, 1] = x[..., 0]

    x4[..., 2, 0] = -x[..., 0]
    x4[..., 2, 1] = -x[..., 1]

    x4[..., 3, 0] = x[..., 1]
    x4[..., 3, 1] = -x[..., 0]

    return x4


def arccos2(c):
    c = np.clip(c, a_min=-1, a_max=+1)
    return np.arccos(c)


def get_arc(xy, radius, theta0=0., theta1=2 * np.pi, n=0.01):

    theta0, theta1 = theta_wrapper(theta0=theta0, theta1=theta1)
    n = angle_resolution_wrapper(n, angle=theta1 - theta0)

    theta = np.linspace(start=theta0, stop=theta1, num=n)
    x = xy[0] + np.cos(theta) * radius
    y = xy[1] + np.sin(theta) * radius
    return np.array([x, y]).T


def angle_resolution_wrapper(n, angle):

    if isinstance(n, float):
        resolution = n
        n = int(min(abs(angle), np.pi*2) / resolution + 1)
    else:
        assert isinstance(n, int)
    return n


def theta_wrapper(theta0, theta1):
    if theta1 is None:
        theta1 = theta0

    if theta1 < theta0:
        theta1 += 2*np.pi

    theta0 += np.pi * 2
    theta1 += np.pi * 2
    return theta0, theta1


def rectangle(limits: np.ndarray):
    v = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=int)

    if limits is not None:
        v = limits[np.arange(2)[np.newaxis, :].repeat(4, axis=0), v]

    e = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    return v, e


def get_triangle_center(x):
    return x.mean(axis=-2)


def cube(limits: np.ndarray = None) -> (np.ndarray, np.ndarray, np.ndarray):
    v = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=int)

    if limits is not None:
        v = limits[np.arange(3)[np.newaxis, :].repeat(8, axis=0), v]

    # via
    # from itertools import combinations
    # d = 3
    # def get_vi(x):
    #     return np.nonzero((v == x).sum(axis=-1) == 3)[0][0]
    # for s, e in combinations(v, 2):
    #     if np.any(np.sum(np.abs(s - e)) == d):
    #         print(get_vi(s), get_vi(e))
    # for a, b, c, d in combinations(v, 4):
    #     print(a, b, c, d)
    e = np.array([[0, 1], [0, 2], [0, 4],
                  [1, 3], [1, 5],
                  [2, 3], [2, 6],
                  [3, 7],
                  [4, 5], [4, 6],
                  [5, 7],
                  [6, 7]], dtype=int)

    f = np.array([[0, 2, 3, 1],
                  [0, 4, 5, 1],
                  [0, 4, 6, 2],
                  [1, 5, 7, 3],
                  [2, 6, 7, 3],
                  [4, 6, 7, 5]], dtype=int)

    return v, e, f


def faces4_to_3(f4):
    # d - c         c     d - c
    # |   |  =    / |  +  | /
    # a - b     a - b     a
    #
    # abcd   +  abc    +  acd

    assert f4.ndim == 2
    n = len(f4)
    f3 = np.zeros((2*n, 3), dtype=int)
    f3[0::2, :] = f4[:, [0, 1, 2]]
    f3[1::2, :] = f4[:, [0, 2, 3]]
    return f3


def box(limits):
    x = np.array([[limits[0, 0], limits[1, 0]],
                  [limits[0, 1], limits[1, 0]],
                  [limits[0, 1], limits[1, 1]],
                  [limits[0, 0], limits[1, 1]],
                  [limits[0, 0], limits[1, 0]]])
    return x


def fit_plane(x):
    x = np.atleast_2d(x)
    n, _ = x.shape
    assert n >= 3
    centroid = x.mean(axis=0)
    x = x - centroid[np.newaxis, :]

    # Extract the left singular vectors
    left = np.linalg.svd(x.T)[0]

    normal = left[:, -1]

    return centroid, normal


def get_parallel_orthogonal(p: np.ndarray, v: np.ndarray) -> (np.ndarray, np.ndarray):
    parallel = p * (p * v).sum(axis=-1, keepdims=True) / (p * p).sum(axis=-1, keepdims=True)
    orthogonal = v - parallel
    return parallel, orthogonal


def get_orthonormal(v: np.ndarray) -> np.ndarray:
    """
    get a 3d vector which ist orthogonal to v.
    Note that the solution is not unique.
    """
    idx_0 = v == 0
    if np.any(idx_0):
        v_o1 = np.array(idx_0, dtype=float)

    else:
        v_o1 = np.array([1.0, 1.0, 0.0])
        v_o1[-1] = -np.sum(v * v_o1) / v[-1]

    v_o1 /= np.linalg.norm(v_o1)
    return v_o1


def make_rhs(xyz: np.ndarray, order: tuple = (0, 1)) -> np.ndarray:
    # rhs = rand-hand coordinate system
    # xyz -> rhs
    # 1. keep rhs[order[0]]
    # 2. make rhs[order[1]] orthogonal to rhs[order[0]]
    # 3. calculate the third vector as cross product of the first two

    def __normalize(*x):
        return [xx / np.linalg.norm(xx) for xx in x]

    cross_correlation_rhs = np.array([[0, +3, -2],
                                      [-3, 0, +1],
                                      [+2, -1, 0]])
    # rhs[cce[i,j]-1] = cross(rhs[i], rhs[j])
    # ^ with sign
    # xyz = xyz.T

    xyz = np.array(__normalize(*xyz))

    i, j = order
    k = cross_correlation_rhs[i, j]
    k, k_sign = np.abs(k)-1, np.sign(k)
    xyz[j] = __normalize(xyz[j] - xyz[j].dot(xyz[i]) * xyz[i])[0]
    xyz[k] = k_sign*np.cross(xyz[i], xyz[j])
    return xyz


def projection_point_line(p: np.ndarray, x0: np.ndarray, x1: np.ndarray, clip: bool = False) -> np.ndarray:
    """
    http:#mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    Projection from point p to the line defined by {x0 + mu*x1}
    """
    x21 = x1 - x0
    mu = -((x0 - p) * x21).sum(axis=-1) / (x21 * x21).sum(axis=-1)
    if clip:
        mu = np.clip(mu, 0, 1)
    x0_p = x0 + mu * x21
    return x0_p


def distance_point_line(p: np.ndarray, x0: np.ndarray, x1: np.ndarray, clip: bool = False) -> np.ndarray:
    pp = projection_point_line(p=p, x0=x0, x1=x1, clip=clip)
    return np.linalg.norm(pp - p, axis=-1)


def __flip_and_clip_mu(mu: np.ndarray):
    #  x | x | x
    #    0   1
    #  x | 0 | x - 1

    diff_mu = np.zeros_like(mu)
    b = mu < 0
    diff_mu[b] = mu[b]
    b = mu > 1
    diff_mu[b] = mu[b] - 1
    return diff_mu


def __clip_ppp(o: np.ndarray,
               u: np.ndarray,
               v: np.ndarray,
               uu: np.ndarray,
               vv: np.ndarray) -> (np.ndarray, np.ndarray):

    n = np.cross(u, v)
    if u.shape[-1] == 2:
        mua = np.cross(v, o) / n
        mub = -np.cross(u, o) / n
    else:
        nn = (n*n).sum(axis=-1) + 1e-11  # assert you don't divide through zero
        mua = (+n * np.cross(v, o)).sum(axis=-1) / nn
        mub = (-n * np.cross(u, o)).sum(axis=-1) / nn

    uv = (u*v).sum(axis=-1)
    mua2 = mua + uv / uu * __flip_and_clip_mu(mu=mub)
    mub2 = mub + uv / vv * __flip_and_clip_mu(mu=mua)

    mua2 = np.clip(mua2, 0, 1)
    mub2 = np.clip(mub2, 0, 1)
    return mua2, mub2


def projection_point_plane(p: np.ndarray,
                           o: np.ndarray,
                           u: np.ndarray,
                           v: np.ndarray,
                           clip: bool = False) -> np.ndarray:
    """
    Projection of point p on the plane ouv.
    Defined by its origin o and two vectors spanning the plane u and v.
    |
    |          p .
    |            |
    |            v
    |     v ^  - - -
    |      /        /
    |     o ----->
    |           u
    |
    If clip is True, the projection is clipped / projected on the sheet o, o+u, o+u+v, o+v
    """

    o = o - p
    if clip:
        mua, mub = __clip_ppp(o=o, u=u, v=v, uu=(u*u).sum(axis=-1), vv=(v*v).sum(axis=-1))
        return o + mua * u + mub * v + p

    else:
        n = np.cross(u, v)
        p0 = n * (n*o).sum(axis=-1, keepdims=True) / (n*n).sum(axis=-1)
        return p0 + p


def __line_line(x1: np.ndarray, x3: np.ndarray,
                o: np.ndarray, u: np.ndarray, v: np.ndarray, uu: np.ndarray, vv: np.ndarray,
                __return_mu: bool) -> (np.ndarray, np.ndarray):

    mua, mub = __clip_ppp(o=o, u=u, v=-v, uu=uu, vv=vv)  # attention sign change for v

    xa = x1 + mua[..., np.newaxis] * u
    xb = x3 + mub[..., np.newaxis] * v
    if __return_mu:
        return (xa, xb), (mua, mub)
    else:
        return xa, xb


def two_to_three(*args):
    res = []
    for a in args:
        a3 = np.zeros(a.shape[:-1] + (3,))
        a3[..., :2] = a
        res.append(a3)
    return res


def line_line(line_a: np.ndarray, line_b: np.ndarray, __return_mu: bool = False) -> (np.ndarray, np.ndarray):
    """
    (x1-x3) --- (x1-x4)
       |           |
       |           |
    (x2-x3) --- (x2-x4)

     o ---> v
     |
     v
     u

     u = (x2-x3) - (x1-x3) = x2 - x1
     v = (x1-x4) - (x1-x3) = x3 - x4
     o = x1 - x3
    """

    x1, x2 = line_a
    x3, x4 = line_b

    # if x1.shape[-1] == 2:
    #     x1, x2, x3, x4 = two_to_three(x1, x2, x3, x4)

    u = x2 - x1
    v = x4 - x3  # attention, changed sign, to make it consistently AB = B-A
    o = x1 - x3

    uu = (u*u).sum(axis=-1)
    vv = (v*v).sum(axis=-1)
    return __line_line(x1=x1, x3=x3,
                       o=o, u=u, v=v, uu=uu, vv=vv,
                       __return_mu=__return_mu)


def line_line_pairs(lines: np.ndarray, pairs: np.ndarray, __return_mu: bool = False) -> (np.ndarray, np.ndarray):
    a, b = pairs.T
    x1, x3 = lines[..., a, 0, :], lines[..., b, 0, :]
    uv = lines[..., :, 1, :] - lines[..., :, 0, :]
    u = uv[..., a, :]
    v = uv[..., b, :]
    o = x1 - x3

    uuvv = (uv * uv).sum(axis=-1)
    return __line_line(x1=x1, x3=x3,
                       o=o, u=u, v=v, uu=uuvv[..., a], vv=uuvv[..., b],
                       __return_mu=__return_mu)


def line_line_pairs_d2_jac(lines: np.ndarray, pairs: np.ndarray) -> (np.ndarray, np.ndarray):
    (xa, xb), (mua, mub) = line_line_pairs(lines, pairs, __return_mu=False)

    dxaxb_dx = np.zeros(mua.shape + (2, 2))
    dxaxb_dx[..., 0, 0] = +mua - 1
    dxaxb_dx[..., 0, 1] = -mua
    dxaxb_dx[..., 1, 0] = -mub + 1
    dxaxb_dx[..., 0, 1] = +mub

    d = xb - xa
    d2 = (d*d).sum(axis=-1)
    dd2_dx = 2*d * dxaxb_dx
    return d2, dd2_dx


def __line2capsule(xa: np.ndarray,
                   xb: np.ndarray,
                   ra: np.ndarray,
                   rb: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    ra, rb = np.atleast_1d(ra, rb)
    d = xb - xa
    n = np.linalg.norm(d, axis=-1)
    d_n = d / (n[..., np.newaxis] + 1e-9)
    xa = xa + d_n * ra[..., np.newaxis]
    xb = xb - d_n * rb[..., np.newaxis]

    dd = n - ra - rb
    return xa, xb, dd


def capsule_capsule(line_a: np.ndarray,
                    line_b: np.ndarray,
                    radius_a: np.ndarray,
                    radius_b: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    xa, xb = line_line(line_a=line_a, line_b=line_b)
    xa, xb, n = __line2capsule(xa=xa, xb=xb, ra=radius_a, rb=radius_b)
    return xa, xb, n


def capsule_capsule_pairs(lines: np.ndarray,
                          pairs: np.ndarray,
                          radii: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    xa, xb = line_line_pairs(lines=lines, pairs=pairs)
    xa, xb, n = __line2capsule(xa=xa, xb=xb, ra=radii[pairs[:, 0]], rb=radii[pairs[:, 1]])
    return xa, xb, n


def distance_point_plane(p, o, u, v,
                         clip: bool = False) -> np.ndarray:
    pp = projection_point_plane(p=p, o=o, u=u, v=v, clip=clip)
    return np.linalg.norm(pp - p, axis=-1)


def circle_circle_intersection(xy0, r0, xy1, r1):
    """
    https:#stackoverflow.com/a/55817881/7570817
    https:#mathworld.wolfram.com/Circle-CircleIntersection.html

    circle 1: (x0, y0), radius r0
    circle 2: (x1, y1), radius r1
    """

    d = np.linalg.norm(xy1 - xy0)

    # non-intersecting
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(r0 ** 2 - a ** 2)
        d01 = (xy1 - xy0) / d

        xy2 = xy0 + a * d01[::+1] * [+1, +1]
        xy3 = xy2 + h * d01[::-1] * [+1, -1]
        xy4 = xy2 + h * d01[::-1] * [-1, +1]

        return xy3, xy4


def ray_sphere_intersection(rays, spheres, r):
    """
    rays: n x n_rays x 2 x 3    (axis=-2: origin, target)
    spheres: n x n_spheres x 3  (axis=-1: x, y, z)
    r: n_spheres

    return: n x n_rays x n_spheres (boolean array) with res[:, o, j] = True if ray o intersects with sphere j
    Formula from: https:#en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection

    rays = np.random.random((10, 4, 2, 3))
    spheres = np.random.random((10, 5, 3))
    r = np.ones(5) * 0.1
    res = ray_sphere_intersection_2(rays=rays, spheres=spheres, r=r)
    """

    o = rays[..., 0, :]
    u = np.diff(rays, axis=-2)
    u = u / np.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[..., :3]
    if r is None:
        r = spheres[..., 3:].T

    co = (o[..., np.newaxis, :] - c[..., np.newaxis, :, :])
    res = (u * co).sum(axis=-1)**2 - (co**2).sum(axis=-1) + r**2
    return res >= 0


def angle_between_vectors(a, b):
    an = a / np.linalg.norm(a, axis=-1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=-1, keepdims=True)

    angle = arccos2((an * bn).sum(axis=-1))
    return angle


def angle_between_axis_and_point(f, p, axis=2):

    d = (p - f[..., :-1, -1])
    dn = d / np.linalg.norm(d, axis=-1, keepdims=True)

    v = f[..., :-1, axis]

    angle = arccos2((dn * v).sum(axis=-1))
    return angle


def rotation_between_vectors(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    s = np.linalg.norm(v, axis=-1)
    c = (a * b).sum(axis=-1)

    vx = np.zeros(a.shape[:-1] + (3, 3))
    vx[..., 0, 1] = -v[..., 2]
    vx[..., 0, 2] = v[..., 1]
    vx[..., 1, 0] = v[..., 2]
    vx[..., 1, 2] = -v[..., 0]
    vx[..., 2, 0] = -v[..., 1]
    vx[..., 2, 1] = v[..., 0]

    i = np.zeros(a.shape[:-1] + (3, 3))
    i[..., :, :] = np.eye(3)

    r = i + vx + ((1 - c) / s ** 2)[..., np.newaxis, np.newaxis]*(vx @ vx)
    return r


# --- Random -----------------------------------------------------------------------------------------------------------
def sample_spheres(n, r, limits):
    max_iter = 100000
    u = math2.get_upper(n)

    for i in range(max_iter):
        x = random2.random_uniform_ndim(low=limits[:, 0], high=limits[:, 1], shape=n)
        dx = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dxn = np.linalg.norm(dx, axis=-1)
        dxn = dxn[u]

        if np.all(dxn > 2 * r):
            return x

    raise RuntimeError(f"Not {n} spheres in {limits} found")


def sample_points_on_disc(radius, shape=None):
    rho = np.sqrt(np.random.uniform(low=0, high=radius**2, size=shape))
    theta = np.random.uniform(low=0, high=2*np.pi, size=shape)
    xy = np.empty(np.shape(theta)+(2,))
    xy[..., 0] = rho * np.cos(theta)
    xy[..., 1] = rho * np.sin(theta)

    return xy


def sample_points_on_sphere_3d(shape=None):
    shape = np2.shape_wrapper(shape=shape)
    x = np.empty(tuple(shape) + (3,))
    theta = np.random.uniform(low=0, high=2*np.pi, size=shape)
    phi = arccos2(1 - 2 * np.random.uniform(low=0, high=1, size=shape))
    sin_phi = np.sin(phi)
    x[..., 0] = sin_phi * np.cos(theta)
    x[..., 1] = sin_phi * np.sin(theta)
    x[..., 2] = np.cos(phi)

    return x


def sample_points_in_sphere_nd(shape, n_dim: int):
    shape = np2.shape_wrapper(shape=shape)
    r = np.random.uniform(low=0, high=1, size=shape) ** (1/n_dim)
    x = np.random.normal(loc=0, scale=1, size=tuple(shape) + (n_dim,))
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    x = x * r[..., np.newaxis]
    return x


def sample_points_in_ellipse_nd(shape, size):
    n_dim = len(size)
    x = sample_points_in_sphere_nd(shape=shape, n_dim=n_dim)
    x = x * size
    return x


def sample_points_on_sphere_nd(shape, n_dim: int):

    safety = 1.2

    shape = np2.shape_wrapper(shape=shape)
    volume_sphere = hyper_sphere_volume(n_dim)
    volume_cube = 2**n_dim
    safety_factor = int(np.ceil(safety * volume_cube/volume_sphere))

    size_w_n_dim = shape + (n_dim,)
    size_sample = (safety_factor,) + size_w_n_dim

    x = np.random.uniform(low=-1, high=1, size=size_sample)
    x_norm = np.linalg.norm(x, axis=-1)
    bool_keep = x_norm < 1
    n_keep = bool_keep.sum()
    assert n_keep > np.size(shape)
    raise NotImplementedError


def hyper_sphere_volume(n_dim: int, r: float = 1.):
    """https: # en.wikipedia.org / wiki / Volume_of_an_n - ball"""
    n2 = n_dim  # 2
    if n_dim % 2 == 0:
        return (np.pi ** n2) / math.factorial(n2) * r**n_dim
    else:
        return 2*(math.factorial(n2)*(4*np.pi)**n2) / math.factorial(n_dim) * r**n_dim


def get_points_on_circle(x: np.ndarray,
                         r: np.ndarray,
                         n: int = 10,
                         endpoint: bool = False):
    r = np.atleast_1d(r)
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=endpoint)
    sc = np.stack((np.sin(theta), np.cos(theta))).T
    points = x[..., np.newaxis, :] + r[..., np.newaxis, np.newaxis] * sc
    return points


def get_points_on_multicircles(x, r, n=10, endpoint1=False, endpoint2=True):
    points = get_points_on_circle(x=x, r=r, n=n, endpoint=endpoint1)
    hull = ConvexHull(points.reshape(-1, 2))
    if endpoint2:
        i = np.concatenate([hull.vertices, hull.vertices[:1]])
    else:
        i = hull.vertices
    hull = points.reshape(-1, 2)[i]
    return points, hull


def get_points_on_sphere(x=None, r=None, n=None, mode="fibonacci", squeeze=True):

    if x is None:
        x = np.zeros((1, 3))

    if r is None:
        r = 1.

    if n is None:
        n = 100

    x = np.atleast_2d(x)
    r = np.atleast_1d(r)

    if mode == "fibonacci":
        assert isinstance(n, int)
        x = x[:, np.newaxis, :] + r[..., np.newaxis, np.newaxis]*fibonacci_sphere(n=n)[np.newaxis, :, :]
    elif mode == "parametric":
        if len(n) == 2:
            n_phi, n_theta = n
        else:
            n_phi, n_theta = int(np.ceil(np.sqrt(n)))
        phi = np.linspace(0, 2*np.pi, num=n_phi, endpoint=False)
        theta = np.linspace(0, np.pi, num=n_theta, endpoint=False)
        phi, theta = np.meshgrid(phi, theta, indexing="ij")
        xx = np.sin(theta) * np.cos(phi)
        yy = np.sin(theta) * np.sin(phi)
        zz = np.cos(theta)

        x = np.stack((xx, yy, zz), axis=-1)
        x = x*r

    else:
        raise ValueError

    if squeeze and np.size(r) == 1:
        x = x[0]

    return x


def get_points_on_multisphere(x, r, n):
    if isinstance(r, float):
        r = np.full(len(x), r)
    points = get_points_on_sphere(x=x, r=r, n=n)
    hull = ConvexHull(points.reshape(-1, 3))
    return points, hull


def fibonacci_sphere(n: int = 100) -> np.ndarray:  # 3d
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = np.linspace(1, -1, n)
    r = np.sqrt(1 - y*y)              # radius at y
    theta = phi * np.arange(n)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return np.array((x, y, z)).T


def get_points_on_sphere_nd():
    # n=100, d=3
    raise NotImplementedError
    # https://stackoverflow.com/questions/9046106/algorithm-to-rasterize-and-fill-a-hypersphere/21575035#21575035
    # https://math.stackexchange.com/questions/3291489/can-the-fibonacci-lattice-be-extended-to-dimensions-higher-than-3/3297830#3297830
    # https://stackoverflow.com/questions/57123194/how-to-distribute-points-evenly-on-the-surface-of-hyperspheres-in-higher-dimensi


def hcp_grid(limits: np.ndarray, radius: float) -> np.ndarray:
    """
    hexagonal closed packing
    https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
    """

    n_dim = len(limits)
    assert n_dim in (2, 3)
    size = limits[:, 1] - limits[:, 0]

    nx = size[0] // (2 * radius)
    ny = size[1] // (np.sqrt(3) * radius)

    if n_dim == 2:
        i, j = np.ogrid[0:nx, 0:ny]
        x = 2*i + (j % 2)
        y = np.sqrt(3)*j + (0*i)
        hcp = np.concatenate([x[:, :, np.newaxis],
                              y[:, :, np.newaxis]],
                             axis=-1)

    elif n_dim == 3:
        nz = size[2] // (2/3*np.sqrt(6) * radius)

        i, j, k = np.ogrid[0:nx, 0:ny, 0:nz]
        x = 2*i + ((j+k) % 2)
        y = np.sqrt(3)*(j+1/3*(k % 2)) + (0*i)
        z = 2*np.sqrt(6)/3 * k + (0*i*j)
        hcp = np.concatenate([x[:, :, :, np.newaxis],
                              y[:, :, :, np.newaxis],
                              z[:, :, :, np.newaxis]],
                             axis=-1)

    else:
        raise ValueError

    hcp = limits[:, 0] + radius + hcp * radius

    return hcp


def get_distance_to_ellipsoid(x: np.ndarray, shape: np.ndarray) -> np.ndarray:
    assert x.shape[-1] == len(shape)
    d = ((x / shape)**2).sum(axis=-1) - 1
    return d


# mesh
def refine_triangle_mesh(p, f):
    """
    divide each triangle into 3 new triangles
    works for meshes in 2d and 3d
    """
    n_points, d = p.shape
    n_faces, d2 = f.shape
    assert d == d2

    c = get_triangle_center(x=p[f])

    p2 = np.zeros((n_points + n_faces, d))
    p2[:len(p), :] = p
    p2[len(p):, :] = c

    f2 = np.zeros((n_faces*3, d), dtype=int)
    f2[:, 0] = f.ravel()
    f2[:, 1] = np.roll(f, shift=-1, axis=1).ravel()
    f2[:, 2] = np.repeat(np.arange(n_points, n_points+n_faces), 3)

    return p2, f2


def discretize_triangle_mesh(p, f, voxel_size):
    pf = p[f]
    d = np.vstack((np.linalg.norm(pf[..., 0, :] - pf[..., 1, :], axis=-1),
                   np.linalg.norm(pf[..., 0, :] - pf[..., 2, :], axis=-1),
                   np.linalg.norm(pf[..., 1, :] - pf[..., 2, :], axis=-1)))
    d_max = np.max(d, axis=0)
    n = (3*(d_max // voxel_size)).astype(int)
    n, i, c = np.unique(n, return_inverse=True, return_counts=True)

    x2 = []
    for j, nn in enumerate(n):
        x = pf[i == j, :]
        x = discretize_triangle(x=x, n=nn)
        x2.append(x.reshape((-1, 3)))

    x2 = np.concatenate(x2, axis=0)
    return x2


def discretize_triangle(x=None,
                        a=None, b=None, c=None,
                        n=2):
    """there is no ordering in the output"""
    if a is None:
        if n <= 2:
            return x

        a = x[..., 0, :]
        b = x[..., 1, :]
        c = x[..., 2, :]

    else:
        if n <= 2:
            return np.concatenate([a[..., np.newaxis, :], b[..., np.newaxis, :], c[..., np.newaxis, :]], axis=-2)

    *shape, n_dim = a.shape

    u, v = np.meshgrid(*[np.linspace(0, 1, n)]*2, indexing="ij")
    u, v = u[:, :, np.newaxis], v[:, :, np.newaxis]
    a, b, c = a[..., np.newaxis, np.newaxis, :], b[..., np.newaxis, np.newaxis, :], c[..., np.newaxis, np.newaxis, :]

    x2 = u*a + v*b + (1 - u - v)*c
    i = v <= 1 - u
    x2 = x2[..., i[:, :, 0], :]
    return x2.reshape(shape + [i.sum(), n_dim])


def get_x_intersections(x_a, x_b, threshold=0.001, verbose=0):
    if len(x_a) * len(x_b) < 1000000:
        dn_ab = np.linalg.norm(x_a[:, np.newaxis, :] - x_b[np.newaxis, :, :], axis=-1)

        intersection = dn_ab < threshold
        b_a = np.any(intersection, axis=1)
        b_b = np.any(intersection, axis=0)
        return b_a, b_b

    else:
        b_a = np.zeros(len(x_a), dtype=bool)
        b_b = np.zeros(len(x_b), dtype=bool)
        for i in range(len(x_a)):
            if verbose > 0:
                printing.progress_bar(i=i, n=len(x_a), )
            dn_ab = np.linalg.norm(x_a[i, :] - x_b[np.newaxis, :, :], axis=-1)
            intersection = dn_ab < threshold
            b_a[i] = np.any(intersection)
            b_b = np.logical_or(b_b, intersection)

        return b_a, b_b[0]
    

def string_of_pearls2surface(x, r):
    eps0 = 1e-6
    n_arc = 0.05

    # go back and forth over the line
    x = np.concatenate([x, x[-2::-1], x[[1]]], axis=0)
    r = np.concatenate([r, r[-2::-1], r[[1]]], axis=0)

    # calculate the steps
    dx = x[1:] - x[:-1]
    dxn = dx / np.linalg.norm(dx, axis=1, keepdims=True)

    # get the tangential points on the circles
    x4 = get_ortho_star_2d(dxn)
    x0 = x[:-1, np.newaxis, :] + r[:-1, np.newaxis, np.newaxis] * x4
    x1 = x[1:, np.newaxis, :] + r[1:, np.newaxis, np.newaxis] * x4

    # get intersections
    k = np.array([x0[:, 1], x1[:, 1]])
    ki = np.array(line_line(line_a=k[:, :-1], line_b=k[:, 1:]))

    # alternate between arcs and intersections depending on if the angles are obtuse or acute
    p = [k[[0], 0]]
    for i in range(len(r)-2):
        print(np.linalg.norm(ki[0][i] - ki[1][i]))
        if np.linalg.norm(ki[0][i] - ki[1][i]) < eps0:
            p.append(ki[[0], i])

        else:
            theta0 = np.arctan2(x4[i+1, 1, 1], x4[i+1, 1, 0])
            theta1 = np.arctan2(x4[i, 1, 1], x4[i, 1, 0])
            c = get_arc(x[i+1], radius=r[i+1], theta0=theta0, theta1=theta1, n=n_arc)[::-1]
            p.append(c)

    p = np.concatenate(p, axis=0)
    return p


def test_discretize_triangle():
    x0 = np.array([[0, 0],
                  [1, 0],
                  [0, 1]])
    x2 = discretize_triangle(a=x0[0], b=x0[1], c=x0[2], n=10)

    from wzk.mpl2 import new_fig
    fig, ax = new_fig(aspect=1)
    ax.plot(*x0.T, color="blue", marker="o")
    ax.plot(*x2.T, color="red", marker="x")


def test_string_of_pearls2surface():
    from wzk.mpl2 import new_fig, plot_circles
    x = np.array([[0, 0],
                  [0.05, 0],
                  [0.06, 0],
                  [1, 2],
                  [0, 1],
                  [0, 3],
                  [1, 2.5]])
    r = np.array([0.2, 0.19, 0.18, 0.15, 0.1, 0.3, 0.1])

    # x = np.random.random((10, 2))
    # r = np.random.random(10) / 20
    #
    p = string_of_pearls2surface(x, r)
    #
    fig, ax = new_fig(aspect=1)
    ax.plot(*x.T, marker="o", color="k")
    plot_circles(x=x, r=r, ax=ax, alpha=0.1, edgecolor="k", facecolor="none")
    ax.plot(*p.T, color="r")


if __name__ == "__main__":
    test_string_of_pearls2surface()
    # test_discretize_triangle()


# def line_line33(u, v, w):
    # raise NotImplementedError("https://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment()")
    #
    # a = (u*u).sum(axis=-1) # always >= 0
    # b = (u*v).sum(axis=-1)
    # c = (v*v).sum(axis=-1) # always >= 0
    # d = (u*w).sum(axis=-1)
    # e = (v*w).sum(axis=-1)
    # D = a*c - b*b  # always >= 0
    # # sc, sN  # sc = sN / sD, default sD = D >= 0
    # # tc, tN  # tc = tN / tD, default tD = D >= 0
    # sD = D
    # tD = D
    # eps = 1e-6
    # # compute the line parameters of the two closest points
    #
    # i = d > eps
    # sN = np.where(i, b*e - c*d, 0)
    # sD = np.where(i, a*e - b*d, 1)
    # tN = np.where(i, )
    #
    # i = sN < 0
    # sN[i] = 0
    # tD[i] = e
    #
    # tN[i] = e
    #
    # if D < 1e-9:  # the lines are almost parallel
    #     sN = 0.0       # force using point P0 on segment S1
    #     sD = 1.0       # to prevent possible division by 0.0 later
    #     tN = e
    #     tD = c
    #
    # else:                 # get the closest points on the infinite lines
    #     sN = (b*e - c*d)
    #     tN = (a*e - b*d)
    #     if sN < 0.0:  #        # sc < 0 => the s=0 edge is visible
    #         sN = 0.0
    #         tN = e
    #         tD = c
    #     elif sN > sD:  # sc > 1  => the s=1 edge is visible
    #         sN = sD
    #         tN = e + b
    #         tD = c
    #
    # if tN < 0.0:            # tc < 0 => the t=0 edge is visible
    #     tN = 0.0
    #     # recompute sc for this edge
    #     if -d < 0.0:
    #         sN = 0.0
    #     elif -d > a:
    #         sN = sD
    #     else:
    #         sN = -d
    #         sD = a
    #
    #
    # elif tN > tD:     # tc > 1  => the t=1 edge is visible
    #     tN = tD
    #     # recompute sc for this edge
    #     if (-d + b) < 0.0:
    #         sN = 0
    #     elif (-d + b) > a:
    #         sN = sD
    #     else:
    #         sN = (-d +  b)
    #         sD = a
    #
    # # finally do the division to get sc and tc
    # sc = (abs(sN) < SMALL_NUM ? 0.0 : sN / sD)
    # tc = (abs(tN) < SMALL_NUM ? 0.0 : tN / tD)
    #
    # # get the difference of the two closest points
    # dP = w + (sc * u) - (tc * v)  # =  S1(sc) - S2(tc)
    #
    # return norm(dP);   # return the closest distance
