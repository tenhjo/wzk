from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import ConvexHull

from wzk import printing

from ._types import ArrayLike, ShapeLike, float32, int32
from . import np2


def get_ortho_star_2d(x: ArrayLike):
    x = jnp.asarray(x)
    assert x.shape[-1] == 2

    x4 = jnp.zeros(x.shape[:-1] + (4, 2), dtype=x.dtype)
    x4 = x4.at[..., 0, :].set(x)
    x4 = x4.at[..., 1, 0].set(-x[..., 1])
    x4 = x4.at[..., 1, 1].set(+x[..., 0])
    x4 = x4.at[..., 2, 0].set(-x[..., 0])
    x4 = x4.at[..., 2, 1].set(-x[..., 1])
    x4 = x4.at[..., 3, 0].set(+x[..., 1])
    x4 = x4.at[..., 3, 1].set(-x[..., 0])
    return x4


def arccos2(c: ArrayLike):
    c = jnp.clip(c, min=-1, max=+1)
    return jnp.arccos(c)


def get_arc(xy: ArrayLike,
            radius: float,
            theta0: float = 0.0,
            theta1: float = 2 * jnp.pi,
            n: int | float = 0.01):
    theta0, theta1 = theta_wrapper(theta0=theta0, theta1=theta1)
    n = angle_resolution_wrapper(n, angle=theta1 - theta0)

    theta = jnp.linspace(start=theta0, stop=theta1, num=n)
    x = xy[0] + jnp.cos(theta) * radius
    y = xy[1] + jnp.sin(theta) * radius
    return jnp.stack([x, y], axis=-1)


def angle_resolution_wrapper(n: int | float, angle: float) -> int:
    if isinstance(n, float):
        resolution = n
        n = int(min(abs(angle), np.pi * 2) / resolution + 1)
    else:
        assert isinstance(n, int)
    return n


def theta_wrapper(theta0: float, theta1: float | None) -> tuple[float, float]:
    if theta1 is None:
        theta1 = theta0

    if theta1 < theta0:
        theta1 += 2 * np.pi

    theta0 += np.pi * 2
    theta1 += np.pi * 2
    return theta0, theta1


def rectangle(limits: np.ndarray | None):
    v = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=int32)

    if limits is not None:
        v = limits[np.arange(2)[np.newaxis, :].repeat(4, axis=0), v]

    e = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int32)
    return jnp.asarray(v, dtype=int32), jnp.asarray(e, dtype=int32)


def get_triangle_center(x: ArrayLike):
    return jnp.asarray(x).mean(axis=-2)


def cube(limits: np.ndarray | None = None):
    v = np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0],
                  [0, 1, 1],
                  [1, 0, 0],
                  [1, 0, 1],
                  [1, 1, 0],
                  [1, 1, 1]], dtype=int32)

    if limits is not None:
        v = limits[np.arange(3)[np.newaxis, :].repeat(8, axis=0), v]

    e = np.array([[0, 1], [0, 2], [0, 4],
                  [1, 3], [1, 5],
                  [2, 3], [2, 6],
                  [3, 7],
                  [4, 5], [4, 6],
                  [5, 7],
                  [6, 7]], dtype=int32)

    f = np.array([[0, 2, 3, 1],
                  [0, 4, 5, 1],
                  [0, 4, 6, 2],
                  [1, 5, 7, 3],
                  [2, 6, 7, 3],
                  [4, 6, 7, 5]], dtype=int32)

    return jnp.asarray(v, dtype=int32), jnp.asarray(e, dtype=int32), jnp.asarray(f, dtype=int32)


def faces4_to_3(f4: ArrayLike):
    f4 = np.asarray(f4)
    assert f4.ndim == 2
    n = len(f4)
    f3 = np.zeros((2 * n, 3), dtype=int32)
    f3[0::2, :] = f4[:, [0, 1, 2]]
    f3[1::2, :] = f4[:, [0, 2, 3]]
    return jnp.asarray(f3, dtype=int32)


def box(limits: np.ndarray):
    x = np.array([[limits[0, 0], limits[1, 0]],
                  [limits[0, 1], limits[1, 0]],
                  [limits[0, 1], limits[1, 1]],
                  [limits[0, 0], limits[1, 1]],
                  [limits[0, 0], limits[1, 0]]])
    return jnp.asarray(x)


def fit_plane(x: ArrayLike):
    x = np.atleast_2d(np.asarray(x))
    n, _ = x.shape
    assert n >= 3
    centroid = x.mean(axis=0)
    x = x - centroid[np.newaxis, :]
    left = np.linalg.svd(x.T)[0]
    normal = left[:, -1]

    return jnp.asarray(centroid, dtype=float32), jnp.asarray(normal, dtype=float32)


def get_parallel_orthogonal(p: ArrayLike, v: ArrayLike):
    p = jnp.asarray(p)
    v = jnp.asarray(v)
    parallel = p * (p * v).sum(axis=-1, keepdims=True) / (p * p).sum(axis=-1, keepdims=True)
    orthogonal = v - parallel
    return parallel, orthogonal


def get_orthonormal(v: ArrayLike):
    v = np.asarray(v)
    idx_0 = v == 0
    if np.any(idx_0):
        v_o1 = np.array(idx_0, dtype=float32)
    else:
        v_o1 = np.array([1.0, 1.0, 0.0], dtype=float32)
        v_o1[-1] = -1 * np.sum(v * v_o1) / v[-1]

    v_o1 /= np.linalg.norm(v_o1)
    return jnp.asarray(v_o1, dtype=float32)


def make_rhs(xyz: ArrayLike, order: tuple[int, int] = (0, 1)):
    xyz = np.array(xyz, dtype=float32)

    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)

    cross_correlation_rhs = np.array([[0, +3, -2],
                                      [-3, 0, +1],
                                      [+2, -1, 0]])

    i, j = order
    k = cross_correlation_rhs[i, j]
    k, k_sign = np.abs(k) - 1, np.sign(k)
    xyz[j] = xyz[j] - xyz[j].dot(xyz[i]) * xyz[i]
    xyz[j] = xyz[j] / np.linalg.norm(xyz[j])
    xyz[k] = k_sign * np.cross(xyz[i], xyz[j])
    return jnp.asarray(xyz)


def projection_point_line(p: ArrayLike, x0: ArrayLike, x1: ArrayLike, clip: bool = False):
    p = jnp.asarray(p)
    x0 = jnp.asarray(x0)
    x1 = jnp.asarray(x1)
    x21 = x1 - x0
    mu = -((x0 - p) * x21).sum(axis=-1) / (x21 * x21).sum(axis=-1)
    if clip:
        mu = jnp.clip(mu, 0, 1)
    x0_p = x0 + mu[..., jnp.newaxis] * x21
    return x0_p


def distance_point_line(p: ArrayLike, x0: ArrayLike, x1: ArrayLike, clip: bool = False):
    pp = projection_point_line(p=p, x0=x0, x1=x1, clip=clip)
    return jnp.linalg.norm(pp - p, axis=-1)


def __flip_and_clip_mu(mu: ArrayLike):
    diff_mu = jnp.zeros_like(mu)
    diff_mu = jnp.where(mu < 0, mu, diff_mu)
    diff_mu = jnp.where(mu > 1, mu - 1, diff_mu)
    return diff_mu


def __clip_ppp(o: ArrayLike, u: ArrayLike, v: ArrayLike, uu: ArrayLike, vv: ArrayLike):
    n = jnp.cross(u, v)
    if u.shape[-1] == 2:
        mua = jnp.cross(v, o) / n
        mub = -jnp.cross(u, o) / n
    else:
        nn = (n * n).sum(axis=-1) + 1e-11
        mua = (+n * jnp.cross(v, o)).sum(axis=-1) / nn
        mub = (-n * jnp.cross(u, o)).sum(axis=-1) / nn

    uv = (u * v).sum(axis=-1)
    mua2 = mua + uv / uu * __flip_and_clip_mu(mu=mub)
    mub2 = mub + uv / vv * __flip_and_clip_mu(mu=mua)

    mua2 = jnp.clip(mua2, 0, 1)
    mub2 = jnp.clip(mub2, 0, 1)
    return mua2, mub2


def projection_point_plane(p: ArrayLike, o: ArrayLike, u: ArrayLike, v: ArrayLike, clip: bool = False):
    p = jnp.asarray(p)
    o = jnp.asarray(o) - p
    u = jnp.asarray(u)
    v = jnp.asarray(v)

    if clip:
        mua, mub = __clip_ppp(o=o, u=u, v=v, uu=(u * u).sum(axis=-1), vv=(v * v).sum(axis=-1))
        return o + mua[..., jnp.newaxis] * u + mub[..., jnp.newaxis] * v + p

    n = jnp.cross(u, v)
    p0 = n * (n * o).sum(axis=-1, keepdims=True) / (n * n).sum(axis=-1, keepdims=True)
    return p0 + p


def __line_line(x1: ArrayLike,
                x3: ArrayLike,
                o: ArrayLike,
                u: ArrayLike,
                v: ArrayLike,
                uu: ArrayLike,
                vv: ArrayLike,
                _return_mu: bool):
    mua, mub = __clip_ppp(o=o, u=u, v=-v, uu=uu, vv=vv)

    xa = x1 + mua[..., jnp.newaxis] * u
    xb = x3 + mub[..., jnp.newaxis] * v
    if _return_mu:
        return (xa, xb), (mua, mub)
    return xa, xb


def two_to_three(*args: ArrayLike):
    res = []
    for a in args:
        a = jnp.asarray(a)
        a3 = jnp.zeros(a.shape[:-1] + (3,), dtype=a.dtype)
        a3 = a3.at[..., :2].set(a)
        res.append(a3)
    return res


def line_line(line_a: ArrayLike, line_b: ArrayLike, _return_mu: bool = False):
    line_a = jnp.asarray(line_a)
    line_b = jnp.asarray(line_b)

    x1, x2 = line_a
    x3, x4 = line_b

    u = x2 - x1
    v = x4 - x3
    o = x1 - x3

    uu = (u * u).sum(axis=-1)
    vv = (v * v).sum(axis=-1)
    return __line_line(x1=x1, x3=x3,
                       o=o, u=u, v=v, uu=uu, vv=vv,
                       _return_mu=_return_mu)


def line_line_pairs(lines: ArrayLike, pairs: ArrayLike, _return_mu: bool = False):
    lines = jnp.asarray(lines)
    pairs = np.asarray(pairs)
    a, b = pairs.T
    x1, x3 = lines[..., a, 0, :], lines[..., b, 0, :]
    uv = lines[..., :, 1, :] - lines[..., :, 0, :]
    u = uv[..., a, :]
    v = uv[..., b, :]
    o = x1 - x3

    uuvv = (uv * uv).sum(axis=-1)
    return __line_line(x1=x1, x3=x3,
                       o=o, u=u, v=v, uu=uuvv[..., a], vv=uuvv[..., b],
                       _return_mu=_return_mu)


def __line2capsule(xa: ArrayLike, xb: ArrayLike, ra: ArrayLike, rb: ArrayLike):
    ra, rb = np.atleast_1d(np.asarray(ra), np.asarray(rb))
    xa = jnp.asarray(xa)
    xb = jnp.asarray(xb)
    d = xb - xa
    n = jnp.linalg.norm(d, axis=-1)
    d_n = d / (n[..., jnp.newaxis] + 1e-9)
    xa = xa + d_n * ra[..., np.newaxis]
    xb = xb - d_n * rb[..., np.newaxis]

    dd = n - ra - rb
    return xa, xb, dd


def capsule_capsule(line_a: ArrayLike, line_b: ArrayLike, radius_a: ArrayLike, radius_b: ArrayLike):
    xa, xb = line_line(line_a=line_a, line_b=line_b)
    xa, xb, d = __line2capsule(xa=xa, xb=xb, ra=radius_a, rb=radius_b)
    return xa, xb, d


def capsule_capsule_pairs(lines: ArrayLike, pairs: ArrayLike, radii: ArrayLike):
    xa, xb = line_line_pairs(lines=lines, pairs=pairs)
    radii = np.asarray(radii)
    xa, xb, n = __line2capsule(xa=xa, xb=xb, ra=radii[pairs[:, 0]], rb=radii[pairs[:, 1]])
    return xa, xb, n


def distance_point_plane(p: ArrayLike, o: ArrayLike, u: ArrayLike, v: ArrayLike, clip: bool = False):
    pp = projection_point_plane(p=p, o=o, u=u, v=v, clip=clip)
    return jnp.linalg.norm(pp - jnp.asarray(p), axis=-1)


def circle_circle_intersection(xy0: ArrayLike, r0: float, xy1: ArrayLike, r1: float):
    xy0 = np.asarray(xy0)
    xy1 = np.asarray(xy1)
    d = np.linalg.norm(xy1 - xy0)

    if d > r0 + r1:
        return None
    if d < abs(r0 - r1):
        return None
    if d == 0 and r0 == r1:
        return None

    a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r0 ** 2 - a ** 2)
    d01 = (xy1 - xy0) / d

    xy2 = xy0 + a * d01[::+1] * [+1, +1]
    xy3 = xy2 + h * d01[::-1] * [+1, -1]
    xy4 = xy2 + h * d01[::-1] * [-1, +1]

    return jnp.asarray(xy3), jnp.asarray(xy4)


def ray_sphere_intersection(rays: ArrayLike, spheres: ArrayLike, r: ArrayLike | None):
    rays = jnp.asarray(rays)
    spheres = jnp.asarray(spheres)

    o = rays[..., 0, :]
    u = jnp.diff(rays, axis=-2)
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)

    c = spheres[..., :3]
    if r is None:
        r = spheres[..., 3:].T

    co = (o[..., jnp.newaxis, :] - c[..., jnp.newaxis, :, :])
    res = (u * co).sum(axis=-1) ** 2 - (co ** 2).sum(axis=-1) + jnp.asarray(r) ** 2
    return res >= 0


def angle_between_vectors(a: ArrayLike, b: ArrayLike):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    an = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    bn = b / jnp.linalg.norm(b, axis=-1, keepdims=True)

    angle = arccos2((an * bn).sum(axis=-1))
    return angle


def angle_between_axis_and_point(f: ArrayLike, p: ArrayLike, axis: int = 2):
    f = jnp.asarray(f)
    p = jnp.asarray(p)
    d = (p - f[..., :-1, -1])
    dn = d / jnp.linalg.norm(d, axis=-1, keepdims=True)

    v = f[..., :-1, axis]
    angle = arccos2((dn * v).sum(axis=-1))
    return angle


def rotation_between_vectors(a: ArrayLike, b: ArrayLike):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    a = a / jnp.linalg.norm(a, axis=-1, keepdims=True)
    b = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
    v = jnp.cross(a, b)
    s = jnp.linalg.norm(v, axis=-1)
    c = (a * b).sum(axis=-1)

    vx = jnp.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype)
    vx = vx.at[..., 0, 1].set(-v[..., 2])
    vx = vx.at[..., 0, 2].set(v[..., 1])
    vx = vx.at[..., 1, 0].set(v[..., 2])
    vx = vx.at[..., 1, 2].set(-v[..., 0])
    vx = vx.at[..., 2, 0].set(-v[..., 1])
    vx = vx.at[..., 2, 1].set(v[..., 0])

    i = jnp.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype)
    i = i.at[..., :, :].set(jnp.eye(3, dtype=a.dtype))

    factor = ((1 - c) / (s ** 2 + 1e-12))[..., jnp.newaxis, jnp.newaxis]
    r = i + vx + factor * (vx @ vx)
    return r


def _rng_key(key: jax.Array | None = None) -> jax.Array:
    if key is None:
        return jax.random.PRNGKey(np.random.randint(0, 2**31 - 1))
    return key


def sample_points_on_disc(radius: float, shape: ShapeLike | None = None, key: jax.Array | None = None):
    shape = np2.shape_wrapper(shape=shape)
    key = _rng_key(key=key)
    key_rho, key_theta = jax.random.split(key)
    rho = jnp.sqrt(jax.random.uniform(key_rho, shape=shape, minval=0.0, maxval=radius ** 2, dtype=float32))
    theta = jax.random.uniform(key_theta, shape=shape, minval=0.0, maxval=2 * jnp.pi, dtype=float32)
    return jnp.stack((rho * jnp.cos(theta), rho * jnp.sin(theta)), axis=-1)


def sample_points_on_sphere_3d(shape: ShapeLike | None = None, key: jax.Array | None = None):
    shape = np2.shape_wrapper(shape=shape)
    key = _rng_key(key=key)
    key_theta, key_u = jax.random.split(key)
    theta = jax.random.uniform(key_theta, shape=shape, minval=0.0, maxval=2 * jnp.pi, dtype=float32)
    u = jax.random.uniform(key_u, shape=shape, minval=0.0, maxval=1.0, dtype=float32)
    phi = jnp.arccos(jnp.clip(1 - 2 * u, min=-1.0, max=1.0))
    sin_phi = jnp.sin(phi)
    x = jnp.empty(tuple(shape) + (3,), dtype=float32)
    x = x.at[..., 0].set(sin_phi * jnp.cos(theta))
    x = x.at[..., 1].set(sin_phi * jnp.sin(theta))
    x = x.at[..., 2].set(jnp.cos(phi))
    return x


def sample_points_in_sphere_nd(shape: ShapeLike, n_dim: int, key: jax.Array | None = None):
    shape = np2.shape_wrapper(shape=shape)
    key = _rng_key(key=key)
    key_r, key_x = jax.random.split(key)
    r = jax.random.uniform(key_r, shape=shape, minval=0.0, maxval=1.0, dtype=float32) ** (1 / n_dim)
    x = jax.random.normal(key_x, shape=tuple(shape) + (n_dim,), dtype=float32)
    x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    return x * r[..., jnp.newaxis]


def sample_points_in_ellipse_nd(shape: ShapeLike, size: ArrayLike, key: jax.Array | None = None):
    n_dim = len(size)
    x = sample_points_in_sphere_nd(shape=shape, n_dim=n_dim, key=key)
    x = x * jnp.asarray(size)
    return x


def hyper_sphere_volume(n_dim: int, r: float = 1.0) -> float:
    n2 = n_dim
    if n_dim % 2 == 0:
        return (np.pi ** n2) / math.factorial(n2) * r ** n_dim
    return 2 * (math.factorial(n2) * (4 * np.pi) ** n2) / math.factorial(n_dim) * r ** n_dim


def get_points_on_circle(x: ArrayLike, r: ArrayLike, n: int = 10, endpoint: bool = False):
    x = jnp.asarray(x)
    r = jnp.atleast_1d(r)
    theta = jnp.linspace(0, 2 * jnp.pi, num=n, endpoint=endpoint)
    sc = jnp.stack((jnp.sin(theta), jnp.cos(theta)), axis=-1)
    points = x[..., jnp.newaxis, :] + r[..., jnp.newaxis, jnp.newaxis] * sc
    return points


def get_points_on_multicircles(x: ArrayLike, r: ArrayLike, n: int = 10, endpoint1: bool = False, endpoint2: bool = True):
    points = np.asarray(get_points_on_circle(x=x, r=r, n=n, endpoint=endpoint1))
    hull = ConvexHull(points.reshape(-1, 2))
    if endpoint2:
        i = np.concatenate([hull.vertices, hull.vertices[:1]])
    else:
        i = hull.vertices
    hull = points.reshape(-1, 2)[i]
    return jnp.asarray(points), jnp.asarray(hull)


def fibonacci_sphere(n: int = 100):
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = np.linspace(1, -1, n)
    r = np.sqrt(1 - y * y)
    theta = phi * np.arange(n)
    x = np.cos(theta) * r
    z = np.sin(theta) * r
    return jnp.asarray(np.array((x, y, z)).T)


def get_distance_to_ellipsoid(x: ArrayLike, shape: ArrayLike):
    x = jnp.asarray(x)
    shape = jnp.asarray(shape)
    assert x.shape[-1] == len(shape)
    d = ((x / shape) ** 2).sum(axis=-1) - 1
    return d


def get_x_intersections(x_a: ArrayLike,
                        x_b: ArrayLike,
                        threshold: float = 0.001,
                        map_i_ab: bool = True,
                        verbose: int = 0):
    x_a = np.asarray(x_a)
    x_b = np.asarray(x_b)
    if len(x_a) * len(x_b) < 1_000_000:
        dn_ab = np.linalg.norm(x_a[:, np.newaxis, :] - x_b[np.newaxis, :, :], axis=-1)

        intersection = dn_ab < threshold
        i_ab = np.array(np.nonzero(intersection)).T

    else:
        i_ab = np.zeros((0, 2), dtype=int32)
        for i_a in range(len(x_a)):
            if verbose > 0:
                printing.progress_bar(i=i_a, n=len(x_a))
            dn_ab = np.linalg.norm(x_a[i_a, :] - x_b[:, :], axis=-1)
            intersection = dn_ab < threshold

            if np.any(intersection):
                ib_b = np.nonzero(intersection)[0]
                ib_a = np.full(len(ib_b), i_a, int32)
                i_ab = np.concatenate((i_ab, np.vstack((ib_a, ib_b)).T), axis=0, dtype=int32)

    i_a = np.unique(i_ab[:, 0])
    i_b = np.unique(i_ab[:, 1])

    if map_i_ab:
        _, i_ab[:, 0] = np.unique(i_ab[:, 0], return_inverse=True)
        _, i_ab[:, 1] = np.unique(i_ab[:, 1], return_inverse=True)

    return i_a, i_b, i_ab
